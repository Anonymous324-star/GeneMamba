# Standard library imports
import os
import argparse
import pickle
import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions import Bernoulli
import pyarrow as pa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from transformers import Trainer, AutoTokenizer, TrainingArguments, MambaForCausalLM, BertForMaskedLM
from dotmap import DotMap

from mamba_ssm import Mamba2, Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from ..utils import get_tokenizer, MambaTrainer


class Classifier(nn.Module):
    """
    A neural network classifier with configurable layers.
    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        hidden_features (int): Number of hidden features in each hidden layer.
        num_layers (int, optional): Number of layers in the network. Default is 3.
    Attributes:
        layers (nn.ModuleList): List of layers in the network.
    """

    def __init__(self, input_features, output_features, hidden_features, num_layers = 3):
        super(Classifier, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_features, hidden_features))
        self.layers.append(nn.BatchNorm1d(hidden_features))
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.LeakyReLU(0.1))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.BatchNorm1d(hidden_features))
            self.layers.append(nn.Dropout(0.5))
            self.layers.append(nn.LeakyReLU(0.1))
        self.layers.append(nn.Linear(hidden_features, output_features))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        """
        Defines the computation performed at every call.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.

        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class GeneMamba2(nn.Module):

    def __init__(self, config, model_path, tokenizer_path, args):
        """
        Initializes the model with the given configuration, model path, tokenizer path, and additional arguments.
        Args:
            config (object): The configuration object for the model.
            model_path (str): The path to the pre-trained model. If None, a new model will be initialized from scratch.
            tokenizer_path (str): The path to the tokenizer.
            args (object): Additional arguments that may contain mode information.
        Attributes:
            tokenizer (object): The tokenizer object initialized from the tokenizer path.
            vocab_size (int): The size of the vocabulary.
            id2symbol (dict): A dictionary mapping IDs to symbols, loaded from "id2symbol.pkl".
            symbol2id (dict): A dictionary mapping symbols to IDs, loaded from "symbol2id.pkl".
            config (object): The configuration object for the model, updated with vocabulary size and mode.
            mode (str): The mode of the model, either from args or defaulting to "mean".
            model (object): The MambaModel object, either loaded from the pre-trained model path or initialized from scratch.
            device (torch.device): The device on which the model will be run, either CUDA if available or CPU.
        """
        super().__init__()
        
        self.tokenizer = get_tokenizer(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.id2symbol = pickle.load(open(os.path.join(os.path.dirname(__file__), "id2symbol.pkl"), "rb"))
        self.symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))

        config.vocab_size = self.vocab_size
        
        if args != None and args.mode != None:
            config.mode = args.mode
        else:
            config.mode = "mean"

        self.config = config
        self.mode = config.mode
        
        if model_path == None:
            print("Loaded model from scratch")
            self.model = MambaModel(config)
        else:
            print("Loaded model from ", model_path)
            self.model = MambaModel.from_pretrained(model_path, config = config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_ids):
        # call the forward method of the backbone model
        return self.model(input_ids)


    def finetune(self, train_dataset, training_args):
        """
        Fine-tunes the model on the provided training dataset using the specified training arguments.
        Args:
            train_dataset (Dataset): The dataset to be used for training.
            training_args (TrainingArguments): The arguments to configure the training process.
        Returns:
            None
        Raises:
            ValueError: If the tokenizer does not have a CLS token and cannot add one.
        Notes:
            - Adds a CLS token to the tokenizer if it does not already have one.
            - Resizes the model's token embeddings to match the tokenizer's vocabulary size.
            - Uses MambaTrainer to train the model on the provided dataset.
            - Prints a message indicating the completion of the fine-tuning process and the location where the model is saved.
        """
        
        # add the cls token, and adjust the embeddings
        if self.tokenizer.cls_token_id is None:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.vocab_size = len(self.tokenizer)
        self.model.resize_token_embeddings()
        
        trainer = MambaTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = training_args,
            train_dataset = train_dataset,
        )
        trainer.train()
        print(f"Finished finetuning, model saved to {training_args.output_dir}")
    

    def zero_shot(self, input_ids):
        pass
    
    def resize_token_embeddings(self):
        self.model.resize_token_embeddings()

    def get_embedding(self, input_ids):
        return self.model.get_input_embeddings()
    
    def get_gene_embedding(self, input_ids):
        embedding = self.model.get_gene_embedding(input_ids)
        return embedding


class GeneMamba(nn.Module):
    """
    GeneMamba is a neural network model for gene-related tasks, built on top of the MambaForCausalLM architecture.
    Args:
        config (dict): Configuration dictionary for the model.
        model_path (str): Path to the pre-trained model. If None, a new model is initialized.
        tokenizer_path (str): Path to the tokenizer.
        args (Namespace): Additional arguments.
    Attributes:
        args (Namespace): Additional arguments.
        tokenizer (Tokenizer): Tokenizer for processing input text.
        id2symbol (dict): Mapping from gene ensemble IDs to gene symbols.
        symbol2id (dict): Mapping from gene symbols to gene ensemble IDs.
        model (MambaForCausalLM): The underlying MambaForCausalLM model.
        device (torch.device): The device on which the model is running.
    """
    def __init__(self, config, model_path, tokenizer_path, args):
        super(GeneMamba, self).__init__()
        self.args = args
        self.tokenizer = get_tokenizer(tokenizer_path)

        self.id2symbol = pickle.load(open(os.path.join(os.path.dirname(__file__), "id2symbol.pkl"), "rb"))
        self.symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))

        config = MambaForCausalLM.config_class.from_pretrained("state-spaces/mamba-130m-hf")
        config.vocab_size = self.tokenizer.vocab_size

        if model_path == None:
            self.model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
            # modify original architecture to GeneMamba
            self.model.backbone.embeddings = nn.Embedding(self.tokenizer.vocab_size, self.model.config.hidden_size)
            self.model.lm_head = nn.Linear(self.model.config.hidden_size, self.tokenizer.vocab_size, bias = False)
            print("Loaded model from scratch")
        else:
            self.model = MambaForCausalLM.from_pretrained(model_path, config = config, local_files_only = True, ignore_mismatched_sizes = False)
            print("Loaded model from ", model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs
    
    def get_hidden_states(self, input_ids):
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        return hidden_states

    def get_gene_embedding(self, input_ids):
        embedding = self.model.backbone.embeddings(input_ids)
        return embedding

    def _id2symbol(self, gene_ensemble_id):
        return self.id2symbol[gene_ensemble_id]
    
    def _symbol2id(self, gene_symbol):
        return self.symbol2id[gene_symbol]


class GeneMambaFormer(nn.Module):
    def __init__(self, model_path, tokenizer_path, args):
        super(GeneMamba, self).__init__()
        self.args = args
        self.tokenizer = get_tokenizer(tokenizer_path)

        self.id2symbol = pickle.load(open(os.path.join(os.path.dirname(__file__), "id2symbol.pkl"), "rb"))
        self.symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))

        config = MambaForCausalLM.config_class.from_pretrained("state-spaces/mamba-130m-hf")
        config.vocab_size = self.tokenizer.vocab_size

        if model_path == None:
            self.model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
            self.model.backbone.embeddings = nn.Embedding(self.tokenizer.vocab_size, self.model.config.hidden_size)
            self.model.lm_head = nn.Linear(self.model.config.hidden_size, self.tokenizer.vocab_size, bias = False)
            print("Loaded model from scratch")
        else:
            self.model = MambaForCausalLM.from_pretrained(model_path, config = config, local_files_only = True, ignore_mismatched_sizes = False)
            print("Loaded model from ", model_path)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs
    
    def get_hidden_states(self, input_ids):
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        return hidden_states

    def get_gene_embedding(self, input_ids):
        embedding = self.model.backbone.embeddings(input_ids)
        return embedding

    def _id2symbol(self, gene_ensemble_id):
        return self.id2symbol[gene_ensemble_id]
    
    def _symbol2id(self, gene_symbol):
        return self.symbol2id[gene_symbol]



class GeneMambaForCellAnnotation(GeneMamba):
    def __init__(self, model_path, tokenizer_path, is_finetune, args, output_dim_cls, hidden_dim, num_layers_cls):
        
        # initialize the GeneMamba model
        super().__init__(model_path, tokenizer_path, args)

        self.model = self.model.backbone
        self.is_finetune = is_finetune
        self.args = args
        self.id2symbol = pickle.load(open(os.path.join(os.path.dirname(__file__), "id2symbol.pkl"), "rb"))
        self.symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))

        # if not finetuning, freeze the backbone parameters
        if not self.is_finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # add a classifier on top of the backbone
        self.classifier = Classifier(self.model.config.hidden_size, output_dim_cls, hidden_dim, num_layers_cls)
        self.classifier.to(self.device)
    
    
    def forward(self, input_ids):
        """
        Forward method for the model.
        Passes input through the GeneMamba backbone and then through the classifier.
        """

        # pass through the GeneMamba backbone
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        print("hidden_states shape: ", hidden_states.shape)
        # global average pooling on hidden states to get a fixed-size representation
        pooled_output = hidden_states.mean(dim=1)
        # pass the pooled representation to the classifier
        logits = self.classifier(pooled_output)
        print("logits shape: ", logits.shape)

        return logits
    
    def save_classifier(self, classifier_path):
        torch.save(self.classifier.state_dict(), classifier_path)
    
    def load_classifier(self, classifier_path):
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(self.device)
        

class GeneMambaForGeneClassification(GeneMamba):
    def __init__(self, model_path, tokenizer_path, is_finetune, args, output_dim_cls, hidden_dim, num_layers_cls):
        
        # Initialize the GeneMamba model
        super().__init__(model_path, tokenizer_path, args)

        # Extract the backbone from the GeneMamba model, remove the lm_head layer
        self.model = self.model.backbone
        self.is_finetune = is_finetune
        self.args = args

        # if not finetuning, freeze the backbone parameters
        if not self.is_finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Add a classifier on top of the backbone
        self.classifier = Classifier(self.model.config.hidden_size, output_dim_cls, hidden_dim, num_layers_cls)
        self.classifier.to(self.device)
    
    
    def forward(self, input_ids):
        """
        Forward method for the model.
        Passes input through the GeneMamba backbone and then through the classifier.
        """

        # Pass through the GeneMamba backbone (e.g., extract hidden states)
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        
        # global average pooling on hidden states to get a fixed-size representation
        pooled_output = hidden_states.mean(dim=1)  # Average over the sequence dimension
        # print("pooled_output shape: ", pooled_output.shape)
        
        # Pass the pooled representation to the classifier
        logits = self.classifier(pooled_output)

        return logits


    def get_hidden_states(self, input_ids):
        """
        Forward method for the model.
        Passes input through the GeneMamba backbone and then through the classifier.
        """

        # Pass through the GeneMamba backbone (e.g., extract hidden states)
        outputs = self.model(input_ids)
        hidden_states = outputs.last_hidden_state
        
        return hidden_states



    def save_classifier(self, classifier_path):
        torch.save(self.classifier.state_dict(), classifier_path)
    
    def load_classifier(self, classifier_path):
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(self.device)

    

class EncoderLayer(nn.Module):
    """
    EncoderLayer is a neural network module that applies a Mamba2 layer followed by a residual connection.

    Args:
        input_dim (int): The dimension of the input features.

    Attributes:
        mamba (Mamba2): The Mamba2 layer used for processing the input features.
    """
    def __init__(self, input_dim):
        super(EncoderLayer, self).__init__()
        self.mamba = Mamba2(d_model=input_dim, d_state=64, d_conv=4, expand=2)

    def forward(self, X):
        # ssm forward pass + residual connection
        output = self.mamba(X) + X
        return X

class mamba_mixer(nn.Module):
    """
    A neural network module that applies a series of encoder layers to input data, 
    with support for different aggregation modes.
    Args:
        mode (str): The aggregation mode to use. Options are "mean", "concat", "sum", and "gate".
        d_model (int, optional): The dimension of the model. Default is 512.
        mamba_layer (int, optional): The number of encoder layers. Default is 24.
    """
    def __init__(self, mode, d_model=512, mamba_layer=24):
        super(mamba_mixer, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model) for _ in range(mamba_layer)])
        self.mode = mode
        
        if mode == "concat" or mode == "gate":
            self.aggr = torch.nn.Linear(d_model * 2, d_model)

    def flip_sc(self, X, mask):
        """
        Reverses the sequence of the input tensor X based on the provided mask.
        Args:
            X (torch.Tensor): The input tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor): The padding mask tensor of shape (batch_size, seq_length).
        Returns:
            torch.Tensor: The reversed input tensor.
        """
        batch_size, seq_length, embedding_dim = X.size()
        lengths = (~mask).sum(dim=1)
        pos_tensor = torch.arange(seq_length, device=X.device).unsqueeze(0).expand(batch_size, -1)
        flip_mask = pos_tensor < lengths.unsqueeze(1)
        reversed_positions = torch.where(flip_mask, lengths.unsqueeze(1) - 1 - pos_tensor, pos_tensor)

        # use gather to reverse the sequence for higher efficiency
        X_reverse= torch.gather(X, 1, reversed_positions.unsqueeze(-1).expand(-1, -1, embedding_dim))

        return X_reverse
    
    
    def forward(self, X, padding_mask=None):
        """
        Applies the encoder layers to the input tensor X and aggregates the results 
        based on the specified mode.
        Args:
            X (torch.Tensor): The input tensor of shape (batch_size, seq_length, embedding_dim).
            padding_mask (torch.Tensor, optional): The padding mask tensor of shape (batch_size, seq_length). 
                                                    Default is None.
        Returns:
            torch.Tensor: The output tensor after applying the encoder layers and aggregation.
        """
        
        for i in range(len(self.layers)):
            if (padding_mask is not None):
                X_flip = self.flip_sc(X, padding_mask)
            else:
                X_flip = X.flip([1])
            
            X_f = self.layers[i](X)
            X_b = self.layers[i](X_flip)

            if padding_mask is not None:
                X_b = self.flip_sc(X_b, padding_mask)
            else:
                X_b = X_b.flip([1])
            
            if self.mode == "mean":
                X = (X_f + X_b) / 2

            elif self.mode == "concat":
                X = torch.cat([X_f, X_b], dim=-1)
                X = self.aggr(X)

            elif self.mode == "sum":
                X = X_f + X_b

            elif self.mode == "gate":
                z = torch.sigmoid(self.aggr(torch.cat([X_f, X_b], dim=-1)))
                X = z * X_f + (1 - z) * X_b
            else:
                raise ValueError("Invalid mode")
                
        return X



from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel

class MambaModel(PreTrainedModel):
    
    # define the legal input names for the base class
    model_input_names = ["input_ids", "padding_mask"]

    def __init__(self, config, **kwargs):
        """
        Initializes the model with the given configuration.
        Args:
            config (object): Configuration object containing model parameters.
            **kwargs: Additional keyword arguments.
        Attributes:
            d_model (int): Dimensionality of the model.
            mamba_layer (int): Number of Mamba layers.
            vocab_size (int): Size of the vocabulary.
            hidden_dim (int): Dimensionality of the hidden layers.
            mode (str): Mode of the model.
            embeddings (nn.Embedding): Embedding layer.
            mamba_mixer (mamba_mixer): Mamba mixer layer.
            norm_f (RMSNorm): Normalization layer.
            lm_head (nn.Linear): Linear layer for language modeling head.
        """

        super().__init__(config)
        self.d_model = config.d_model
        self.mamba_layer = config.mamba_layer
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.d_model
        self.mode = config.mode
        
        self.embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.mamba_mixer = mamba_mixer(self.mode, self.d_model, self.mamba_layer)
        
        self.norm_f = RMSNorm(self.d_model)
        
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

    
    def forward(self, input_ids, padding_mask = None):
        """
        Perform a forward pass through the model.
        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs.
            padding_mask (torch.Tensor, optional): Tensor containing padding mask. Defaults to None.
        Returns:
            SequenceClassifierOutput: An object containing the logits and hidden states.
        """
        
        output = {}
        outputs = self.embeddings(input_ids)
        outputs = self.mamba_mixer(outputs, padding_mask)
        outputs = self.norm_f(outputs)
        logits = self.lm_head(outputs)

        return SequenceClassifierOutput(
            logits = logits,
            hidden_states = outputs
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """
        Load a pre-trained model from a checkpoint.
        Args:
            checkpoint_path (str): The path to the directory containing the model checkpoint.
            **kwargs: Additional keyword arguments, including:
                - config (dict): Configuration dictionary for initializing the model.
        Returns:
            model: The model instance with weights loaded from the checkpoint.
        """
        
        # initialize the model
        model = cls(kwargs.get('config'))

        # load weights from safetensors
        from safetensors.torch import load_file
        state_dict = load_file(f"{checkpoint_path}/model.safetensors")

        model.load_state_dict(state_dict)

        return model
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings = value
        self.embeddings = self.embeddings.to(self.device)
    
    def resize_token_embeddings(self):
        """
        Resize the token embeddings and the language model (lm) head for fine-tuning.
        """
        # resize the embeddings for fine-tuning
        old_embeddings = self.get_input_embeddings()
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_embeddings = nn.Embedding(old_num_tokens + 1, old_embedding_dim)
        
        new_embeddings.weight.data[:old_num_tokens, :] = old_embeddings.weight.data

        new_embeddings.weight.data[-1] = torch.mean(old_embeddings.weight.data, dim=0)
        self.set_input_embeddings(new_embeddings)
        
        # reshape the lm_head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(old_embedding_dim, old_num_tokens + 1)
        new_lm_head.weight.data[:old_num_tokens, :] = old_lm_head.weight.data
        new_lm_head.weight.data[-1] = torch.mean(old_lm_head.weight.data, dim=0)
        self.lm_head = new_lm_head

    def get_gene_embedding(self, input_ids):
        return self.embeddings(input_ids)


class GeneTransformer(nn.Module):

    def __init__(self, config, model_path, tokenizer_path, args):
        super().__init__()
        
        self.tokenizer = get_tokenizer(tokenizer_path)

        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            print(f"Added mask token: {self.tokenizer.mask_token}, id = {self.tokenizer.mask_token_id}")
            
        self.vocab_size = self.tokenizer.vocab_size
        self.id2symbol = pickle.load(open(os.path.join(os.path.dirname(__file__), "id2symbol.pkl"), "rb"))
        self.symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path == None:
            print("Loaded model from scratch")
            self.model = BertForMaskedLM(config)

        else:
            print("Loaded model from ", model_path)
            self.model = BertForMaskedLM.from_pretrained(model_path)
            print(self.model)
        

        self.model.to(self.device)

    def forward(self, input_ids, return_logits = False):
        return self.model(input_ids)


    def finetune(self, train_dataset, training_args):

        trainer = MambaTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = training_args,
            train_dataset = train_dataset,
        )
        trainer.train()
        print(f"Finished finetuning, model saved to {training_args.output_dir}")
    
    def get_embedding(self, input_ids):
        return self.model.get_input_embeddings()


class GeneMamba2ForCellClassification(GeneMamba2):
    def __init__(self, config, model_path, tokenizer_path, args, output_dim_cls, hidden_dim, num_layers_cls):
        super().__init__(config, model_path, tokenizer_path, args)
        self.output_dim_cls = output_dim_cls
        self.num_layers_cls = num_layers_cls
        
        self.classifier = Classifier(self.model.config.d_model, output_dim_cls, hidden_dim, num_layers_cls)
        self.classifier.to(self.device)

        self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        self.vocab_size = len(self.tokenizer)
        self.model.resize_token_embeddings()
    
    def forward(self, input_ids, padding_mask, return_cls = False):
        outputs = self.model(input_ids)
        cls_representation = outputs.hidden_states[:, 0, :]
        logits = self.classifier(cls_representation)

        if return_cls:
            return logits, cls_representation
        else:
            return logits
    
    def predict(self, input_ids, padding_mask):
        logits = self.forward(input_ids, padding_mask)
        return torch.argmax(logits, dim = 1)

    def save_classifier(self, classifier_path):
        torch.save(self.classifier.state_dict(), classifier_path)
    
    def load_classifier(self, classifier_path):
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(self.device)