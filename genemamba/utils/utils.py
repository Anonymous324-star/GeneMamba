# Standard library imports
import os

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.functional import normalize

import numpy as np
import pandas as pd
import pyarrow as pa
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

from tqdm import tqdm, trange
from dotmap import DotMap
import pickle

from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    MambaForCausalLM,
)

from datasets import load_from_disk



class CellDataset(Dataset):
    """
    A custom dataset class for handling cell data.

    Args:
        cells (list): A list of cell data, where each cell is a dictionary of tensors.
        tokenizer (Tokenizer): A tokenizer instance used for processing the cell data.

    Attributes:
        cells (list): Stores the list of cell data.
        tokenizer (Tokenizer): Stores the tokenizer instance.

    Methods:
        __len__(): Returns the number of cells in the dataset.
        __getitem__(idx): Retrieves the cell data at the specified index.

    Example:
        >>> dataset = CellDataset(cells, tokenizer)
        >>> len(dataset)
        100
        >>> dataset[0]
        {'input_ids': tensor([...]), 'attention_mask': tensor([...])}
    """

    def __init__(self, cells, tokenizer):
        # self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.cells = cells
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.cells[idx].items()}
        return item


def read_arrow_file(path_to_arrow_data, bulk_id, num_samples, read_batch_size):
    """
    Reads data from an Arrow file in batches and returns it as a pandas DataFrame.

    Parameters:
        path_to_arrow_data (str): The file path to the Arrow data.
        bulk_id (int): The bulk identifier used to calculate the start and end points for reading.
        num_samples (int): The number of samples to read.
        read_batch_size (int): The size of each batch to read.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the read data.
    """
    with open(path_to_arrow_data, 'rb') as f:
        reader = pa.ipc.RecordBatchStreamReader(f)
        i = 0
        df = None

        start_index = (bulk_id * 1000000) // read_batch_size
        end_index = (bulk_id * 1000000 + num_samples) // read_batch_size
        
        total_batches = end_index - start_index

        with tqdm(total=total_batches * read_batch_size, desc=f"Loading cells from {start_index * read_batch_size} to {end_index * read_batch_size}") as pbar:
            for batch in reader:
                if i >= start_index:
                    batch_df = batch.to_pandas()

                    if df is None:
                        df = batch_df
                    else:
                        df = pd.concat([df, batch_df], ignore_index=True)

                    # update progress bar
                    pbar.update(read_batch_size)

                i += 1

                if i >= end_index:
                    # print(f"Loaded {(i * read_batch_size) - bulk_id * 1000000} cells")
                    break

        return df

def load_arrow_file(path_to_arrows_folder, bulk_id, num_samples, read_batch_size):
    """
    Load data from an Arrow file in batches and return it as a pandas DataFrame.

    Parameters:
        path_to_arrows_folder (str): The path to the folder containing the Arrow files.
        bulk_id (int): The bulk identifier used to calculate the start and end indices for reading.
        num_samples (int): The total number of samples to read from the Arrow file.
        read_batch_size (int): The number of samples to read in each batch.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    dataset = load_from_disk(path_to_arrows_folder)
    start_index = (bulk_id * 1000000) // read_batch_size
    end_index = (bulk_id * 1000000 + num_samples) // read_batch_size

    df = None

    for i in trange(0, num_samples, read_batch_size):
        batch_df = pd.DataFrame(dataset[start_index + i: start_index + i + read_batch_size])
        if df is None:
            df = batch_df
        else:
            df = pd.concat([df, batch_df], ignore_index=True)

    return df


def build_dataset(path_to_arrow_data, tokenizer, args):
    """
    Builds a dataset for pretraining from Arrow data.
    
    Args:
        path_to_arrow_data (str): Path to the Arrow file containing the data.
        tokenizer (PreTrainedTokenizer): Tokenizer to use for processing the text data.
        args (Namespace): Arguments containing the following attributes:
            - bulk_id (str): Identifier for the bulk data.
            - num_samples (int): Number of samples to read from the Arrow file.
            - seq_len (int): Sequence length for the input data.
    
    Returns:
        CellDataset: A dataset object containing the preprocessed data.
    """
    
    df = read_arrow_file(path_to_arrow_data, args.bulk_id, args.num_samples, 1000)

    result = []

    for i, text in tqdm(enumerate(df["input_ids"]), total=len(df["input_ids"]), desc = "Building pretrain dataset"):
        # result.append({"input_ids": torch.tensor(text[:args.seq_len])})
        if len(text) >= args.seq_len:
            result.append({"input_ids": torch.tensor(text[:args.seq_len])})
        else:
            result.append({"input_ids": torch.tensor(np.hstack((text, [tokenizer.pad_token_id] * (args.seq_len - len(text)))))})

    sample_dataset = CellDataset(result, tokenizer)

    return sample_dataset

def build_test_dataset(path_to_arrow_data, tokenizer, args):
    
    df = read_arrow_file(path_to_arrow_data, args.bulk_id, args.num_test_samples, 1000)

    result = []

    for i, text in tqdm(enumerate(df["input_ids"]), total=len(df["input_ids"]), desc = "Building test dataset"):
        if len(text) >= args.seq_len:
            result.append({"input_ids": torch.tensor(text[:args.seq_len])})
        else:
            result.append({"input_ids": torch.tensor(np.hstack((text, [tokenizer.pad_token_id] * (args.seq_len - len(text)))))})

    sample_dataset = CellDataset(result, tokenizer)

    return sample_dataset


def build_downstream_dataset(input_data, tokenizer):

    result = []

    for i in range(input_data.shape[0]):
        result.append({"input_ids": torch.from_numpy(input_data[i])})
    
    
    sample_dataset = CellDataset(result, tokenizer)
    return sample_dataset


def get_tokenizer(tokenizer_path):
    """
    Loads a pre-trained tokenizer from the specified file path and sets special tokens.

    Args:
        tokenizer_path (str): The file path to the pre-trained tokenizer.

    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer with special tokens set.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    # only for the transformer based model
    # if tokenizer.mask_token is None:
    #     tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    return tokenizer

class GFTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits

        if loss is None:
            raise ValueError("Loss is None. Check the model and input data.")

        return (loss, logits) if return_outputs else loss



class MambaTrainer(Trainer):
    def __init__(self, *args, coexpression_file=None, **kwargs):
        """
        Custom Trainer with preloaded tokenizer and gene-to-token ID mapping.
        """
        super().__init__(*args, **kwargs)
        print("Coexpression file:", coexpression_file)
        if coexpression_file is not None:
            self.coexpression_file = coexpression_file
        else:
            # check if the file exists
            if os.path.exists(os.path.join(os.path.dirname(__file__), "gene_pairs_50000.txt")):
                self.coexpression_file = os.path.join(os.path.dirname(__file__), "gene_pairs_50000.txt")
            else:
                raise FileNotFoundError("Co-expression file not found.")

        self.gamma = 0.1 # hyperparameter for trade-off between LM loss and InfoNCE loss

        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.special_tokens = {self.pad_token_id, self.unk_token_id}

        self.gene_name_to_token_id = self.create_gene_to_token_mapping(coexpression_file)

        self.coexp_matrix = self.load_sparse_coexp_matrix(coexpression_file)
    
    def log(self, logs):
        # customize logging precision
        if "loss" in logs:
            logs["loss"] = f"{logs['loss']:.10f}" 
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Efficient compute_loss using preloaded gene-to-token mapping and sparse co-expression matrix.
        """

        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.size()
        outputs = model(input_ids=input_ids)

        lm_logits = outputs.logits
        labels = input_ids.to(lm_logits.device)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()  # shift labels to match shifted logits

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        

        hidden_states = outputs.hidden_states  # hidden states (batch_size, seq_len, hidden_dim)

        hidden_states_flat = hidden_states.view(batch_size * seq_len, -1)

        hidden_states_norm = normalize(hidden_states_flat, p=2, dim=-1)

        flat_input_ids = input_ids.view(-1)

        special_tokens_tensor = torch.tensor(list(self.special_tokens), device=flat_input_ids.device)

        valid_mask = ~torch.stack([flat_input_ids == token for token in special_tokens_tensor]).any(dim=0)

        valid_input_ids = flat_input_ids[valid_mask]
        valid_hidden_states = hidden_states_norm[valid_mask]

        valid_gene_indices = valid_input_ids.cuda()

        sparse_indices = self.coexp_matrix._indices()  # non-zero indices
        sparse_values = self.coexp_matrix._values()    # non-zero values

        source_mask = torch.isin(sparse_indices[0], valid_gene_indices)
        target_mask = torch.isin(sparse_indices[1], valid_gene_indices)
        valid_mask = source_mask & target_mask

        filtered_indices = sparse_indices[:, valid_mask] 
        filtered_values = sparse_values[valid_mask]

        gene_to_local = {gene.item(): idx for idx, gene in enumerate(valid_gene_indices)}
        local_source_indices = torch.tensor([gene_to_local[gene.item()] for gene in filtered_indices[0]])
        local_target_indices = torch.tensor([gene_to_local[gene.item()] for gene in filtered_indices[1]])

        source_embeddings = valid_hidden_states[local_source_indices]
        target_embeddings = valid_hidden_states[local_target_indices]
        
        sims = torch.sum(source_embeddings * target_embeddings, dim=-1)  # Shape: (num_valid,)

        temperature = 0.1  # adjust as needed
        scaled_sims = sims / temperature

        positive_pairs = filtered_values == 1
        logits = scaled_sims  # cosine similarities scaled by temperature

        all_pairs = torch.arange(len(logits), device=logits.device)
        denominator = torch.logsumexp(logits[all_pairs], dim=0)

        info_nce_loss = -torch.sum(logits[positive_pairs] - denominator) / positive_pairs.sum()

        total_loss = lm_loss + self.gamma * info_nce_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def create_gene_to_token_mapping(self, coexpression_file):
        """
        Create a mapping from gene names to token IDs using the tokenizer.
        """
        gene_names = set()

        with open(coexpression_file, "r") as f:
            for line in f:
                gene1, gene2, _ = line.strip().split(",")
                gene_names.add(gene1)
                gene_names.add(gene2)

        gene_name_to_token_id = {}
        symbol2id = pickle.load(open(os.path.join(os.path.dirname(__file__), "symbol2id.pkl"), "rb"))
        for gene_name in gene_names:
            token_id = self.tokenizer.convert_tokens_to_ids(symbol2id.get(gene_name, "<unk>"))
            gene_name_to_token_id[gene_name] = token_id

        return gene_name_to_token_id

    def load_sparse_coexp_matrix(self, coexpression_file):
        """
        Create a sparse co-expression matrix using the gene-to-token mapping.
        """
        indices = []
        values = []

        with open(coexpression_file, "r") as f:
            for line in f:
                gene1, gene2, label = line.strip().split(",")
                if gene1 in self.gene_name_to_token_id and gene2 in self.gene_name_to_token_id:
                    idx1 = self.gene_name_to_token_id[gene1]
                    idx2 = self.gene_name_to_token_id[gene2]
                    indices.append([idx1, idx2])
                    values.append(float(label))  # Label should be 0 or 1


        indices = torch.tensor(indices).t()  # shape: (2, num_nonzero)
        values = torch.tensor(values)        # shape: (num_nonzero,)

        num_tokens = max(self.gene_name_to_token_id.values()) + 1
        coexp_matrix = torch.sparse_coo_tensor(indices, values, size=(num_tokens, num_tokens), device='cuda')

        return coexp_matrix


def get_last_checkpoint(output_dir):
    """
    Retrieves the path to the last checkpoint in the specified output directory.

    This function navigates through the directory structure to find the most recent checkpoint
    based on the bulk ID and number of epochs. It handles cases where the bulk ID is 0 and adjusts
    the directory path accordingly to find the last checkpoint from the previous epoch if necessary.

    Args:
        output_dir (str): The directory where checkpoints are stored. The directory structure is expected
                          to follow the pattern: /base_dir/num_epochs/bulk_idm/

    Returns:
        str: The path to the last checkpoint file if found, otherwise None.

    Raises:
        Exception: If no checkpoints are found in the specified directory.
    """
    checkpoint_parent_dir = output_dir
    base_dir = "/".join(checkpoint_parent_dir.split("/")[:-2])
    bulk_id = checkpoint_parent_dir.split("/")[-1].replace("m", "")
    num_epochs = checkpoint_parent_dir.split("/")[-2]
    print(f"bulk_id: {bulk_id}, num_epochs: {num_epochs}")
    if int(bulk_id) == 0 and int(num_epochs) == 1:
        return None
    if int(bulk_id) == 0 and int(num_epochs) > 1:
        checkpoint_folders = os.listdir(os.path.join(base_dir, str(int(num_epochs) - 1)))
        last_bulk_id = max([int(f.split("m")[0]) for f in checkpoint_folders])
        checkpoint_parent_dir = base_dir + "/" + str(int(num_epochs) - 1) + "/" + str(last_bulk_id) + "m"

    checkpoints = [d for d in os.listdir(checkpoint_parent_dir) if d.startswith("checkpoint")]
    # sort checkpoints based on the step number
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        last_checkpoint = os.path.join(checkpoint_parent_dir, checkpoints[-1])  # get the latest checkpoint
        print(f"Loading the last checkpoint from: {last_checkpoint}")
    else:
        raise Exception(f"No checkpoints found at {checkpoint_parent_dir}.")
        last_checkpoint = None

    return last_checkpoint



def prepare_data(adata, model, dataset_name, max_genes = 2048):
    """
    Prepares the data for the GeneMamba model by encoding gene names, permuting gene IDs based on expression values, 
    and encoding cell type labels.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    model : object
        The model object which contains a tokenizer for encoding gene names.
    dataset_name : str
        The name of the dataset being processed.
    max_genes : int, optional
        The maximum number of genes to include in the input data (default is 2048).
    Returns:
    --------
    adata : AnnData
        The input annotated data matrix with potentially modified cell type labels.
    input_data : np.ndarray
        The permuted gene IDs based on expression values, limited to `max_genes`.
    y : np.ndarray
        The original cell type labels.
    y_numerical : np.ndarray
        The numerical encoded cell type labels.
    num_classes : int
        The number of unique cell type classes.

    """
    if "celltype" not in adata.obs:
        adata.obs["celltype"] = adata.obs["str_labels"]

    y = np.array(adata.obs['celltype'].values.tolist())
    
    label_encoder = LabelEncoder()
    y_numerical = label_encoder.fit_transform(y)
    
    num_classes = len(label_encoder.classes_)

    """
        First convert to Ensemble gene ids, then permute the gene ids based on the expression values.
    """

    # map the permuted gene ids to the GeneMamba gene ids
    import pickle
    current_directory = os.path.dirname(os.path.abspath(__file__))
    symbol2id = pickle.load(open(os.path.join(current_directory, "../models/symbol2id.pkl"), "rb"))

    mapped_gene_ids = []
    for i in range(adata.X.shape[1]):
        gene_name = adata.var.index[i]
        if dataset_name == "pbmc12k":
            mapped_gene_ids.append(model.tokenizer.encode(gene_name)[0])
        elif dataset_name != "pbmc12k" and gene_name in symbol2id:
            mapped_gene_ids.append(model.tokenizer.encode(symbol2id[gene_name])[0])
        else:
            mapped_gene_ids.append(model.tokenizer.encode(gene_name)[0])
    # print(mapped_gene_ids)
    gene_ids = np.array(mapped_gene_ids)
    permuted_gene_ids = []

    for i in range(adata.X.shape[0]):
        if isinstance(adata.X, csr_matrix):
            expression_values = adata.X[i].toarray().flatten()
        else:
            expression_values = adata.X[i].flatten()
        # expression_values = adata.X[i].toarray().flatten()  # Convert sparse matrix to dense
        sorted_indices = np.argsort(-expression_values)  # Sort in descending order
        permuted_gene_ids.append(gene_ids[sorted_indices])

    input_data = np.array(permuted_gene_ids)[:, :max_genes]

    return adata, input_data, y, y_numerical, num_classes


def calculate_AvgBIO(cell_representation, y_numerical, y_test, y_pred_class, return_pred = True):
    
    num_classes = len(np.unique(y_numerical))
    kmeans = KMeans(n_clusters=num_classes, random_state=12).fit(cell_representation)
    cluster_labels = kmeans.labels_

    ari = adjusted_rand_score(y_numerical, cluster_labels)
    nmi = normalized_mutual_info_score(y_numerical, cluster_labels)
    asw = (silhouette_score(cell_representation, cluster_labels) + 1) / 2
    
    AvgBIO = (ari + nmi + asw) / 3

    ari_pred = adjusted_rand_score(y_test, y_pred_class)
    nmi_pred = normalized_mutual_info_score(y_test, y_pred_class)
    asw_pred = (silhouette_score(cell_representation, y_numerical) + 1) / 2

    AvgBIO_pred = (ari_pred + nmi_pred + asw_pred) / 3

    return AvgBIO, AvgBIO_pred if return_pred else AvgBIO


def pearson_correlation(x, y, abs = False):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    covariance = torch.mean((x - x_mean) * (y - y_mean))
    x_std = torch.std(x)
    y_std = torch.std(y)
    correlation = covariance / (x_std * y_std)

    # make sure the value is between 0 and 1, also preserve the absolute value
    if abs:
        correlation = torch.abs(correlation)

    return correlation


def load_g2v(file_path):
    g2v = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            g2v[line[0]] = np.array([float(i) for i in line[1:]])
    
    return g2v

def kl_divergence(P, Q):
    return np.sum(P * np.log(P / Q))

def jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)


def permute_genes_by_expression(adata, dataset_name, tokenizer, symbol2id):
    """
    Permute gene expression values for each cell in the given AnnData object.

    Parameters:
        adata (AnnData): An AnnData object containing gene expression data.
        dataset_name (str): The name of the dataset being processed.
        tokenizer (Tokenizer): A tokenizer object used to convert gene names to token IDs.
        symbol2id (dict): A dictionary mapping gene symbols to IDs.

    Returns:
        np.ndarray: A 2D numpy array where each row contains permuted gene IDs based on expression values for each cell.
    """
    # create gene list for specific dataset
    mapped_gene_ids = []
    for i in range(adata.X.shape[1]):
        gene_name = adata.var.index[i]
        if dataset_name == "pbmc12k" or dataset_name == "perirhinal_cortex" or dataset_name == "pbmc12k_rand":
            mapped_gene_ids.append(tokenizer.convert_tokens_to_ids(gene_name))
        elif dataset_name != "pbmc12k" and gene_name in symbol2id:
            mapped_gene_ids.append(tokenizer.convert_tokens_to_ids(symbol2id[gene_name]))
        else:
            mapped_gene_ids.append(tokenizer.convert_tokens_to_ids("<unk>"))
    gene_ids = np.array(mapped_gene_ids)
    permuted_gene_ids = []

    # permute gene expression values for each cell
    for i in range(adata.X.shape[0]):
        if isinstance(adata.X, csr_matrix):
            expression_values = adata.X[i].toarray().flatten()
        else:
            expression_values = adata.X[i].flatten()
        # expression_values = adata.X[i].toarray().flatten() 
        sorted_indices = np.argsort(-expression_values) 
        permuted_gene_ids.append(gene_ids[sorted_indices])

    return np.array(permuted_gene_ids)

