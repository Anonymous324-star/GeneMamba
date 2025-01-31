
import torch
from transformers import Trainer
import os

import pyarrow as pa 
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments

import argparse

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments, MambaForCausalLM

from dotmap import DotMap

import sys
import os
import torch

# from trange import trange

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from genemamba.models import Classifier, GeneMamba, GeneMambaForCellAnnotation, GeneMambaForGeneClassification, GeneMamba2, GeneMamba2ForCellClassification
from genemamba.utils import permute_genes_by_expression, standardize_columns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--ckpt_path", type = str)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--num_epochs", type=int, default=5)

args = parser.parse_args()

#%%
current_path = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(current_path, "results")

CHECKPOINT_PATH = args.ckpt_path
DATA_PATH = args.data_path
TOKENIZER_PATH = args.tokenizer_path

model_name = CHECKPOINT_PATH.split("/")[-2]
mamba_layer = int(model_name.split("_")[1][:-1])
d_model = int(model_name.split("_")[2][:-1])


# make the sub directories to save the results
SAVE_PATH = os.path.join(SAVE_PATH, model_name)
sub_directories = ["predictions", "metrics", "figures", "repr"]
for sub_dir in sub_directories:
    os.makedirs(os.path.join(SAVE_PATH, sub_dir), exist_ok=True)



import scanpy as sc

dataset_name = args.dataset_name

adata = sc.read_h5ad(os.path.join(DATA_PATH ,f'processed/{dataset_name}_processed.h5ad'))
print(f"Read data from {dataset_name}_processed.h5ad")

print(adata)


# adata = standardize_columns(adata, dataset_name)
# assert "batch" in adata.obs.columns and "celltype" in adata.obs.columns


from sklearn.preprocessing import LabelEncoder

y_names = np.array(adata.obs['celltype'].values.tolist())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_names)

num_class = len(label_encoder.classes_)


from transformers import PretrainedConfig

config = PretrainedConfig.from_dict({
    "d_model": d_model,
    "mamba_layer": mamba_layer,
})



model_cell_cls = GeneMamba2ForCellClassification(config, model_path=CHECKPOINT_PATH, tokenizer_path = TOKENIZER_PATH, args=None, output_dim_cls = num_class, hidden_dim= 512, num_layers_cls = 4)


permuted_gene_ids = permute_genes_by_expression(adata, dataset_name, model_cell_cls.tokenizer, model_cell_cls.symbol2id)


seq_len = args.seq_len
input_data = permuted_gene_ids[:, :seq_len]



# add the cls token to the input data
input_data = np.hstack([np.array([model_cell_cls.tokenizer.cls_token_id for _ in range(input_data.shape[0])]).reshape(-1, 1), input_data])


#%%
from sklearn.model_selection import train_test_split
import numpy as np

def manual_stratified_split(X, y, test_size=0.1, random_state=None):
    # separate the samples for each class
    unique_classes = np.unique(y)
    X_train, X_test, y_train, y_test = [], [], [], []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        
        if len(cls_indices) > 1:

            cls_train, cls_test = train_test_split(cls_indices, test_size=test_size, random_state=random_state)
        else:
            # if a class has only one sample, put it in the training set
            cls_train, cls_test = cls_indices, []
        
        X_train.extend(X[cls_train])
        y_train.extend(y[cls_train])
        X_test.extend(X[cls_test])
        y_test.extend(y[cls_test])
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#%%
# train and test split is done and stored in the adata.obs["partition"] column, so we can extract the train and test data from there

X_train = input_data[adata.obs["partition"] == "train"]
X_test = input_data[adata.obs["partition"] == "test"]
y_train = y[adata.obs["partition"] == "train"]
y_test = y[adata.obs["partition"] == "test"]

print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}, Train label shape: {y_train.shape}, Test label shape: {y_test.shape}")


from torch.utils.data import DataLoader, Dataset

class GeneDataset(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = GeneDataset(X_train, y_train)
test_dataset = GeneDataset(X_test, y_test)
all_dataset = GeneDataset(input_data, y)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False)


from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_pred, y_prob, y_true):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "Micro-F1 score": f1_score(y_true, y_pred, average='micro'),
        "Macro-F1 score": f1_score(y_true, y_pred, average='macro'),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        # "auc_roc": roc_auc_score(y_true, y_prob, multi_class = 'ovr'),
    }
    return metrics


epochs = args.num_epochs
optimizer = torch.optim.Adam(model_cell_cls.parameters(), lr=1e-4)
loss = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model_cell_cls.train()
    for i, batch in enumerate(train_loader):
        data = batch[0]
        target = batch[1]
        data = data.to(model_cell_cls.device)
        target = target.to(model_cell_cls.device)
        model_cell_cls = model_cell_cls.to(model_cell_cls.device)
        
        optimizer.zero_grad()
        output = model_cell_cls(data, None)
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss_val}")

    model_cell_cls.eval()
    with torch.no_grad():
        # add code to compute the metrics
        pred_prob = []
        pred_label = []
        targets = []
        cell_repr = []

        for i, batch in enumerate(test_loader):
            data = batch[0]
            target = batch[1]
            data = data.to(model_cell_cls.device)
            target = target.to(model_cell_cls.device)
            model_cell_cls = model_cell_cls.to(model_cell_cls.device)

            output, output_test_repr = model_cell_cls(data, None, return_cls = True)
            cell_repr.append(output_test_repr.cpu().numpy())
            
            # calculate the probability from the output
            pred_prob.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())

            _, predicted = torch.max(output, 1)
            pred_label.append(predicted.cpu().numpy())
            targets.append(target.cpu().numpy())
        
        pred_prob = np.concatenate(pred_prob)
        pred_label = np.concatenate(pred_label)
        targets = np.concatenate(targets)
        cell_repr = np.concatenate(cell_repr)

        # break
        # save the predictions
        np.save(os.path.join(SAVE_PATH, f"predictions/pred_prob_{dataset_name}_{epoch}.npy"), pred_prob)
        np.save(os.path.join(SAVE_PATH, f"predictions/pred_label_{dataset_name}_{epoch}.npy"), pred_label)
        np.save(os.path.join(SAVE_PATH, f"predictions/targets_{dataset_name}_{epoch}.npy"), targets)
        
        metrics = compute_metrics(pred_label, pred_prob, targets)
        
        with open(os.path.join(SAVE_PATH, f"metrics/metrics_{dataset_name}_{epoch}.txt"), "w") as f:
            print(metrics, file=f)
            print(metrics)
        

        # draw scatter plot for the first two components
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cell_repr)

        plt.figure(figsize=(8, 8))

        plt.scatter(pca_result[:, 0], pca_result[:, 1], c = targets)
        plt.savefig(os.path.join(SAVE_PATH, f"figures/scatter_{dataset_name}_{epoch}.png"))
        # plt.show()



model_cell_cls.eval()

def cell_embeddings(data_loader, model_cell_cls):
    cell_repr = []

    for i, batch in enumerate(data_loader):
        data = batch[0]
        target = batch[1]
        data = data.to(model_cell_cls.device)
        target = target.to(model_cell_cls.device)
        model_cell_cls = model_cell_cls.to(model_cell_cls.device)

        output, output_test_repr = model_cell_cls(data, None, return_cls = True)
        cell_repr.append(output_test_repr.detach().cpu().numpy())
        if i % 10 == 0:
            print(f"Processed {i} batches")

    cell_repr = np.concatenate(cell_repr)
    return cell_repr


test_cell_repr = cell_embeddings(test_loader, model_cell_cls)
save_path_test = os.path.join(SAVE_PATH, f"repr/{dataset_name}_test_cell_repr.npy")
np.save(save_path_test, test_cell_repr)
del test_cell_repr


train_cell_repr = cell_embeddings(train_loader, model_cell_cls)
save_path_train = os.path.join(SAVE_PATH, f"repr/{dataset_name}_train_cell_repr.npy")
np.save(save_path_train, train_cell_repr)
del train_cell_repr

all_cell_repr = cell_embeddings(all_loader, model_cell_cls)
save_path_all = os.path.join(SAVE_PATH, f"repr/{dataset_name}_cell_repr.npy")
np.save(save_path_all, all_cell_repr)
del all_cell_repr



