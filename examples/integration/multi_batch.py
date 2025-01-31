# %%
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

import scib

import warnings

warnings.filterwarnings("ignore")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from genemamba.models import Classifier, GeneMamba, GeneMambaForCellAnnotation, GeneMambaForGeneClassification, GeneMamba2, GeneMamba2ForCellClassification
from genemamba.utils import permute_genes_by_expression, eval_scib_metrics, standardize_columns


# %%
import scib
import scanpy as sc

def eval_scib_metrics(
    adata: sc.AnnData,
    batch_key: str = "batch",
    label_key: str = "celltype"
):
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_genemamba",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    print(f"{results}")

    result_dict = results[0].to_dict()
    print(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )
    
    result_dict["avg_batch"] = np.mean(
        [
            result_dict["ASW_label/batch"],
            result_dict["graph_conn"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--architecture", type=str)


args2 = parser.parse_args()

current_path = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(current_path, "results")

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


args = DotMap({
    "dataset_name": args2.dataset_name,
    "seq_len": 2048,
    "batch_size": 24,
    "num_epochs": 5,
    "test_size": 0.1
})

# %%
import scanpy as sc

# Load the .h5ad file
dataset_name = args.dataset_name
adata = sc.read_h5ad(os.path.join(args2.data_path, "origin", f'{dataset_name}.h5ad'))

print(adata)


adata = standardize_columns(adata, dataset_name)
assert "batch" in adata.obs.columns and "celltype" in adata.obs.columns


embedding_path = os.path.join(current_path, "..", "annotation", "results", args2.architecture, "repr", f"{dataset_name}_cell_repr.npy")

if not os.path.exists(embedding_path):
    print(f"Embedding path {embedding_path} does not exist, please run the annotation pipeline first.")
else:
    cell_repr = np.load(embedding_path)
    adata.obsm["X_genemamba"] = cell_repr
    print(f"Loaded embedding from {embedding_path}")


results = {}

print("calculating scib metrics")
results = eval_scib_metrics(adata, batch_key="batch", label_key="celltype")


print("plotting umap of batch")
adata.obs['batch'] = pd.Categorical(adata.obs['batch'])
sc.pp.neighbors(adata, use_rep="X_genemamba")
sc.tl.umap(adata, min_dist=0.3)
fig = sc.pl.umap(
    adata,
    color=["batch"],
    title=[f"{dataset_name}, batch, avg_batch = {results.get('avg_batch', 0.0):.4f}"],
    frameon=False,
    return_fig=True,
    show=False,
    legend_loc="right margin",
    # save=os.path.join(SAVE_PATH, f"{dataset_name}_batch_umap.png")
)

# fig.set_size_inches(10, 5)
fig.savefig(os.path.join(SAVE_PATH, f"{dataset_name}_batch_umap.png"), bbox_inches="tight", dpi=300)
results["batch_umap"] = fig


print("plotting umap of celltype")
sc.pp.neighbors(adata, use_rep="X_genemamba")
sc.tl.umap(adata, min_dist=0.3)
fig = sc.pl.umap(
    adata,
    color=["celltype"],
    title=[
        f"{dataset_name}, celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
    ],
    frameon=False,
    return_fig=True,
    show=False,
    legend_loc="right margin",
    # save=os.path.join(SAVE_PATH, f"{dataset_name}_celltype_umap.png")
)

fig.savefig(os.path.join(SAVE_PATH, f"{dataset_name}_celltype_umap.png"), bbox_inches="tight", dpi=300)
results["celltype_umap"] = fig


print(results)

