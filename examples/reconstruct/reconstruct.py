import sys
import os

import scanpy as sc
import anndata as ad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm, trange
from dotmap import DotMap

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, TrainingArguments, MambaForCausalLM, BertForMaskedLM
from datasets import Dataset, load_from_disk


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genemamba.models import Classifier, GeneMamba, GeneMambaForCellAnnotation, GeneMambaForGeneClassification, GeneMamba2, GeneTransformer
from genemamba.utils import build_dataset, MambaTrainer, prepare_data, pearson_correlation, load_g2v, kl_divergence, jensen_shannon_divergence, levenshtein_distance, bleu_score


plt.rcParams['font.family'] = 'STIXGeneral'


current_directory = os.path.dirname(os.path.abspath(__file__))



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default = "pbmc12k")
parser.add_argument("--dataset_path", type=str, default = "datasets/downstream/pbmc12k")
parser.add_argument("--pretrained_model_path", type=str)
parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--seed", type=int)

args = parser.parse_args()


dataset_name = args.dataset_name
pretrained_model_path = args.pretrained_model_path
seed = args.seed

model_name = pretrained_model_path.split("/")[-2]
d_model = int(model_name.split("_")[2][:-1])
mamba_layer = int(model_name.split("_")[1][:-1])

setting = "_".join(pretrained_model_path.split("/")[-2:])

print("model name: ", model_name)
print(f"d_model: {d_model}, mamba_layer: {mamba_layer}, setting: {setting}")


if "GeneMamba" in model_name:
    from transformers import PretrainedConfig

    config = PretrainedConfig.from_dict({
        "d_model": d_model,
        "mamba_layer": mamba_layer,
    })

    model = GeneMamba2(config, model_path=pretrained_model_path, tokenizer_path=args.tokenizer_path, args=None)

elif "GeneTransformer" in model_name:

    model = GeneTransformer(config = None, model_path = pretrained_model_path, tokenizer_path=args.tokenizer_path, args=None)

else:
    model = GeneMamba(model_path=pretrained_model_path, tokenizer_path=args.tokenizer_path, args=None)

print(model)



tokenized_dataset = load_from_disk(os.path.join(args.dataset_path, "reconstruct", args.dataset_name, f"{args.dataset_name}.dataset"))
print(tokenized_dataset)


def get_output_rankings(model, batch_size = 4):
    rankings_list = []

    size = len(tokenized_dataset)

    def _extend_batch(batch_dataset: Dataset):
        
        max_size = max(batch_dataset['length'])
        
        batch_ = [pad_tensor(x, max_size, model.tokenizer.pad_token_id) 
                    for x in batch_dataset['input_ids']]
        
        batch_ = torch.stack(batch_).to(model.device)

        return batch_

    def pad_tensor(t: torch.Tensor,
                max_size: int,
                pad_token_id: int = 0) -> torch.Tensor:
        """
        Pad a tensor to a max size
        """
        
        return F.pad(t, pad = (0, max_size - t.numel()), 
                    mode = 'constant', value = pad_token_id)

    for i in trange(0, size, batch_size, 
                    desc = "GeneMamba (extracting embeddings)"):
        
        max_range = min(i+batch_size, size)
        batch_dataset = tokenized_dataset.select(list(range(i, max_range)))
        batch_dataset.set_format(type = 'torch')
        
        batch = _extend_batch(batch_dataset)
        
        batch = batch.cuda()
        model = model.cuda()

        model_output = model(batch)

        # now, get the ranking reconstruction
        out_rankings = (model_output.logits
                        .argmax(axis=-1)
                        .detach().cpu().numpy())
        
        # save the rankings with the original order
        rankings_list.extend(out_rankings)

        torch.cuda.empty_cache()
        del model_output
        del batch
    
    return rankings_list


import pickle

if not os.path.exists(os.path.join(current_directory, f"results/rankings/{model_name}/")):
    os.makedirs(os.path.join(current_directory, f"results/rankings/{model_name}/"))

if f"output_rankings_{dataset_name}_{setting}.pkl" in os.listdir(os.path.join(current_directory, f"results/rankings/{model_name}/")):
    print("Loading output rankings")
    
    with open(os.path.join(current_directory, f"results/rankings/{model_name}/output_rankings_{dataset_name}_{setting}.pkl"), "rb") as f:
        output_rankings = pickle.load(f)
else:
    output_rankings = get_output_rankings(model)
    print("Saving output rankings")
    
    with open(os.path.join(current_directory, f"results/rankings/{model_name}/output_rankings_{dataset_name}_{setting}.pkl"), "wb") as f:
        pickle.dump(output_rankings, f)


import numpy as np
input_rankings = [np.array(item) for item in tokenized_dataset['input_ids']]


def get_venn(input_rankings, output_rankings, n_cells = 10):

    n_all_cells = len(input_rankings)

    if n_cells < n_all_cells:
        np.random.seed(seed)
        rand_cells = np.random.choice(range(n_all_cells), 
                                        n_cells, replace = False)
        input_rankings = [input_rankings[i] 
                            for i in rand_cells]
        output_rankings = [output_rankings[i] 
                            for i in rand_cells]

        # only consider the first input_rankings[i] genes
        output_rankings = [output_rankings[i][:len(input_rankings[i])]
                            for i in range(len(input_rankings))]
    else:
        Raise("n_cells should be less than the total number of cells")
        # sample_ids = [[i] * input_rankings[i].shape[0]
        #                           for i in range(len(input_rankings))]

    sample_ids = [tokenized_dataset.select([i])['adata_order']
                for i in rand_cells]
    sample_ids = [sample_ids[i] * input_rankings[i].shape[0]
                    for i in range(len(input_rankings))]

    positions = [np.arange(input_rankings[i].shape[0])
                    for i in range(len(input_rankings))]
    positions = [np.max(positions[i]) - positions[i] 
                    for i in range(len(input_rankings))]
    positions = [positions[i] / np.max(positions[i]) 
                    for i in range(len(input_rankings))]

    input_rankings = np.concatenate(input_rankings, axis = 0)
    sample_ids = np.concatenate(sample_ids, axis = 0)
    output_rankings = np.concatenate(output_rankings, axis = 0)
    positions = np.concatenate(positions, axis = 0)

    import pandas as pd

    input_df = pd.DataFrame({"token": input_rankings,
                                "sample_id": sample_ids,
                                "input_rank": positions})

    output_df = pd.DataFrame({"token": output_rankings,
                                "sample_id": sample_ids,
                                "output_rank": positions})
    if not os.path.exists(os.path.join(current_directory, f"results/tables/{model_name}/")):
        os.makedirs(os.path.join(current_directory, f"results/tables/{model_name}/"))

    input_df.to_csv(os.path.join(current_directory, f"results/tables/{model_name}/input_rankings_{dataset_name}_{setting}_{seed}.csv"))
    output_df.to_csv(os.path.join(current_directory, f"results/tables/{model_name}/output_rankings_{dataset_name}_{setting}_{seed}.csv"))

    input_tokens = input_df["token"].values
    output_tokens = output_df["token"].values
    set_input_tokens = set(input_tokens)
    set_output_tokens = set(output_tokens)

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    # create Venn diagram
    plt.figure(figsize=(6,6), dpi = 300)
    venn = venn2([set_input_tokens, set_output_tokens], ('Input tokens', 'Output tokens'))

    # display the plot
    plt.title(f"Venn Diagram for input tokens and output tokens \n of dataset {dataset_name}")

    if not os.path.exists(os.path.join(current_directory, f"results/figures/{model_name}/")):
        os.makedirs(os.path.join(current_directory, f"results/figures/{model_name}/"))
    plt.savefig(os.path.join(current_directory, f"results/figures/{model_name}/venn_diagram_{dataset_name}_{setting}_{seed}.png"))

    return input_df, output_df


input_df, output_df = get_venn(input_rankings, output_rankings)

def visualize_kde(input_df, output_df, n_cells = 10):
    import scipy.stats as stats

    input_tokens = input_df["token"].values
    output_tokens = output_df["token"].values

    l_distance = levenshtein_distance(input_tokens, output_tokens)
    BLEU_score = bleu_score(input_tokens, output_tokens)

    output_df = output_df.groupby(["token", "sample_id"]).mean().reset_index()

    df = pd.merge(input_df, output_df, 
                    on = ["token", "sample_id"], 
                    how = "outer")

    df = df.fillna(0)
    
    mean_rank_df = df[df['input_rank'] > 0].groupby(["token"]).apply(lambda x: x.input_rank.mean()).reset_index()

    mean_rank_df = mean_rank_df.rename(columns = {0: "mean_rank"})


    df = pd.merge(df, mean_rank_df, on = ["token"], how = "left")


    df = df.fillna(0)

    # calculate correlation with stats.pearsonr for each sample between input and output rankings
    cors = df.groupby("sample_id").apply(lambda x: stats.pearsonr(x["input_rank"], x["output_rank"])[0])
    mean_cors = df.groupby("sample_id").apply(lambda x: stats.pearsonr(x["input_rank"], x["mean_rank"])[0])
    
    cors_df = pd.concat([cors, mean_cors], axis = 1)
    cors_df = cors_df.rename(columns = {0: "correlation", 
                                        1: "mean_correlation"})

    rankings_df = df

    import seaborn as sns
    import matplotlib.pyplot as plt


    def input_output_correlation(input_rank, output_rank):
        spearman_corr = stats.spearmanr(input_rank, output_rank)[0]
        pearson_corr = stats.pearsonr(input_rank, output_rank)[0]
        return spearman_corr, pearson_corr

    print("ranking_df: ", rankings_df.sample(10))
    spearman_corr, pearson_corr = input_output_correlation(rankings_df['input_rank'], rankings_df['output_rank'])
    mean_spearman_corr, mean_pearson_corr = input_output_correlation(rankings_df['input_rank'], rankings_df['mean_rank'])


    print(f"{dataset_name}, {setting} and {seed}")
    print(f"Spearman correlation: {spearman_corr}")
    print(f"Pearson correlation: {pearson_corr}")
    print(f"Mean Spearman correlation: {mean_spearman_corr}")
    print(f"Mean Pearson correlation: {mean_pearson_corr}")
    print(f"Mean Levenshtein distance: {l_distance}")
    print(f"Mean BLEU score: {BLEU_score}")


    if not os.path.exists(os.path.join(current_directory, f"results/correlations/{model_name}/")):
        os.makedirs(os.path.join(current_directory, f"results/correlations/{model_name}/"))

    with open(os.path.join(current_directory, f"results/correlations/{model_name}/correlations_{dataset_name}_{setting}_{seed}.txt"), "w") as f:
        content = f"{dataset_name}, {setting} and {seed} \n Spearman correlation: {spearman_corr} \n Pearson correlation: {pearson_corr} \n Mean Spearman correlation {mean_spearman_corr} \n Mean Pearson correlation {mean_pearson_corr} \n Mean Levenshtein distance: {l_distance} \n Mean BLEU score: {BLEU_score}"
        
        f.write(content)

    cmap = "mako_r"

    n_all_available_cells = rankings_df['sample_id'].nunique()
    n_all_available_cells

    if n_cells < n_all_available_cells:
        rand_cells = np.random.choice(rankings_df['sample_id'].unique(), 
                                        n_cells, replace = False)
        rankings_df = rankings_df[rankings_df['sample_id'].isin(rand_cells)]

    # set seaborn style
    sns.set_style("white")

    fig, ax1 = plt.subplots(ncols = 1, 
                                    sharey = True, 
                                    tight_layout = True, 
                                    figsize = (8, 6), dpi = 300)
    ax1.set_xlabel("Input ranks")
    ax1.set_ylabel("Reconstructed ranks")

    sns.kdeplot(x = "input_rank", 
                y = "mean_rank", 
                data = rankings_df,
                cmap = cmap,
                fill = True, 
                shade = True,
                thresh = 0, ax = ax1)

    plt.suptitle(f"Correlation between input and reconstructed rankings \n of dataset {dataset_name}")

    plt.savefig(os.path.join(current_directory, f"results/figures/{model_name}/kde_plot_{dataset_name}_{setting}_{seed}.png"))

visualize_kde(input_df, output_df)
