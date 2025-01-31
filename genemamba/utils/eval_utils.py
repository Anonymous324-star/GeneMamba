import numpy as np


def standardize_columns(adata, dataset_name):
    
    column_mappings = {
        "pbmc10k": {
            "celltype": "cell_type",
            "batch": None
        },
        "pancreas": {
            "celltype": "celltype",
            "batch": None
        },
        "bmmc": {
            "celltype": "cell_type",
            "batch": "batch"
        },
        "immune": {
            "celltype": "final_annotation",
            "batch": "batch"
        },
        "pbmc12k": {
            "celltype": "str_labels",
            "batch": "batch"
        },
        "pbmc12k_rand": {
            "celltype": "str_labels",
            "batch": "batch"
        },
        "pbmc68k": {
            "celltype": "celltype",
            "batch": None
        },
        "covid19": {
            "celltype": "celltype",
            "batch": "batch"
        },
        "perirhinal_cortex": {
            "celltype": "cell_type",
            "batch": "sample_id"
        },
        "myeloid": {
            "celltype": "cell_type",
            "batch": "batch"
        },
        "zheng68k": {
            "celltype": "celltype",
            "batch": None
        },
        "hpancreas": {
            "celltype": "Celltype",
            "batch": None
        },
        "ms": {
            "celltype": "Factor Value[inferred cell type - authors labels]",
            "batch": "str_batch"
        }
    }
    
    if dataset_name not in column_mappings:
        raise ValueError(f"Dataset name '{dataset_name}' not recognized.")
    
    
    adata.obs["celltype"] = adata.obs[column_mappings[dataset_name]["celltype"]]
    adata.obs["batch"] = adata.obs.get(column_mappings[dataset_name]["batch"], None)
    
    
    if "celltype" not in adata.obs.columns or "batch" not in adata.obs.columns:
        raise ValueError("celltype or batch column not found in adata.obs")
    
    return adata


from sklearn.model_selection import train_test_split
import numpy as np

def manual_stratified_split(adata, test_size=0.1, random_state=None):
    # get cell types as y
    y = adata.obs['celltype'].values
    unique_classes = np.unique(y)
    
    partition = np.array(['train'] * adata.n_obs)
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        
        if len(cls_indices) > 1:
            cls_train, cls_test = train_test_split(cls_indices, test_size=test_size, random_state=random_state)
            partition[cls_test] = 'test'
        else:
            # If a class has only one sample, keep it in the training set
            cls_train, cls_test = cls_indices, []

    adata.obs['partition'] = partition
    return adata

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



def levenshtein_distance(seq1, seq2):
    # Create a distance matrix
    len_seq1 = len(seq1) + 1
    len_seq2 = len(seq2) + 1

    # Initialize matrix of zeros
    distance_matrix = [[0 for x in range(len_seq2)] for x in range(len_seq1)]

    # Populate matrix with base case values
    for i in range(len_seq1):
        distance_matrix[i][0] = i
    for j in range(len_seq2):
        distance_matrix[0][j] = j

    # Fill in the matrix with Levenshtein algorithm
    for i in range(1, len_seq1):
        for j in range(1, len_seq2):
            if seq1[i-1] == seq2[j-1]:
                cost = 0  # No cost if characters are the same
            else:
                cost = 1  # Cost of 1 for a substitution

            distance_matrix[i][j] = min(
                distance_matrix[i-1][j] + 1,      # Deletion
                distance_matrix[i][j-1] + 1,      # Insertion
                distance_matrix[i-1][j-1] + cost  # Substitution
            )

    # The last cell contains the Levenshtein distance
    return distance_matrix[-1][-1]


def bleu_score(candidate, reference, max_n=4):
    
    from collections import Counter
    import math

    def n_gram_counts(sequence, n):
        """Helper function to count n-grams in a sequence."""
        return Counter([tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)])

    def modified_precision(candidate, reference, n):
        """Calculate modified precision for n-grams."""
        candidate_counts = n_gram_counts(candidate, n)
        reference_counts = n_gram_counts(reference, n)

        overlap = {
            ngram: min(count, reference_counts.get(ngram, 0))
            for ngram, count in candidate_counts.items()
        }

        overlap_count = sum(overlap.values())
        candidate_count = sum(candidate_counts.values())

        # avoid division by zero
        return overlap_count / candidate_count if candidate_count > 0 else 0


    def brevity_penalty(candidate, reference):
        """Calculate brevity penalty to penalize short candidate sequences."""
        c = len(candidate)
        r = len(reference)

        if c > r:
            return 1
        elif c == 0:  # avoid division by zero
            return 0
        else:
            return math.exp(1 - r / c)


    """Compute the BLEU score for a candidate sequence given a reference sequence."""
    precisions = [modified_precision(candidate, reference, n) for n in range(1, max_n + 1)]

    # logarithmic average of precisions (geometric mean)
    if min(precisions) > 0:
        precision_product = sum([math.log(p) for p in precisions]) / max_n
        geometric_mean = math.exp(precision_product)
    else:
        geometric_mean = 0

    bp = brevity_penalty(candidate, reference)

    return bp * geometric_mean

def exact_match(input_ids, output_ids):
    return np.mean(input_ids == output_ids)
