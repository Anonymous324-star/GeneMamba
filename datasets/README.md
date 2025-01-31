## Downstream tasks

Under the example folder, there are scripts to run the downstream tasks.

First download the data from the link https://drive.google.com/drive/folders/1R_L3-ivnrsupHeDSkFugjCr1AwLEzdyL?usp=sharing, and put all the datasets under the datasets/downstream folder.

Then, for each task, change the path arguments to your local path, and run the run.sh script, this will output the results all in the results folder under each task directory.



### Multi-batch Integration:

BMMC: This dataset includes scRNA-seq data from bone marrow mononuclear cells, encompassing various hematopoietic cell types. It is utilized to study hematopoiesis and bone marrow microenvironments.


Immune: This dataset contains scRNA-seq data from human immune cells across different tissues or conditions. It aids in understanding the diversity and function of the human immune system.


PBMC12k: The PBMC12k dataset consists of scRNA-seq data from 12,000 peripheral blood mononuclear cells obtained from a healthy donor. It serves as a reference for immune cell types and their gene expression profiles in the bloodstream.

Perirhinal Cortex: This dataset includes scRNA-seq data from cells of the perirhinal cortex, a region of the brain involved in memory and recognition. It is used to study neuronal and glial cell types and their gene expression profiles in this specific
brain area.

### Cell type annotation:

hPancrea: This dataset comprises single-cell RNA sequencing (scRNA-seq) data from human pancreatic cells, including various cell types such as alpha, beta, delta, and acinar cells. It is utilized to study cellular heterogeneity and gene expression
profiles within the pancreas.

MS: The multiple sclerosis (MS) dataset includes scRNA-seq data from peripheral blood mononuclear cells (PBMCs) of individuals with MS. It aids in understanding immune cell composition and gene expression changes associated with MS

Myeloid: This dataset contains scRNA-seq data focusing on myeloid cells, such as monocytes and macrophages, from various tissues or conditions. It is essential for studying the role of myeloid cells in immune responses and diseases.


We standardize the downstream datasets by converting the batch key to "batch" and the label key to "celltype." We then use a manual stratified split to divide the dataset into train/test with a 0.9/0.1 ratio to avoid missing rare classes in the test set. For example, in the COVID dataset, using `train_test_split()` directly would cause the 22nd class to be missing, as it contains only 3 samples, and a 0.3 test sample rate would result in zero samples, causing errors during AUC-ROC calculation. The split column is labeled "partition," where "train" is for training and "test" is for testing. 

