# please replace the paths with your own paths
model_paths=(
    "/path/to/ckpts/GeneMamba_24l_512d/checkpoint"
    "/path/to/ckpts/GeneMamba_24l_768d/checkpoint"
    "/path/to/ckpts/GeneMamba_48l_512d/checkpoint"
    "/path/to/ckpts/GeneMamba_48l_768d/checkpoint"
)

datasets=(
    "pbmc12k"
    "immune"
    "pancreas"
    "zheng68k"
)

seeds=(0 1 2 3 4)


# loop through all model paths and seeds
for dataset in "${datasets[@]}"; do
    for model_path in "${model_paths[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running: python reconstruct.py --dataset_name $dataset --pretrained_model_path $model_path --seed $seed"
            
            # please replace the paths with your own paths
            python reconstruct.py --dataset_name "$dataset" --dataset_path /path/to/datasets/downstream --pretrained_model_path "$model_path" --seed "$seed" --tokenizer_path "/path/to/gene_tokenizer.json"
        done
    done
done