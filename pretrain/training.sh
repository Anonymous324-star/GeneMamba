for bulk_id in 0 1 2 3 4 5 6 7 8 9
do
    torchrun --nproc-per-node=4 train.py --dataset_path path/to/dataset_file/data.arrow --coexp path/to/coexpression_file --tokenizer_path path/to/tokenizer_file --output_dir path/to/output_dir --bulk_id $bulk_id --d_model 512 --mamba_layer 24 --mode gate --seq_len 2048 --batch_size 24
done

