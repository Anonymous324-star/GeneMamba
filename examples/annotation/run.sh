# Cell type annotation for the following datasets:
python cell_type_annotation.py --dataset_name hpancreas --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 30 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name ms --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 30 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name myeloid --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 30 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name myeloid_b --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 30 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint


# Multi_batch integration for the following datasets:

python cell_type_annotation.py --dataset_name pbmc12k --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 5 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name bmmc --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 16 --num_epochs 5 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name immune --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 5 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name covid19 --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 5 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint

python cell_type_annotation.py --dataset_name perirhinal_cortex --data_path path_to_datasets/downstream --tokenizer_path path_to_tokenizer_file --seq_len 2048 --batch_size 24 --num_epochs 5 --ckpt_path /path/to/ckpts/GeneMamba_24l_512d/checkpoint



