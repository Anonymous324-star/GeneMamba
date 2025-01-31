import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotmap import DotMap

from transformers import AutoTokenizer, TrainingArguments, MambaForCausalLM

from genemamba.models import Classifier, GeneMamba, GeneMamba2
from genemamba.utils import build_dataset, MambaTrainer, get_last_checkpoint


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--bulk_id", type=int)
parser.add_argument("--seq_len", type=int)
parser.add_argument("--d_model", type=int)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--mamba_layer", type=int)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--mode", type=str)
parser.add_argument("--coexp", type=str)

args2 = parser.parse_args()

# for distrubted training
from torch import distributed as dist

from transformers import TrainerCallback

def setup_distributed_training(backend='nccl', init_method='env://', world_size=1, rank=0):
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

def cleanup_distributed_training():
    dist.destroy_process_group()

def setup_distributed_training(backend='nccl'):
    dist.init_process_group(backend=backend, init_method='env://')

setup_distributed_training()


args = DotMap(
    {
        "learning_rate": 5e-5,
        "batch_size": args2.batch_size,
        "gradient_accumulation_steps": 1,
        "optim": "adamw_torch",
        "num_epochs": 1,
        "num_samples": 1000000,
        "bulk_id": args2.bulk_id, # The bulk id of the dataset
        "seq_len": args2.seq_len,
        "output_dir": args2.output_dir,
        "mode": args2.mode,
        "coexpression_file": args2.coexp,
    }
)


last_checkpoint = get_last_checkpoint(f"{args.output_dir}/{args.num_epochs}/{args.bulk_id}m")
last_checkpoint

from transformers import PretrainedConfig

config = PretrainedConfig.from_dict({
    "d_model": args2.d_model,
    "mamba_layer": args2.mamba_layer
})

agent = GeneMamba2(config, model_path = last_checkpoint, tokenizer_path = args2.tokenizer_path, args = args)

print(agent.model)
print(f"Number of parameters: {agent.model.num_parameters() / 1e6:.2f}M")


train_dataset = build_dataset(args2.dataset_path, agent.tokenizer, args)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=agent.tokenizer)

trainer = MambaTrainer(
    model = agent.model,
    train_dataset = train_dataset,
    tokenizer = agent.tokenizer,
    args=TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs = 1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        output_dir=f"{args.output_dir}/{args.num_epochs}/{args.num_samples // 1000000 + args.bulk_id}m",
        logging_dir=f"{args.output_dir}/{args.num_epochs}/{args.num_samples // 1000000 + args.bulk_id}m_logging",
        logging_steps=100,
        save_steps=100,
        ddp_find_unused_parameters=True,
    ),
    coexpression_file=args.coexpression_file,
    # data_collator=data_collator,
)


# if last_checkpoint:
#     trainer.train(resume_from_checkpoint=last_checkpoint)
# else:
#     trainer.train()

trainer.train()
