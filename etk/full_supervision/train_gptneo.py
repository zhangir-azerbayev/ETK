import sys 
import os
import yaml 
import math

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter 

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import get_parameter_names

from data.dataset import read_mathqapython, MathQAPython

config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
param_count = cfg['param_count']
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
max_length = cfg['max_length']
epochs = cfg['epochs']
batch_size = cfg['batch_size']
optim = cfg['optimizer']
lr = optim['lr']
weight_decay = optim['weight_decay']
scheduler_type = optim['scheduler_type']
scheduler_kwargs = optim['scheduler_kwargs']

os.mkdir(f"results_train/{experiment_name}/")

print('loading data and configuring tokenizer')
data = read_mathqapython('data/mathqapython_train.json')

tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")
tokenizer.pad_token = tokenizer.eos_token 

train_set = MathQAPython(data, tokenizer, max_length)

print('loading model')
model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")

print('initializing training')
# Setting up optimizer 
# Parameter stuff is copied from huggingface 
decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)


steps_per_epoch = math.ceil(len(train_set)/batch_size)
if scheduler_type=="exponential": 
    gamma = scheduler_kwargs["gamma"]
    lr_lambda = lambda step: gamma ** (step//steps_per_epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


training_args = TrainingArguments(output_dir=f"./results_train/{experiment_name}",
                                  num_train_epochs=epochs,
                                  per_device_train_batch_size=batch_size, 
                                  logging_steps=steps_per_epoch*5,
                                  save_steps=steps_per_epoch*5,
                                  warmup_steps = 100, 
                                  )

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]), 
            'labels': torch.stack([f[0] for f in data])
           }

tb_writer = SummaryWriter(log_dir=f"./results_train/{experiment_name}/tb_log")
tb_callback = TensorBoardCallback(tb_writer)

with open(f"./results_train/{experiment_name}/config.yml", "w") as f: 
    yaml.dump(cfg, f)

Trainer(model=model, args=training_args, train_dataset=train_set, 
        data_collator=data_collator, callbacks=[tb_callback], 
        optimizers = (optimizer, scheduler)).train()

