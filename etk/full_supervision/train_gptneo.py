import sys 
import os
import yaml 
import math

from data.mathqa_dataset.py import MathQATrainSet

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names

def load_trainset_from_log(path, tokenizer, max_length): 
    with open(path) as f: 
        result = json.load(path)

    log = result["log"]

    filtered_log = [x if x["passed"]==1 for x in log]

    dataset = MathQATrainSet(filtered_log, tokenizer, max_length)

    return dataset

config_path = sys.argv[1]
with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["devices"]
experiment_name = cfg["experiment_name"]
teacher_data_path = cfg["log_path"]
lr = cfg["lr"]
epochs = cfg["epochs"]
batch_size = cfg["batch_size"]
weight_decay = cfg["weight_decay"]
param_count = cfg["param_count"]
max_length = cfg["max_length"]

results_dir = f"results/{experiment_name}"
os.mkdir(results_dir)

# Configures tokenizer and data
tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_trainset_from_log(teacher_data_path, tokenizer, max_length)

#Loads model
model = GPTNeoForCausalLM("EleutherAI/gpt-neo-{param_count}")

# Optimizer 
decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() 
            if n in decay_parameters],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() 
            if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda = lambda x: x)


# Configure training
steps_per_epoch = math.ceil(len(dataset)/batch_size)
training_args = TrainingArguments(output_dir=results_dir,
                                  num_train_epochs=epochs,
                                  per_device_train_batch_size=batch_size,
                                  logging_steps=steps_per_epoch*5,
                                  save_steps=steps_per_epoch*5,
                                  warmup_steps = 100,
                                  )


# Runs training 
Trainer(model=model, args=training_args, train_dataset=dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler)).train()
