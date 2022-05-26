import sys 
import os
import yaml 
import math
import json

from etk.data.mathqa_dataset import MathQATrainSet

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names

def load_trainset_from_log(path, tokenizer, max_length): 
    with open(path, "r") as f: 
        result = json.load(f)

    log = result["log"]

    filtered_log = [x for x in log if True in x["passed_lst"]]

    dataset = MathQATrainSet(filtered_log, tokenizer, max_length)

    return dataset

def gptneo_data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]),
            'labels': torch.stack([f[0] for f in data])
           }

def incoder_data_collator(data):
    """
    incoder tokenizer puts an eos token at the beginning of 
    every sequence. 
    """
    return {'input_ids': torch.stack([f[0][1:] for f in data]),
            'attention_mask': torch.stack([f[1][1:] for f in data]),
            'labels': torch.stack([f[0][1:] for f in data])
           }


config_path = sys.argv[1]
with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["devices"]
experiment_name = cfg["experiment_name"]
teacher_data_path = cfg["teacher_data_path"]
lr = cfg["lr"]
epochs = cfg["epochs"]
batch_size = cfg["batch_size"]
weight_decay = cfg["weight_decay"]
model_name = cfg["model_name"]
model_type = cfg["model_type"] 
if model_type=="gptneo": 
    data_collator = gptneo_data_collator
elif model_type=="incoder": 
    data_collator = incoder_data_collator
else: 
    raise ValueError("invalid `model_type`")
max_length = cfg["max_length"]

results_dir = f"train_results/{experiment_name}"
os.mkdir(results_dir)

# Configures tokenizer and data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = '<|endoftext|>'

dataset = load_trainset_from_log(teacher_data_path, tokenizer, max_length)

#Loads model
model = AutoModelForCausalLM.from_pretrained(model_name)

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
scheduler = LambdaLR(optimizer, lr_lambda = lambda x: 1)


# Configure training
with open(os.path.join(results_dir, "config.yml"), "w") as f: 
    yaml.dump(cfg, f)

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
