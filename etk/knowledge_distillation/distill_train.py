import sys 
import os
import yaml 
import math

import torch 
import torch.nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter 

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import TrainingArguments, Trainer 
from transformers import AdamW
from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import get_parameter_names

from etk.data.mathqa_dataset import read_gsm8k_with_code, MathQAPython
from etk.eval_utils import batch_loader, gptneo_tokens_to_programs
from etk.execution import semisafe_evaluate

config_path = sys.argv[1]

with open(config_path, "r") as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg['experiment_name']
param_count = cfg['param_count']
teacher_param_count = cfg['teacher_param_count']
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['devices']
max_length = cfg['max_length']
epochs = cfg['epochs']
batch_size = cfg['batch_size']
optim = cfg['optimizer']
lr = optim['lr']
weight_decay = optim['weight_decay']
scheduler_type = optim['scheduler_type']
scheduler_kwargs = optim['scheduler_kwargs']

# os.mkdir(f"./results/{experiment_name}/")

print('loading data and configuring tokenizer')
data = read_gsm8k_with_code("../few_shot/results/gptneo_gsm8k_full_pass20.jsonl")
print(len(data))
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


training_args = TrainingArguments(output_dir=f"./results/{experiment_name}",
                                  num_train_epochs=epochs,
                                  per_device_train_batch_size=batch_size, 
                                  logging_steps=100,
                                  save_steps=steps_per_epoch,
                                  warmup_steps = 100, 
                                  )

def data_collator(data):
    return {'input_ids': torch.stack([f[0] for f in data]),
            'attention_mask': torch.stack([f[1] for f in data]), 
            'labels': torch.stack([f[0] for f in data])
           }

tb_writer = SummaryWriter(log_dir=f"./results_train/{experiment_name}/tb_log")
tb_callback = TensorBoardCallback(tb_writer)

with open(f"./results/{experiment_name}/config.yml", "w+") as f: 
    yaml.dump(cfg, f)

print("loading teacher model")
teacher = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/gpt-neo-{teacher_param_count}").eval()

class DistilTrainer(Trainer):
    """Subclass of Trainer that implements a custom knowledge distillation loss."""
    def __init__(self, temp=1.0, teacher=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # cross-entropy loss between student and teacher logits. see distilBERT paper/code for details
        self.ce_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.temperature = temp


        self.teacher = teacher
        self.teacher.to("cuda:0") # this is a workaround to make sure the teacher is on the right device--ideally wouldn't do it this way

    def compute_loss(self, model, inputs, return_outputs=False):
        """ Computes distillation loss for a batch of data. """
        student_logits = model(**inputs)
        loss_CLM = student_logits.loss
        s_logits = student_logits.logits

        with torch.no_grad():
            teacher_logits = self.teacher(**inputs) # note: variable "teacher" is defined above in body of script
            t_logits = teacher_logits.logits

        # from DistilBERT code. https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py
        # filter out only non-padding tokens to use for loss computation
        # print(inputs["attention_mask"])
        mask = (inputs["attention_mask"] > 0).unsqueeze(-1).expand_as(s_logits) 
        s_logits_slct = torch.masked_select(s_logits, mask)  
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
        t_logits_slct = torch.masked_select(t_logits, mask)
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))
        

        loss_CE = self.ce_loss(
            F.log_softmax(s_logits_slct / self.temperature, dim=-1),
            F.softmax(t_logits_slct / self.temperature, dim=-1)
        ) * self.temperature ** 2 # unsure why the T^2 is necessary--couldn't find it in the paper, but it's in the distilBERT code
        
        # potential TODO: add cosine embedding loss between student and teacher hidden states
        
        # print(loss_CE, loss_CLM)
        loss = loss_CE + loss_CLM
        # print(loss)
        return (loss, student_logits) if return_outputs else loss



print("starting training")
DistilTrainer(teacher=teacher, model=model, args=training_args, train_dataset=train_set, 
        data_collator=data_collator, callbacks=[tb_callback], 
        optimizers = (optimizer, scheduler)).train()

