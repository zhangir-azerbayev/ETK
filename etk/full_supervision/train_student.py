import sys 
import os
import yaml 
import math
import json
from tqdm import tqdm

from etk.data.mathqa_dataset import MathQATrainSet, read_gsm8k
from etk.eval_utils import tokens_to_gsm8k_log_entry, batch_loader

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer 
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names

def load_trainset_from_log(path, 
                           tokenizer, 
                           max_length, 
                           train_on_dev = False): 
    with open(path, "r") as f: 
        result = json.load(f)

    log = result["log"]

    if not train_on_dev: 
        # Filters dev set
        with open("../data/gsm8k/dev_idxs.json") as f: 
            dev_idxs = json.load(f)

        log = [x for x in log if x["task_id"] not in dev_idxs]
    
    # Filters out examples with no solutions
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["devices"]
experiment_name = cfg["experiment_name"]
teacher_data_path = cfg["teacher_data_path"]
lr = cfg["lr"]
epochs = cfg["epochs"]
batch_size = cfg["batch_size"]
grad_accum=cfg["gradient_accumulation_steps"]
eval_batch_size = cfg["eval_batch_size"]
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
# config values to be used for validation
max_generation_length = cfg["max_gen_tokens"] + cfg["max_length"]
num_samples = cfg["num_samples"]
temp = cfg["temp"]
train_on_dev = cfg["train_on_dev"]

results_dir = f"train_results/{experiment_name}"
os.mkdir(results_dir)

# Configures tokenizer and data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_trainset_from_log(teacher_data_path, tokenizer, max_length, 
        train_on_dev)
eval_dataset = read_gsm8k("../data/gsm8k/gsm8k_dev.jsonl")

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

num_gpus = torch.cuda.device_count()
steps_per_epoch = math.ceil(len(dataset)/(batch_size*num_gpus*grad_accum))
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                         100, 
                                                         epochs*steps_per_epoch, 
                                                         )

                                                        


# Configure training
with open(os.path.join(results_dir, "config.yml"), "w") as f: 
    yaml.dump(cfg, f)

training_args = Seq2SeqTrainingArguments(output_dir=results_dir,
                                  num_train_epochs=epochs,
                                  evaluation_strategy="steps",
                                  eval_steps=steps_per_epoch*10,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=eval_batch_size,
                                  logging_steps=steps_per_epoch*10,
                                  save_steps=steps_per_epoch*10,
                                  remove_unused_columns=False,
                                  max_grad_norm=1.0,
                                  gradient_accumulation_steps=grad_accum,
                                  )

class TrainerWithEval(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, **kwargs):
        "Overrides the regular Seq2SeqTrainer.evaluate() since do_sample cannot be set via a training argument."
        print("Validating...")
        log = []
        eval_dataset = self.eval_dataset
        dataloader = batch_loader(eval_dataset, self.args.per_device_eval_batch_size) 
        for batch in tqdm(dataloader): 
            labels = [instance.answer for instance in batch]
            texts = [instance.text for instance in batch]
            task_ids = [instance.task_id for instance in batch]

            batch_length = len(texts)
            tokenizer.padding_side = "left"
            encoded_texts = tokenizer(texts, 
                                return_tensors="pt", 
                                max_length=max_length, 
                                truncation=True,
                                # padding='max_length', 
                                ).to("cuda")
            tokenizer.padding_side = "right"
            prompt_lens = [torch.sum(x) for x in encoded_texts["attention_mask"]]

            outputs = model.generate(input_ids=encoded_texts["input_ids"], 
                                    attention_mask=encoded_texts["attention_mask"], 
                                    do_sample=True, 
                                    temperature=temp, 
                                    max_new_tokens=cfg["max_gen_tokens"],
                                    num_return_sequences=num_samples,
                                    pad_token_id=tokenizer.eos_token_id
                                    )
            

            outputs = torch.reshape(outputs, (batch_length, num_samples, -1))
            

            for text, task_id, label, outs, prompt_len in zip(texts, 
                    task_ids, labels, outputs, prompt_lens): 

                log_entry, _ = tokens_to_gsm8k_log_entry(outs, 
                                                        label,
                                                        task_id,
                                                        text,
                                                        prompt_len, 
                                                        tokenizer, 
                                                        model_type,
                                                        verbose=False)

                log.append(log_entry)
                
        num_examples = len(log)
        num_passed = sum([x["passk"] for x in log])
        pass_k = num_passed/num_examples
        pass_1 = sum([x["pass1"] for x in log])/num_examples
        print(f"pass@{num_samples}: {pass_k}\n")
        return {f"pass@{num_samples}": pass_k}
    

# Runs training 
TrainerWithEval(model=model, args=training_args, train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler)).train()
