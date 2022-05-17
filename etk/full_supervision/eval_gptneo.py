import json 
import sys 
from tqdm import tqdm
import yaml 
import os

import torch

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader, gptneo_tokens_to_log_entry
from etk.execution import semisafe_evaluate

cfg_path = sys.argv[1]
with open(cfg_path) as f: 
    cfg = yaml.safe_load(f)

experiment_name = cfg["experiment_name"]
os.environ["CUDA_VISIBLE_DEVICES"]=cfg["device"]
inference_batch_size = cfg["inference_batch_size"]
max_new_tokens = cfg["max_new_tokens"]
num_samples = cfg["num_samples"]
temp = cfg["temp"]
prompt_length = cfg["prompt_length"]
model_path = cfg["model_path"]
param_count = cfg["param_count"]

results_dir = f"eval_results/{experiment_name}"
os.mkdir(results_dir)


eval_data = read_gsm8k("../data/gsm8k/gsm8k_test.jsonl")

dataloader = batch_loader(eval_data, inference_batch_size) 

tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/gpt-neo-{param_count}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side='left'
model = GPTNeoForCausalLM.from_pretrained(model_path).to("cuda")

log = []
for batch in tqdm(dataloader): 
    labels = [instance.answer for instance in batch]
    texts = [instance.text for instance in batch]
    task_ids = [instance.task_id for instance in batch]


    batch_length = len(texts)

    encoded_texts = tokenizer(texts, 
                          return_tensors="pt", 
                          max_length = prompt_length, 
                          truncation=True,
                          padding='max_length', 
                          ).to("cuda")

    prompt_lens = [torch.sum(x) for x in encoded_texts["attention_mask"]]


    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens,
                             num_return_sequences=num_samples,
                             pad_token_id=tokenizer.eos_token_id
                             )
    

    outputs = torch.reshape(outputs, (batch_length, num_samples, -1))

    for text, task_id, label, outs, prompt_len in zip(texts, 
            task_ids, labels, outputs, prompt_lens): 

        log_entry, _ = gptneo_tokens_to_log_entry(outs, 
                                                  label,
                                                  task_id,
                                                  text,
                                                  prompt_len, 
                                                  tokenizer, 
                                                  verbose=False)

        log.append(log_entry)



num_examples = len(log)
num_passed = sum([x["passk"] for x in log])
pass_k = num_passed/num_examples
pass_1 = sum([x["pass1"] for x in log])/num_examples

to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open(os.path.join(results_dir, "log.json"), "w") as fle: 
    json.dump(to_dump, fle)

with open(os.path.join(results_dir, "config.yml"), "w") as fle: 
    yaml.dump(cfg, fle)




