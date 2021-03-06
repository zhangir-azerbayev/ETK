import json 
import sys 
from tqdm import tqdm
import random 

import torch

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import transformers

from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader, tokens_to_gsm8k_log_entry#gptneo_tokens_to_log_entry
from etk.execution import semisafe_evaluate

random.seed(20)
torch.manual_seed(20)
transformers.set_seed(20)

device = "cuda:5"

# I'm not sure that inference_batch_size > 1 works
inference_batch_size = 1
max_new_tokens = 150
num_samples = 10
temp = 0.2
prompt_length = 756


filename = sys.argv[1]


prompt = open("gsm8k_prompt.txt", "r").read()

dataset = sys.argv[2]
train_data = read_gsm8k(f"../data/gsm8k/gsm8k_{dataset}.jsonl")

dataloader = batch_loader(train_data, max(inference_batch_size, 1))

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side='left'
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

for batch in tqdm(dataloader): 
    labels = [instance.answer for instance in batch]
    prompts = [prompt + instance.text for instance in batch]
    texts = [instance.text for instance in batch]
    task_ids = [instance.task_id for instance in batch]


    batch_length = len(prompts)

    encoded_texts = tokenizer(prompts, 
                          return_tensors="pt", 
                          max_length = prompt_length, 
                          truncation=True,
                        #   padding='max_length', 
                          ).to(device)

    prompt_lens = [torch.sum(x) for x in encoded_texts["attention_mask"]]

    num_forward_passes = max(int(1/inference_batch_size), 1)
    
    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens,
                             num_return_sequences=num_samples//num_forward_passes,
                             pad_token_id=tokenizer.eos_token_id
                             )

    if inference_batch_size < 1: 

        for _ in range(1, num_forward_passes): 
            to_concat = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens,
                             num_return_sequences=num_samples//num_forward_passes,
                             pad_token_id=tokenizer.eos_token_id,
                             )

            outputs = torch.cat((outputs, to_concat), dim=0)


    outputs = torch.reshape(outputs, (batch_length, num_samples, -1))

    for text, task_id, label, outs, prompt_len in zip(texts, 
            task_ids, labels, outputs, prompt_lens): 


        log_entry, _ = tokens_to_gsm8k_log_entry(outs, 
                                                  label,
                                                  task_id, 
                                                  text,
                                                  prompt_len, 
                                                  tokenizer,
                                                  "gptneo",
                                                  )

        with open(f"results/{filename}.jsonl", "a+") as f: 
            record = json.dumps(log_entry)
            f.write(record+'\n')


with open(f"results/{filename}.jsonl") as f:
    log = [json.loads(line) for line in f]


num_examples = len(log)

num_passed = sum([x["passk"] for x in log])
print(f"{num_passed}/{num_examples} passed")
pass_k = num_passed/num_examples

pass_1 = sum([x["pass1"] for x in log])/num_examples


to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open(f"results/{filename}.json", "w") as fle: 
    json.dump(to_dump, fle)
