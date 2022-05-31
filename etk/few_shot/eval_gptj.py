import json 
import sys 
from tqdm import tqdm

import torch

from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM

from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader, gptneo_tokens_to_programs
from etk.execution import semisafe_evaluate

device = f"cuda:0"

inference_batch_size = 1
max_new_tokens = 150
num_samples = 10
temp = float(sys.argv[1])
prompt_length = 756

filename = sys.argv[2]

prompt = open("gsm8k_prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")

dataloader = batch_loader(train_data, inference_batch_size)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.truncation_side='left'

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0)

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="main",
)

model.parallelize()

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
                          ).to(device)

    prompt_lens = [torch.sum(x) for x in encoded_texts["attention_mask"]]


    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens,
                             num_return_sequences=num_samples,
                             pad_token_id=50270,
                             ).cpu()
    outputs2 = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens,
                             num_return_sequences=num_samples,
                             pad_token_id=tokenizer.eos_token_id,
                             ).cpu()
    outputs = torch.cat((outputs, outputs2), dim=0)
    
    # print(outputs.shape)
    outputs = torch.reshape(outputs, (batch_length, num_samples * 2, -1))
    # print(outputs.shape)
    for text, task_id, label, outs, prompt_len in zip(texts, task_ids, labels, outputs, prompt_lens): 

        bodies = gptneo_tokens_to_programs(outs, prompt_len, tokenizer)
        [print("OUTPUT:\n"+body) for body in bodies]
        answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]
        passed_lst = [(abs(answer-label)/max(label, 1e-5))<0.01
                if isinstance(answer, float) else False
                for answer in answers]

        if True in passed_lst: 
            gold_code = bodies[passed_lst.index(True)]
            passed = 1 
        else: 
            gold_code=False
            passed = 0 

        pass_1 = sum(passed_lst)/len(passed_lst)
        with open(f"results/{filename}.jsonl", "a+") as f: 
            record = json.dumps({"task_id": task_id, 
                                 "text": text, 
                                 "answer": label, 
                                 "gold_solution": gold_code,
                                 "passk": int(passed), 
                                 "pass1": pass_1, 
                                 "passed_lst": passed_lst})
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
