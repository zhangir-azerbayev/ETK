from cl.data.dataset import read_gsm8k
from cl.execution import semisafe_evaluate
from tqdm import tqdm
import re
import random
import json
import sys
import openai
from ratelimit import limits, sleep_and_retry
from itertools import zip_longest
import time

def batch_loader(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

@sleep_and_retry
@limits(calls=1, period=60)
def call_api(engine, prompt, max_tokens, n, temperature): 
    return openai.Completion.create(engine=engine, 
            prompt=prompt, max_tokens=max_tokens, n=n, 
            temperature=temperature, 
            )

random.seed(20)
num_prompts = 5
n = 20
temp = 0.4

filename = "codex_gsm8k_full_pass20"

prompt = open("prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")

dataloader = batch_loader(train_data, num_prompts)

for batch in tqdm(dataloader[47+350+131:]): 
    labels = [instance.answer for instance in batch]
    prompts = [prompt + instance.text for instance in batch]
    texts = [instance.text for instance in batch]
    task_ids = [instance.task_id for instance in batch]
    

    outputs = call_api(engine="code-davinci-002", 
                       prompt=prompts, 
                       max_tokens=150, 
                       n=n, 
                       temperature=temp, 
                       )

    outputs = [output["text"] for output in outputs["choices"]]

    for out, label, task_id, text in zip(batch_loader(outputs, n), 
            labels, task_ids, texts):
        re_key = '\nanswer.*?\n'

        bodies = [completion[:re.search(re_key, completion).span()[1]]
            if re.search(re_key, completion) else completion
            for completion in out]

        
        answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]

        passed_lst = [(abs((answer - label)/max(label, 1e-5)) < 0.01) 
                        if isinstance(answer, float) else False 
                        for answer in answers]
        
        if True in passed_lst: 
            gold_code = bodies[passed_lst.index(True)]
            passed = 1
        else: 
            gold_code = False 
            passed = 0 


        pass_1 = sum(passed_lst)/len(passed_lst)
        
        with open(filename+".jsonl", "a+") as f: 
            record = json.dumps({"task_id": task_id, 
                        "text": text, 
                        "answer": label, 
                        "gold_solution": gold_code,
                        "passk": passed, 
                        "pass1": pass_1, 
                        "passed_lst": passed_lst})
            f.write(record+'\n')

with open(filename+'.jsonl') as f:
    log = [json.loads(line) for line in f]


num_examples = len(log)

num_passed = sum([x["passk"] for x in log])
pass_k = num_passed/num_examples

pass_1 = sum([x["pass1"] for x in log])/num_examples


to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open(filename+".json", "w") as fle: 
    json.dump(to_dump, fle)
