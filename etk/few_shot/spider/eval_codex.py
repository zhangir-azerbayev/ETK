from tqdm import tqdm
import re
import random
import json
import sys
import openai
from ratelimit import limits, sleep_and_retry
from itertools import zip_longest
import time
from etk.eval_utils import batch_loader

from test-suite-sql-eval.evaluation import evaluate

time.sleep(60)

@sleep_and_retry
@limits(calls=1, period=60)
def call_api(engine, prompt, max_tokens, n, temperature): 
    return openai.Completion.create(engine=engine, 
            prompt=prompt, max_tokens=max_tokens, n=n, 
            temperature=temperature, 
            )

random.seed(20)
num_prompts = 20
n = 10
temp = 0.2

filename = "codex_spider_test"

prompt = open("../../data/spider/few_shot_prompt.txt").read()

with open("../../data/spider/train_spider_with_prompts.json") as f: 
    train_data = json.load(f)

dataloader = batch_loader(train_data, num_prompts)

for batch in tqdm(dataloader): 
    prompts = [prompt + "\n\n" + x["prompt"] for x in batch]
    
    # for logging
    task_ids = [x["task_id"] for x in batch]
    texts = [x["prompt"] for x in batch]

    outputs = call_api(engine="code-davinci-002", 
                       prompt=prompts, 
                       max_tokens=100, 
                       n=n, 
                       temperature=temp, 
                       )

    outputs = [output["text"] for output in outputs["choices"]]

    for out, text, task_id in zip(batch_loader(outputs, n), texts, task_ids): 
        queries = ["SELECT" + x[:x.index(';')+1] for x in out]
        print(queries)
        sys.exit()

