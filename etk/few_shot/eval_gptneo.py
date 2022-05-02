import json 
import sys 
from tqdm import tqdm

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader, gptneo_tokens_to_programs
from etk.execution import semisafe_evaluate

inference_batch_size = 20
num_samples = 20
temp = 0.4


filename = "gptneo_gsm8k_full_pass20"

prompt = open("gsm8k_prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")

dataloader = batch_loader(train_data, inference_batch_size)

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

for batch in tqdm(dataloader(train_data)): 
    labels = [instance.answer for instance in batch]
    prompts = [prompt + instance.text for instance in batch]
    texts = [instance.text for instance in batch]
    task_ids = [instance.task_id for instance in batch]

    batch_length = len(prompts)

    tokenizer = tokenizer(texts, 
                          return_tensors="pt", 





