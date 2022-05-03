import json 
import sys 
from tqdm import tqdm

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader, gptneo_tokens_to_programs
from etk.execution import semisafe_evaluate

filename = sys.argv[1]

inference_batch_size = 20
max_new_tokens = 150
num_samples = 20
temp = 0.4
prompt_length = 500


filename = "gptneo_gsm8k_full_pass20"

prompt = open("gsm8k_prompt.txt", "r").read()

train_data = read_gsm8k("../data/gsm8k/gsm8k_train.jsonl")

dataloader = batch_loader(train_data, inference_batch_size)

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

for batch in tqdm(dataloader(train_data)): 
    labels = [instance.answer for instance in batch]
    prompts = [prompt + instance.text for instance in batch]
    texts = [instance.text for instance in batch]
    task_ids = [instance.task_id for instance in batch]

    batch_length = len(prompts)

    encoded_texts = tokenizer(texts, 
                          return_tensors="pt", 
                          max_length = prompt_length, 
                          truncation='left', 
                          padding='max_length', 
                          truncation=True, 
                          )

    outputs = model.generate(**encoded_texts, 
                             do_sample=True, 
                             temperature=temp, 
                             max_new_tokens=max_new_tokens
                             num_return_sequences=num_samples
                             )

    outputs = torch.reshape(outputs, (batch_length, num_samples, -1))

    for text, idx, label, outs in zip(texts, task_ids, labels, outputs): 
        bodies = gptneo_tokens_to_programs(outs, prompt_length, tokenizer)
        answers = [semisafe_evaluate(program, 'answer', 1) for program in bodies]
        passed_lst = [(abs(answer-label.item())/max(label.item(), 1e-5))<0.01
                if isinstance(answer, float) else False
                for answer in answers]

        if True in passed_lst: 
            gold_code = bodies[passed_lst.index(True)]
            passed = 1 
        else: 
            gold_code=False
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


with open("results/{filename}.jsonl") as f:
    log = [json.loads(line) for line in f]


num_examples = len(log)

num_passed = sum([x["passk"] for x in log])
pass_k = num_passed/num_examples

pass_1 = sum([x["pass1"] for x in log])/num_examples


to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open("results/{filename}.json", "w") as fle: 
    json.dump(to_dump, fle)
