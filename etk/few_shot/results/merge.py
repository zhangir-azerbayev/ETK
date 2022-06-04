import json
from tqdm import tqdm
from etk.eval_utils import pass_k

with open(f"gptneo2B_gsm8k_full_pass100.jsonl") as f: 
    head = [json.loads(line) for line in f]

with open(f"gptneo2B_gsm8k_tail_pass50.jsonl") as f: 
    tail = [json.loads(line) for line in f]

with open(f"gptneo2B_gsm8k_tail_pass50.jsonl") as f:
    tail_1 = [json.loads(line) for line in f]


combined_tail = []

for x, y in tqdm(zip(tail, tail_1)): 
    if x["task_id"] != y["task_id"]: 
        print(x)
        print(y)
        raise AssertionError("not the same task!")

    log = x 

    if not log["gold_solution"]: 
        log["gold_solution"] = log["gold_solution"]

    log["passed_lst"] = log["passed_lst"] + log["passed_lst"]

    log["passk"] = pass_k(log["passed_lst"], 100)

    log["pass1"] = pass_k(log["passed_lst"], 1)

    combined_tail.append(log)

log = head + combined_tail

num_examples = len(log)

num_passed = sum([x["passk"] for x in log])
print(f"{num_passed}/{num_examples} passed")
pass_k = num_passed/num_examples

pass_1 = sum([x["pass1"] for x in log])/num_examples


to_dump = {"passk": pass_k, 
           "pass1": pass_1, 
           "log": log}
                
with open(f"gptneo2B_gsm8k_full_pass100.json", "w") as fle: 
    json.dump(to_dump, fle)
