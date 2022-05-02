import torch 
import json 
import jsonlines
from pathlib import Path 

class MathQAInstance(): 
    def __init__(self, 
                 text, 
                 code, 
                 dsl_code, 
                 reasoning, 
                 answer, 
                 task_id): 
        self.text = text
        self.code = code
        self.dsl_code = dsl_code
        self.reasoning = reasoning
        self.answer = answer
        self.task_id = task_id

    def set_code(self, code): 
        self.code = code 
    
    def __str__(self): 
        return f"{self.text}\n{self.code}\n{self.answer}\n{self.reasoning}"


def read_gsm8k(path): 
    path = Path(path)

    with jsonlines.open(path) as fle: 
        data = [line for line in fle.iter()]

    
    for i in range(len(data)): 
        solution = data[i]["answer"]
        key = "####"
        idx = solution.find(key) + len(key) + 1

        data[i]["text"] = "# " + data[i]["question"]
        data[i]["code"] = ""
        data[i]["dsl_code"] = ""
        data[i]["reasoning"] = solution
        data[i]["answer"] = int(solution[idx:].replace(',',''))
        data[i]["task_id"] = i
        data[i].pop("question")

    instance_list = [MathQAInstance(**dct) for dct in data]

    return instance_list
