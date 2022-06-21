import torch 
import json 
import jsonlines
from pathlib import Path 

class MathQATrainSet(torch.utils.data.Dataset): 
    def __init__(self, log_lst, tokenizer, max_length, use_teacher=False):
        self.log_lst = log_lst
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_teacher = use_teacher

    def __getitem__(self, idx): 
        dct = self.log_lst[idx]

        full_text = dct["text"] + "\n" + dct["gold_solution"]

        full_text_encode = self.tokenizer(full_text, 
                                          max_length=self.max_length, 
                                          truncation=True, 
                                          padding='max_length', 
                                          return_tensors="pt"
                                          )
        if self.use_teacher:
            if "idx" in dct.keys():
                full_text_encode["idx"] = dct["idx"]
            return full_text_encode
        ids = full_text_encode['input_ids'].squeeze()
        mask = full_text_encode['attention_mask'].squeeze()

        return ids.long(), mask.long()

    def __len__(self): 
        return len(self.log_lst)

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

def read_gsm8k_with_code(path):
    """Reads a few-shot run's output and returns a list of MathQAInstance objects. Only returns solved instances with code.
    
    Args:
        path (str): Path to the jsonlines file containing the data. TODO: make this take in a json file, not jsonlines.
    """
    path = Path(path)

    with jsonlines.open(path) as fle:
        data = [line for line in fle.iter()]

    for i in range(len(data)):
        if data[i]["gold_solution"] is not False:
            # data[i]["text"]
            data[i]["code"] = ""
            data[i]["dsl_code"] = ""
            data[i]["reasoning"] = data[i]["gold_solution"]
            # data[i]["answer"]
            # data[i]["task_id"]
            data[i].pop("gold_solution")
            data[i].pop("pass1")
            data[i].pop("passk")
            data[i].pop("passed_lst")

            
    instance_list = [MathQAInstance(**dct) for dct in data if "dsl_code" in dct.keys()] # only return instances with code

    return instance_list

class MathQAPython(torch.utils.data.Dataset): 
    def __init__(self, instance_list, tokenizer, max_length): 
        self.data = instance_list 
        self.tokenizer = tokenizer 
        self.max_length = max_length
    

    def __getitem__(self, idx): 
        instance = self.data[idx]
        full_text = instance.text + '\n' + instance.code
        answer = instance.answer

        full_text_encode = self.tokenizer(full_text, 
                max_length=self.max_length, truncation=True, 
                padding='max_length', return_tensors='pt')
        ids = full_text_encode['input_ids'].squeeze()
        mask = full_text_encode['attention_mask'].squeeze()

        return ids.long(), mask.long(), answer


    def __len__(self): 
        return len(self.data) 

