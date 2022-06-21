import sys 
import os
import yaml 
import math
import json
from tqdm import tqdm

from etk.data.mathqa_dataset import MathQATrainSet, read_gsm8k
from etk.eval_utils import tokens_to_gsm8k_log_entry, batch_loader

import torch 
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

import transformers
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer 
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names

prompt = ["You are a world renowned philosopher. This your seminal treatise on formal epistemology:", "You are a mediocre psychologist. here is one of your papers:"]

def load_trainset_from_log(result, 
                           tokenizer, 
                           train_max_length, 
                           train_on_dev = False): 
    """
    log dictionary formatted exactly as before
    """

    log = result["log"]

    if not train_on_dev: 
        # Filters dev set
        with open("../data/gsm8k/dev_idxs.json") as f: 
            dev_idxs = json.load(f)

        log = [x for x in log if x["task_id"] not in dev_idxs]
    
    # Filters out examples with no solutions
    filtered_log = [x for x in log if True in x["passed_lst"]]

    dataset = MathQATrainSet(filtered_log, tokenizer, max_length)

    return dataset


class WrappedModel(torch.nn.Module): 
    def __init__(self, model, param_dict): 
        super().__init__()
        self.model = model 
        self.gen_params = param_dict

    def forward(self, x, y): 
        return self.model.generate(input_ids=x, attention_mask=y, **self.gen_params)


def evaluate(model, 
             tokenizer, 
             eval_dataset, 
             few_shot, 
             input_max_length, 
             output_max_length, 
             device_list, 
             temp, 
             inference_batch_size, 
             num_samples
             ): 
    """
    kwargs: 
    few_shot : Bool
    input_max_length : int 
    output_max_length : int
    tokenizer
    model
    device_list : List<int>
    temp : float
    num_samples : int
    inference_batch_size : int
    eval_dataset : List<MathQAInstance>
    """
    first_device = f"cuda:{device_list[0]}"

    model = AutoModelForCausalLM.from_pretrained("/home/lily/zaa7/ETK/etk/full_supervision/train_results/codex_teacher_pass100_gptneo125M_student_validation_pass1_standard_params/checkpoint-28420")
    wrapped_model = WrappedModel(model, 
                                   {"do_sample": True, 
                                    "temperature": temp, 
                                    "max_new_tokens": output_max_length, 
                                    "num_return_sequences": num_samples, 
                                    "pad_token_id": tokenizer.eos_token_id,}
                                   ).to(first_device)
    net = wrapped_model
    net = torch.nn.DataParallel(model, device_ids=device_list)

    dataloader = batch_loader(eval_dataset, inference_batch_size)
    
    eval_log = []
    for batch in tqdm(dataloader): 
        labels = [x.answer for x in batch]
        prompts = [few_shot_prompt + "\n" + x.text if few_shot else x.text 
                for x in batch]
        task_ids = [x.task_id for x in batch]
        texts = [x.text for x in batch]

        batch_length = len(labels)

        inpt = tokenizer(prompts, 
                                  return_tensors="pt", 
                                  max_length=input_max_length, 
                                  truncation=True, 
                                  padding='max_length', 
                                  )#.to(first_device)

        output = net(inpt["input_ids"], inpt["attention_mask"],)
        sys.exit()

        trunced_outs = [x[input_max_length:] for x in output]

        batched_trunced_outs = batch_loader(trunced_outs, batch_length)

        for tensors, label, task_id, text in zip(batched_trunced_outs, 
                                                 labels, 
                                                 task_ids, 
                                                 texts
                                                 ): 
            log_entry = tokens_to_gsm8k_log_entry(tensors, 
                                                  label, 
                                                  task_id, 
                                                  text, 
                                                  0, 
                                                  tokenizer,
                                                  "gptneo", 
                                                  verbose=True,
                                                  )

            eval_log.append(log_entry)

    return eval_log


def main(): 
    device_list = [4,5]

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token


    eval_dataset = read_gsm8k("../data/gsm8k/gsm8k_dev.jsonl")

    evaluate(model=None, 
             tokenizer=tokenizer, 
             eval_dataset=eval_dataset, 
             few_shot=False,
             input_max_length=200, 
             output_max_length=200, 
             device_list=[4,5],
             temp=0.6, 
             inference_batch_size=len(device_list),
             num_samples=20, 
             )
             
            

if __name__=="__main__": 
    main()




