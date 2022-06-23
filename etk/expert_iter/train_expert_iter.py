import sys
import os
import yaml
import math
import json
from tqdm import tqdm

import contextlib

from etk.data.mathqa_dataset import MathQATrainSet, read_gsm8k
from etk.expert_iter.local_eval_utils import tokens_to_gsm8k_log_entry, batch_loader

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

import transformers
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names

from parallelformers import parallelize


def load_trainset_from_log(result, tokenizer, train_max_length, train_on_dev=False):
    """
    log dictionary just a list
    """
    # Filters out examples with no solutions
    dataset = MathQATrainSet(result, tokenizer, train_max_length)

    return dataset


def gptneo_data_collator(data):
    return {
        "input_ids": torch.stack([f[0] for f in data]),
        "attention_mask": torch.stack([f[1] for f in data]),
        "labels": torch.stack([f[0] for f in data]),
    }


def train_m0(model, epochs, tokenizer, batch_size, save_dir, dataset):
    model.cuda()
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        logging_strategy="epoch",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1)

    Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=gptneo_data_collator,
        optimizers=(optimizer, scheduler),
    ).train()

    return model.to("cpu")


def sampling_step(
    model,
    tokenizer,
    few_shot,
    input_max_length,
    output_max_length,
    temp,
    num_samples,
    inference_batch_size,
    solved,
    unsolved,
):
    """
    kwargs:
    few_shot : Bool
    input_max_length : int
    output_max_length : int
    tokenizer
    model
    temp : float
    num_samples : int
    inference_batch_size : int
    eval_dataset : List<Dict>

    Keep in mind this works with eval logs, not List<MathQInstance>
    """
    #parallelize(model, num_gpus=2, fp16=False, verbose=None)
    model.cuda()
    dataloader = batch_loader(unsolved, inference_batch_size)

    new_solved = []
    for batch in tqdm(dataloader):
        labels = [x["answer"] for x in batch]
        prompts = [x["text"] for x in batch]
        task_ids = [x["task_id"] for x in batch]
        texts = [x["text"] for x in batch]

        batch_length = len(labels)

        inpt = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=input_max_length,
            truncation=True,
            padding="max_length",
        ).to("cuda")

        output = model.generate(
            **inpt,
            do_sample=not temp == 0,
            temperature=None if temp == 0 else temp,
            max_new_tokens=output_max_length,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

        trunced_outs = [x.to("cpu") for x in output]

        batched_trunced_outs = batch_loader(trunced_outs, num_samples)

        for tensors, label, task_id, text in zip(
            batched_trunced_outs, labels, task_ids, texts
        ):
            log_entry, _ = tokens_to_gsm8k_log_entry(
                tensors,
                label,
                task_id,
                text,
                0,
                tokenizer,
                "gptneo",
                verbose=False,
            )
            
            if log_entry["gold_solution"]: 
                new_solved.append(log_entry)

    new_solved_task_ids = [x["task_id"] for x in new_solved]
    updated_unsolved = [x for x in unsolved if x["task_id"] not in new_solved_task_ids]

    return solved + new_solved, updated_unsolved


def main():
    num_iters=10
    results_dir = "results/gptneo1B_ei"
    os.mkdir(results_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token

    with open("../few_shot/results/gptneo1B_pass100_train.json") as f: 
        to_init = json.load(f)["log"]

    # filter out dev set
    with open("../data/gsm8k/dev_idxs.json") as f:
        dev_idxs = json.load(f)

    to_init = [x for x in to_init if x["task_id"] not in dev_idxs]

    solved = [x for x in to_init if x["gold_solution"]]
    unsolved = [x for x in to_init if not x["gold_solution"]]
    print(f"NUM SOLVED: {len(solved)}")
    print(f"NUM UNSOLVED: {len(unsolved)}")

    model=None
    for i in range(num_iters): 
        print(f"ITERATION {i}")

        del model
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", output_loading_info=False)
        
        train_data = load_trainset_from_log(solved, tokenizer, train_max_length=275)

        model = train_m0(
            model=model,
            epochs=4, 
            tokenizer=tokenizer, 
            batch_size=1 if len(solved)<100 else 4,
            save_dir=os.path.join(results_dir, f"train{i}"), 
            dataset=train_data,
        )

        solved, unsolved = sampling_step(
            model=model,
            tokenizer=tokenizer,
            few_shot=True,
            input_max_length=175,
            output_max_length=100,
            temp=0.6,
            inference_batch_size=1,
            num_samples=100,
            solved=solved, 
            unsolved=unsolved,
        )

        print(f"NUM SOLVED: {len(solved)}")
        print(f"NUM UNSOLVED: {len(unsolved)}")

        with open(os.path.join(results_dir, f"solved{i}.json"), "w") as f:
            json.dump(solved, f)

if __name__ == "__main__":
    main()
