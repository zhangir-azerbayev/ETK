import re 
import numpy as np
from etk.execution import semisafe_evaluate

def batch_loader(seq, size):
    """
    Iterator that takes in a list `seq` and returns
    chunks of size `size` 
    """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def pass_k(lst, k): 
    """
    lst: Boolean list 
    k: value of pass@k to calculate. 
    """
    n = len(lst)
    c = sum(lst)
    if n - c < k: return 1.0 
    return 1.0 - np.prod(1.0 - k / 
                        np.arange(n-c+1, n+1))


def gptneo_tokens_to_gsm8k_programs(outs, input_length, tokenizer, verbose=False): 
    """
    Converts raw gpt-neo model outputs to executable programs

    outs: rank-2 tensor (sample, token) 
    input_length: length of prompt in tokens
    tokenizer: 
    """

    generated_ids = [ids[input_length:] for ids in outs]
    untrunced_bodies = [tokenizer.decode(sample, skip_special_tokens=False)
            for sample in generated_ids]
    
    if verbose: 
        for x in untrunced_bodies: 
            print("#"*40)
            print("untrunced")
            print(x)

    untrunced_bodies = [x.replace("<|endoftext|>", "") for x in untrunced_bodies]

    re_key = '\nanswer.*?\n'

    bodies = [completion[:re.search(re_key, completion).span()[1]]
        if re.search(re_key, completion) else completion
        for completion in untrunced_bodies]

    if verbose: 
        for x in bodies: 
            print("#"*40)
            print("trunced")
            print(x)

    return bodies

def incoder_tokens_to_gsm8k_programs(outs, input_length, tokenizer, verbose=False): 
    """
    Converts raw incoder output tensors to executable programs targeting gsm8k
    outs: rank-2 tensor (sample, token)
    input_length: length of prompt in tokens
    tokenizer: 
    """
    generated_ids = [ids[input_length:] for ids in outs]
    untrunced_bodies = [tokenizer.decode(sample, skip_specials_tokens=False)
            for sample in generated_ids]
    
    if verbose: 
        for x in untrunced_bodies: 
            print("#"*40)
            print("untrunced")
            print(x)

    eos = "<|endoftext|>"

    untrunced_bodies=[x.replace(eos, "").replace("<|", "").split("</cell>")[0] 
            for x in untrunced_bodies]

    re_key = '\nanswer.*?\n'

    bodies = [completion[:re.search(re_key, completion).span()[1]]
        if re.search(re_key, completion) else completion
        for completion in untrunced_bodies]

    if verbose: 
        for x in bodies: 
            print("#"*40)
            print("trunced")
            print(x)

    return bodies



def tokens_to_gsm8k_log_entry(outs, 
                               label, 
                               task_id,
                               text,
                               input_length, 
                               tokenizer, 
                               model_type,
                               verbose=False
                               ): 
    """
    `model_type` is `"incoder"` or `"gptneo"`
    """
    if verbose: 
        print("#"*40)
        print("text: ")
        print(text)
 
    if verbose: 
        for x in outs: 
            print("#"*40)
            print("raw decoded outputs")
            print(tokenizer.decode(x, skip_special_tokens=False))

    if model_type=="gptneo": 
        bodies = gptneo_tokens_to_gsm8k_programs(outs, 
                                                 input_length, 
                                                 tokenizer, 
                                                 verbose)
    elif model_type=="incoder": 
        bodies = incoder_tokens_to_gsm8k_programs(outs, 
                                                  input_length, 
                                                  tokenizer, 
                                                  verbose
                                                  )
    else: 
        raise ValueError("invalid `model_type`")

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

    log_entry = {"task_id": task_id,
                         "text": text,
                         "answer": label,
                         "gold_solution": gold_code,
                         "passk": passed,
                         "pass1": pass_1,
                         "passed_lst": passed_lst}

    return log_entry, bodies




