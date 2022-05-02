import re 
import numpy as np

def batch_loader(seq, size):
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


def gptneo_tokens_to_programs(outs, input_length, tokenizer): 
    generated_ids = [ids[input_length:] for ids in outs]
    untrunced_bodies = [tokenizer.decode(sample, skip_specials_tokens=False)
            for sample in generated_ids]

    untrunced_bodies = [x.replace("<|endoftext|>", "") for x in untrunced_bodies]

    re_key = '^answer.*?\n'

    bodies = [completion[:re.search(re_key, completion).span()[1]]
        if re.search(re_key, completion) else completion
        for completion in untrunced_bodies]

    return bodies
