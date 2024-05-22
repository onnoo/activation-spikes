import random
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tqdm.auto import tqdm


def get_c4_train(tokenizer: PreTrainedTokenizerBase,
                 n_samples: int = 512,
                 seed: int = 0,
                 seqlen: int = 1024,
                 prefix_ids: Optional[List[int]] = []):
    
    train_dataset = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    random.seed(seed)
    train_samples = []
    
    for _ in tqdm(range(n_samples), desc='get_c4_train'):
        
        while True:
            i = random.randint(0, len(train_dataset) - 1)
            input_ids = tokenizer(train_dataset[i]['text'], add_special_tokens=False).input_ids
            if len(input_ids) > seqlen:
                break
        
        i = random.randint(0, len(input_ids) - seqlen - 1)
        j = i + seqlen
        input_ids = input_ids[i:j]
        
        if prefix_ids is not None:
            input_ids = prefix_ids + input_ids[:len(input_ids) - len(prefix_ids)]
        
        train_samples.append(input_ids)
    
    train_samples = torch.LongTensor(train_samples)
    
    return train_samples
