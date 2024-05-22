import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # suppress lm_eval warning

import torch
import lm_eval
from tqdm import tqdm

from .models import CacheHFLM


def evaluate(model,
             tokenizer,
             tasks,
             limit=None,
             max_length=None,
             prefix_ids=None,
             past_key_values=None):
    """
    tasks=['piqa', 'lambada_openai', 'wikitext', 'hellaswag', 'winogrande']
    past_key_values overrides prefix_ids.
    """

    device = next(model.parameters()).device

    lm = CacheHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        max_batch_size=1,
        max_length=max_length,
    )

    if past_key_values:
        lm.set_past_key_values(past_key_values)
    elif prefix_ids:
        prefix_ids = torch.LongTensor(prefix_ids).unsqueeze(0).to(device)
        lm.set_prefix_ids(prefix_ids)
    
    # lm_eval.tasks.initialize_tasks('./tasks/')
    task_dir = os.path.dirname(os.path.abspath(__file__)) + '/tasks/'
    lm_eval.tasks.include_path(task_dir)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
        write_out=False,
    )

    return results


@torch.no_grad()
def benchmark(model, dataset, past_key_values=None):
    model.eval()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if past_key_values is not None:
        past_key_value_len = past_key_values[0][0].size(2)
        dataset = dataset[:, past_key_value_len:]
    
    # Warmup (slow)
    input_ids = dataset[0].unsqueeze(0).cuda()
    outputs = model(input_ids, past_key_values=past_key_values)
    
    times = []
    max_memory = 0
    
    for input_ids in tqdm(dataset, ncols=80, ascii=True, desc='benchmark'):
        input_ids = input_ids.unsqueeze(0).cuda()
        torch.cuda.synchronize()
        start.record()
        outputs = model(input_ids, past_key_values=past_key_values)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)

    results = {
        'times': times,
        'max_memory': max_memory
    }
    return results
