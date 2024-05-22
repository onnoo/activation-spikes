import json
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from act_spike import get_c4_train, load_quantized_model, get_prefix_ids
from act_spike.parsing_utils import expand_module_names
from act_spike.utils import load_past_key_values


def parse_result_path(output_dir, args):
    result_dir = output_dir.joinpath('bench', f'seqlen_{args.seqlen}')

    if args.fp16:
        sub_dir = 'fp16'
    else:
        sub_dir = 'w8a8'

        if args.act_granul == 'per-token':
            sub_dir = sub_dir + '--per_token'
        elif args.act_granul == 'static':
            sub_dir = sub_dir + '--static'
        
        if args.use_cache:
            sub_dir = sub_dir + '--use_cache'
        
        if args.except_layer:
            sub_dir = sub_dir + '--except_layer'
    
    result_dir = result_dir.joinpath(sub_dir)
    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    fname = f'iter_{args.iter}.json'
    result_path = result_dir.joinpath(fname)

    return result_path

@torch.no_grad()
def benchmark_latency(model, dataset, past_key_values=None):
    model.eval()
    cudnn.benchmark = True

    n_samples = dataset.size(0)

    # Warm up
    for i in range(1):
        input_ids = dataset[i].unsqueeze(0).to('cuda')
        outputs = model(input_ids, past_key_values=past_key_values)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in tqdm(range(0, n_samples), ncols=80):
        input_ids = dataset[i].unsqueeze(0).to('cuda')
        # input_ids = torch.LongTensor(token_ids[a:b]).unsqueeze(0).to('cuda')
        outputs = model(input_ids, past_key_values=past_key_values)
    
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)

    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(0, n_samples), ncols=80):
        input_ids = dataset[i].unsqueeze(0).to('cuda')
        outputs = model(input_ids, past_key_values=past_key_values)
    
    memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    time = time / n_samples
    
    print(f"Peak memory usage: {memory:.2f} MB")
    print(f"inference time: {time:.5f} ms")
    return {'memory': memory, 'time': time}


def main(pretrained,
         seqlen,
         fp16,
         n_samples,
         act_granul,
         use_cache,
         output_dir,
         except_layer,
         result_path,
         **kwargs):
    
    bench_calib_dataset_path = output_dir.joinpath('bench_calib_dataset.pt')
    past_key_values_path = output_dir.joinpath('past_key_values.pt')
    except_layer_fname = f'final_except_layer_{pretrained.split("/")[-1]}'
    except_layer_path = output_dir.joinpath(f'../../../hayunkim/final_except_layer/{except_layer_fname}')

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    load_kwargs = {
        'act_granul': act_granul
    }

    if except_layer:
        data = json.load(except_layer_path.open())
        except_layer = [ 'model.' + name for name, _ in data ]
        load_kwargs['except_layer'] = except_layer  # NOTE: partial linears
    
    if fp16:
        model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16, device_map='cuda')
    else:
        if act_granul == 'static':
            if use_cache:
                act_dict_path = output_dir.joinpath('scale_dict_cached.pt')
            else:
                act_dict_path = output_dir.joinpath('scale_dict.pt')
            act_dict = torch.load(act_dict_path)
            input_scale_dict = { key: value.to('cuda') / 127 for key, value in act_dict.items() }
            
            load_kwargs['input_scale_dict'] = input_scale_dict  # not expanded
        
        model = load_quantized_model(pretrained, **load_kwargs)

    if bench_calib_dataset_path.exists():
        calib_dataset = torch.load(bench_calib_dataset_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        calib_dataset = get_c4_train(tokenizer,
                                     n_samples=512,
                                     seed=0,
                                     seqlen=2048)
        torch.save(calib_dataset, bench_calib_dataset_path)
    
    calib_dataset = calib_dataset[:n_samples, :seqlen]

    if use_cache:
        past_key_values = load_past_key_values(model, past_key_values_path)
    else:
        past_key_values = None
    
    results = benchmark_latency(model, calib_dataset, past_key_values=past_key_values)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', type=str)
    parser.add_argument('--seqlen', type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--act_granul', type=str, default='per-tensor')
    parser.add_argument('--n_samples', type=int, default=512)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--except_layer', action='store_true')
    parser.add_argument('--iter', type=int, default=0)

    args = parser.parse_args()
    arg_dict = args.__dict__

    model_name = args.pretrained.replace('/', '--')
    output_dir = Path('./outputs', model_name)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print(json.dumps(arg_dict, indent=4))
    
    result_path = parse_result_path(output_dir, args)

    main(**arg_dict, output_dir=output_dir, result_path=result_path)
