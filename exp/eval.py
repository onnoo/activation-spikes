import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from act_spike import evaluate, get_prefix_ids, load_quantized_model, get_layers
from act_spike.utils import load_past_key_values


def parse_result_dir(output_dir,
                     fp16,
                     use_cache,
                     sq,
                     osp,
                     weight_quant,
                     act_granul,
                     bmm,
                     except_layer):
    if fp16:
        out_str = 'fp16'
    else:
        out_str = 'w8a8'
        if weight_quant == 'per-tensor':
            out_str = 'per-tensor--' + out_str
        if act_granul == 'per-token':
            out_str = 'act-per-token--' + out_str
        elif act_granul == 'static':
            out_str = 'static--' + out_str
        if bmm:
            out_str = out_str + '--bmm'
        if sq:
            out_str = out_str + '+sq'
        elif osp:
            out_str = out_str + '+osp'
        
        if use_cache:
            out_str = out_str + '+cache'
        if except_layer:
            out_str = out_str + '+except_layer'

    return output_dir.joinpath(out_str)


def main(pretrained,
         tasks,
         use_cache,
         except_layer,
         fp16,
         sq,
         osp,
         bmm,
         weight_quant,
         act_granul,
         output_dir,
         results_dir):
    
    results_path = results_dir.joinpath('eval_results.json')

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    prefix_ids = get_prefix_ids(tokenizer)  # example: BOS
    tasks = tasks.split(',')
    
    if fp16:
        # NOTE: requires sufficient GPU memory
        model = AutoModelForCausalLM.from_pretrained(pretrained,
                                                     torch_dtype=torch.float16,
                                                     device_map='cuda')

        outputs = evaluate(model,
                           tokenizer,
                           tasks=tasks,
                           max_length=2000,
                           prefix_ids=prefix_ids)
        results = outputs['results']

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    else:
        # layer skip
        # cache
        past_key_values = None

        model_kwargs = {}

        if sq:
            if use_cache:
                sq_act_dict_path = output_dir.joinpath('sq_act_dict_cached.pt')
            else:
                sq_act_dict_path = output_dir.joinpath('sq_act_dict.pt')
            sq_act_dict = torch.load(sq_act_dict_path)

            model_kwargs['sq_act_dict'] = sq_act_dict
            model_kwargs['sq_alpha'] = 0.8
        
        if osp:
            if use_cache:
                osp_scale_dict_path = output_dir.joinpath('osp_scale_dict_cached.pt')
            elif except_layer:
                osp_scale_dict_path = output_dir.joinpath('osp_scale_dict_except.pt')
            else:
                osp_scale_dict_path = output_dir.joinpath('osp_scale_dict.pt')
            osp_scale_dict = torch.load(osp_scale_dict_path)
            for key, tensor in osp_scale_dict.items():
                osp_scale_dict[key] = tensor.to('cuda')

            model_kwargs['osp_scale_dict'] = osp_scale_dict
        
        if bmm:
            model_kwargs['quantize_bmm'] = True
        
        if act_granul == 'static':
            if use_cache:
                act_dict_path = output_dir.joinpath('scale_dict_cached.pt')
            else:
                act_dict_path = output_dir.joinpath('scale_dict.pt')
            act_dict = torch.load(act_dict_path)
            input_scale_dict = { key: value.to('cuda') / 127 for key, value in act_dict.items() }
            
            model_kwargs['input_scale_dict'] = input_scale_dict  # not expanded
        
        if except_layer:
            except_layer_fname = f'final_except_layer_{pretrained.split("/")[-1]}'
            except_layer_path = output_dir.joinpath(f'../../../hayunkim/final_except_layer/{except_layer_fname}')

            data = json.load(except_layer_path.open())
            except_layer = [ 'model.' + name for name, _ in data ]
            model_kwargs['except_layer'] = except_layer  # NOTE: partial linears
        
        model = load_quantized_model(pretrained,
                                     granul=weight_quant,
                                     act_granul=act_granul,
                                     **model_kwargs)
        print(next(model.parameters()).device)

        if use_cache:
            prefix_path = output_dir.joinpath(f'past_key_values.pt')
            past_key_values = load_past_key_values(model, prefix_path)
            
            # past_key_values를 저장하자
            # 중요 - 기준 무엇으로?
            # BOS토큰이 탑일 때에는?
        
        outputs = evaluate(model,
                           tokenizer,
                           tasks=tasks,
                           max_length=2000,
                           prefix_ids=prefix_ids,
                           past_key_values=past_key_values)
        results = outputs['results']

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', type=str)
    parser.add_argument('--tasks', type=str, default='wikitext,piqa,lambada_openai,hellaswag,winogrande')
    parser.add_argument('--weight_quant', type=str, default='per-channel')
    parser.add_argument('--act_granul', type=str, default='per-tensor')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--except_layer', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--sq', action='store_true')
    parser.add_argument('--osp', action='store_true')
    parser.add_argument('--bmm', action='store_true')

    args = parser.parse_args()
    arg_dict = args.__dict__

    model_name = args.pretrained.replace('/', '--')
    output_dir = Path('./outputs', model_name)
    results_dir = parse_result_dir(output_dir,
                                   args.fp16,
                                   args.use_cache,
                                   args.sq,
                                   args.osp,
                                   args.weight_quant,
                                   args.act_granul,
                                   args.bmm,
                                   args.except_layer)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with results_dir.joinpath('eval_args.json').open('w') as f:
        json.dump(arg_dict, f, indent=4)

    print(json.dumps(arg_dict, indent=4))

    main(**arg_dict, output_dir=output_dir, results_dir=results_dir)
