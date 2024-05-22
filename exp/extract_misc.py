import json
import argparse
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from act_spike import *
from act_spike.smoothquant import get_sq_act_dict
from act_spike.osp.utils import get_migrate_scales
from act_spike.utils import load_past_key_values


def main(pretrained,
         dtype,
         batch_size,
         seqlen,
         n_samples,
         seed,
         weight_quant,
         no_safetensors,
         output_dir):
    
    dtype = getattr(torch, dtype)

    calib_dataset_path = output_dir.joinpath('calib_dataset.pt')
    act_dict_path = output_dir.joinpath('act_dict.pt')
    hidden_states_path = output_dir.joinpath('hidden_states.pt')
    router_logits_path = output_dir.joinpath('router_logits.pt')
    prefix_dict_path = output_dir.joinpath('prefix_dict.json')
    past_key_values_path = output_dir.joinpath('past_key_values.pt')

    sq_act_dict_path = output_dir.joinpath('sq_act_dict.pt')
    sq_act_dict_cached_path = output_dir.joinpath('sq_act_dict_cached.pt')
    osp_scale_dict_path = output_dir.joinpath('osp_scale_dict.pt')
    osp_scale_dict_cached_path = output_dir.joinpath('osp_scale_dict_cached.pt')

    ### (1) prepare calibration dataset
    if calib_dataset_path.exists():
        calib_dataset = torch.load(calib_dataset_path)
        print('Load calib_dataset from', calib_dataset_path)

    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        prefix_ids = get_prefix_ids(tokenizer)
        calib_dataset = get_c4_train(tokenizer,
                                     n_samples=n_samples,
                                     seed=seed,
                                     seqlen=seqlen,
                                     prefix_ids=prefix_ids)
        torch.save(calib_dataset, calib_dataset_path)
    
    ### (2) get act dict
    if act_dict_path.exists() and hidden_states_path.exists():
        act_dict = torch.load(act_dict_path)
        if router_logits_path.exists():
            router_logits = torch.load(router_logits_path)
        else:
            router_logits = None
        print(f'Load act_dict from', act_dict_path)

    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        act_dict, hidden_states, router_logits = get_act_dict(model,
                                                              calib_dataset,
                                                              use_safetensors=not no_safetensors)
        
        torch.save(act_dict, act_dict_path)
        torch.save(hidden_states, hidden_states_path)

        if router_logits is not None:
            torch.save(router_logits, router_logits_path)
    
    ### (3) QFeM: search threshold
    pass

    ### (4) QFeP: search prefix
    if prefix_dict_path.exists():
        with open(prefix_dict_path) as f:
            prefix_dict = json.load(f)
        print(f'Load prefix_dict from', prefix_dict_path)
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        
        prefix_dict = generate_prefix(tokenizer, model, calib_dataset, act_dict, router_logits)

        with open(prefix_dict_path, 'w') as f:
            json.dump(prefix_dict, f, indent=2)

    ### (5) QFeP: generate prefix
    if past_key_values_path.exists():
        past_key_values = torch.load(past_key_values_path)
        print(f'Load past_key_values from', past_key_values_path)
    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        
        prefix_ids = torch.LongTensor(prefix_dict['meta']['prefix_ids']).unsqueeze(0)
        past_key_values = get_past_key_values(model, prefix_ids)
        torch.save(past_key_values, past_key_values_path)

    ### (6) SmoothQuant: calibrate 
    if sq_act_dict_path.exists():
        print('sq_act_dict: [OK]')
    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        prefix_ids = get_prefix_ids(tokenizer)
        calib_dataset = get_c4_train(tokenizer, prefix_ids=prefix_ids)
        sq_act_dict = get_sq_act_dict(model, calib_dataset)
        torch.save(sq_act_dict, sq_act_dict_path)
    
    if sq_act_dict_cached_path.exists():
        print('sq_act_dict_cached: [OK]')
    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        prefix_ids = prefix_dict['meta']['prefix_ids']
        calib_dataset = get_c4_train(tokenizer, prefix_ids=prefix_ids)
        sq_act_dict = get_sq_act_dict(model, calib_dataset, offset=len(prefix_ids))
        torch.save(sq_act_dict, sq_act_dict_cached_path)
    
    ### (7) OutlierSuppressionPlus: calc best migration scale
    if osp_scale_dict_path.exists():
        print('osp_scale_dict: [OK]')
    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        prefix_ids = get_prefix_ids(tokenizer)
        calib_dataset = get_c4_train(tokenizer, prefix_ids=prefix_ids, n_samples=16)

        scale_list = get_migrate_scales(model, calib_dataset, weight_quant=weight_quant)
        module_templates = [
            'model.layers.{}.input_layernorm',
            'model.layers.{}.post_attention_layernorm',
            'model.layers.{}.mlp.up_proj',
        ]
        num_layers = model.config.num_hidden_layers
        all_names = []

        for i in range(num_layers):
            names = [template.format(i) for template in module_templates]
            all_names.extend(names)
        
        osp_scale_dict = dict(zip(all_names, scale_list))
        torch.save(osp_scale_dict, osp_scale_dict_path)
    
    if osp_scale_dict_cached_path.exists():
        print('osp_scale_dict_cached: [OK]')
    else:
        config = AutoConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)

        prefix_ids = prefix_dict['meta']['prefix_ids']
        num_osp_calib_samples = 16
        calib_dataset = get_c4_train(tokenizer, prefix_ids=prefix_ids, n_samples=num_osp_calib_samples)

        state_shape = past_key_values[0][0].shape

        past_key_values_batch = tuple([
            tuple([tensor.expand(num_osp_calib_samples, *state_shape[1:]) for tensor in state])
            for state in past_key_values
        ])

        scale_list = get_migrate_scales(model,
                                        calib_dataset,
                                        weight_quant=weight_quant,
                                        use_cache=True,
                                        past_key_values=past_key_values_batch)
        module_templates = [
            'model.layers.{}.input_layernorm',
            'model.layers.{}.post_attention_layernorm',
            'model.layers.{}.mlp.up_proj',
        ]
        num_layers = model.config.num_hidden_layers
        all_names = []

        for i in range(num_layers):
            names = [template.format(i) for template in module_templates]
            all_names.extend(names)
        
        osp_scale_dict_cached = dict(zip(all_names, scale_list))
        torch.save(osp_scale_dict_cached, osp_scale_dict_cached_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', type=str)
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seqlen', type=int, default=1024, help='length of calibration example')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight_quant', type=str, default='per-channel')
    parser.add_argument('--n_samples', type=int, default=512)
    parser.add_argument('--no_safetensors', action='store_true')

    args = parser.parse_args()
    arg_dict = args.__dict__

    model_name = args.pretrained.replace('/', '--')
    output_dir = Path('./outputs', model_name)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    with output_dir.joinpath('args.json').open('w') as f:
        json.dump(arg_dict, f, indent=4)
    
    print(json.dumps(arg_dict, indent=4))

    main(**arg_dict, output_dir=output_dir)
