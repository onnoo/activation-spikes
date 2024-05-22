from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from tqdm import tqdm

from .migration_llama import migration
from ..safetensor_utils import CachedTensor, CPUTensor, split_state_dict
from ..parsing_utils import get_layers, get_embeddings, get_norm
from ..inference import _assign_var
from ..quant import QLinear


def qkv_migration(m, x, hidden_states, self, cache, a_qconfig, w_qconfig):
    """
    layernorm forward hook
    """
    weight_list = torch.cat([self.self_attn.q_proj.weight,
                             self.self_attn.k_proj.weight,
                             self.self_attn.v_proj.weight])
    
    extra_dict = {
        'num_heads': self.self_attn.num_heads,
        'num_key_value_heads': self.self_attn.num_key_value_heads,
        'num_key_value_groups': self.self_attn.num_key_value_groups,
        'head_dim': self.self_attn.head_dim,
        'position_ids': cache['position_ids'],
        'attention_mask': cache['attention_mask'],
        'past_key_value': cache['past_key_value'],
        'cache_position': cache['cache_position'],
        'layer_idx': self.self_attn.layer_idx,
        'rotary_emb': self.self_attn.rotary_emb,
    }
    
    best_scale = migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'qkv', extra_dict)
    
    hidden_states /= best_scale
    self.self_attn.q_proj.weight.data *= best_scale
    self.self_attn.k_proj.weight.data *= best_scale
    self.self_attn.v_proj.weight.data *= best_scale
    
    return hidden_states


def up_and_gate_migration(m, x, hidden_states, self, a_qconfig, w_qconfig):
    """
    layernorm forward hook
    """
    weight_list = torch.cat([self.mlp.gate_proj.weight,
                             self.mlp.up_proj.weight])
    
    extra_dict = {
        'act_fn': self.mlp.act_fn
    }

    best_scale = migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'up_and_gate', extra_dict)
    hidden_states /= best_scale
    self.mlp.gate_proj.weight.data *= best_scale
    self.mlp.up_proj.weight.data *= best_scale

    return hidden_states


def down_proj_migration(m, x, self, a_qconfig, w_qconfig):
    """
    down_proj forward pre hook
    """
    hidden_states = x[0]
    weight_list = torch.cat([self.mlp.down_proj.weight])
    
    extra_dict = {}

    best_scale = migration(hidden_states, weight_list, a_qconfig, w_qconfig, 'down_proj', extra_dict)
    hidden_states /= best_scale
    self.mlp.down_proj.weight.data *= best_scale

    return hidden_states


@torch.no_grad()
def get_migrate_scales(model,
                       input_ids: torch.LongTensor,
                       weight_quant: str = 'per-channel',
                       device: str ='cuda',
                       use_cache: bool = False,
                       except_layer: list = [],
                       use_safetensors: bool = True,
                       past_key_values = None,
                       verbose: bool = True):
    """
    OSP migration export
    act_scheme = per-tensor
    weight_scheme = per-channel
    """

    a_qconfig = SimpleNamespace(**{
        'quantizer': 'FixedFakeQuantize',
        'observer': 'MinMaxObserver',
        'bit': 8,
        'symmetric': True,
        'ch_axis': -1
    })
    w_qconfig = SimpleNamespace(**{
        'quantizer': 'FixedQuantize',
        'observer': 'MinMaxObserver',
        'bit': 8,
        'symmetric': True,
        'ch_axis': 0  if weight_quant == 'per-channel' else -1
    })

    decoder_prefix = model.base_model_prefix

    for name, p in model.named_parameters():
        p.requires_grad = False
    
    for name, m in model.named_modules():
        modules_names = name.split('.')
        
        if isinstance(m, nn.Linear) and name.startswith(decoder_prefix):

            parent_module_name = name[:name.rindex('.')]
            parent_module = model.get_submodule(parent_module_name)

            child_module = QLinear(m, granul='per-channel', act_granul='per-tensor')
            setattr(parent_module, modules_names[-1], child_module)

    ## prepare weights
    pretrained = model.name_or_path
    if use_safetensors:
        cached_tensor = CachedTensor(pretrained)
    else:
        cached_tensor = CPUTensor(pretrained)
    
    sub_info = split_state_dict(model)
    sub_info = { k.split('.')[-1]: v for k, v in sub_info.items() }
    
    sub_dict = cached_tensor.infer_state_dict(sub_info['none'], device='cuda', dtype=model.dtype)
    model.load_state_dict(sub_dict, strict=False, assign=True)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    
    batch_size, seqlen = input_ids.size()
    conf_use_cache = model.config.use_cache
    model.config.use_cache = use_cache
    
    layers = get_layers(model)
    embeddings = get_embeddings(model)
    
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(device)
    
    dtype = next(model.parameters()).dtype
    cache = {}
    
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, hidden_states, **kwargs):
            cache['hidden_states'] = hidden_states
            cache.update(kwargs)

            raise ValueError

    layers[0] = Catcher(layers[0])
    
    try:
        model(input_ids.to('cuda'), past_key_values=past_key_values)
    except ValueError:
        pass
    
    layers[0] = layers[0].module
    hidden_states = cache['hidden_states']
    del cache['hidden_states']
    
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()
    
    for i in tqdm(range(model.config.num_hidden_layers), desc='Layer', disable=not verbose):
        layer = layers[i]
        sub_dict = cached_tensor.infer_state_dict(sub_info[str(i)], device=device, dtype=model.dtype)

        model.load_state_dict(sub_dict, strict=False, assign=True)

        for name, buffer in layer.named_buffers():
            _assign_var(layer, name, buffer.cuda())
        
        for name, m in layer.named_parameters():
            m.requires_grad = False
        
        if f'model.layers.{i}.self_attn.q_proj' not in except_layer:
            ### regsiter
            qkv_hook = partial(qkv_migration,
                               self=layer,
                               cache=cache,
                               a_qconfig=a_qconfig,
                               w_qconfig=w_qconfig)
            layer.input_layernorm.register_forward_hook(qkv_hook)

        if f'model.layers.{i}.mlp.up_proj' not in except_layer:
            up_and_gate_hook = partial(up_and_gate_migration,
                                       self=layer,
                                       a_qconfig=a_qconfig,
                                       w_qconfig=w_qconfig)
            layer.post_attention_layernorm.register_forward_hook(up_and_gate_hook)

        if f'model.layers.{i}.mlp.down_proj' not in except_layer:
            down_proj_hook = partial(down_proj_migration,
                                     self=layer,
                                     a_qconfig=a_qconfig,
                                     w_qconfig=w_qconfig)
            layer.mlp.down_proj.register_forward_pre_hook(down_proj_hook)

        ### forward
        hidden_states = layer(hidden_states, **cache)[0]
        
        for name, param in sub_dict.items():
            sub_dict[name] = param.data.to('meta')
        model.load_state_dict(sub_dict, strict=False, assign=True)
        
        torch.cuda.empty_cache()
    
    norm = get_norm(model)
    if norm is not None:
        norm = norm.to(device)
    hidden_states = norm(hidden_states)
    
    from . import migration_llama
    scale_list = migration_llama.scale_list
    for i in range(len(scale_list)):
        scale_list[i] = scale_list[i].cpu()

    return scale_list

