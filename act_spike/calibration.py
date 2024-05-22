from functools import partial
from collections import defaultdict

import torch
from transformers import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from .parsing_utils import get_linear_names, get_layers
from .inference import inference_per_layers_disk


@torch.no_grad()
def get_act_dict(model,
                 calib_dataset: torch.LongTensor,
                 offset: int = 0,
                 use_safetensors: bool = True,
                 stop_layer=-1):
    
    assert model.dtype == torch.float16

    act_dict = defaultdict(list)
    hidden_states = defaultdict(list)
    router_logits = defaultdict(list)

    linear_names = get_linear_names(model)
    layer_class = type(get_layers(model)[0])

    def calib_hook(m, x, y, name):
        x = x[0]
        if len(x.shape) == 3:
            x = x[:, offset:, :]  # skip initial tokens
        elif len(x.shape) == 2:
            x = x[offset:, :]  # skip initial tokens
        x = x.detach().abs().amax(-1).cpu()  # token-wise absmax
        
        act_dict[name].append(x)
    
    def hidden_hook(m, x, y, name):
        h = y[0].detach().abs().amax(-1).cpu()
        
        hidden_states[name].append(h)

    def router_logits_hook(m, x, y, name):
        # for mixtral
        _router_logits = y[1].detach().cpu()

        router_logits[name].append(_router_logits)
    
    
    hooks = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, layer_class):
            hook = module.register_forward_hook(partial(hidden_hook, name=name))
            hooks['hidden'].append(hook)

        if sum([ name.endswith(t) for t in linear_names ]):
            hook = module.register_forward_hook(partial(calib_hook, name=name))
            hooks['calib'].append(hook)

        if isinstance(module, MixtralSparseMoeBlock):
            hook = module.register_forward_hook(partial(router_logits_hook, name=name))
            hooks['router_logits'].append(hook)
    
    num_hooks = len(hooks['calib'])
    print(f'[calibration] register {num_hooks} hooks.')

    _ = inference_per_layers_disk(model, calib_dataset,
                                  use_safetensors=use_safetensors,
                                  stop_layer=stop_layer)

    for hook in (hooks['hidden'] + hooks['calib'] + hooks['router_logits']):
        hook.remove()

    for key, tensors in hidden_states.items():
        hidden_states[key] = torch.cat(tensors)

    for key, tensors in act_dict.items():
        # mixtral: list of (seq,)
        # else: (1, seq)
        if tensors[0].dim() == 2:
            act_dict[key] = torch.cat(tensors)

    for key, tensors in router_logits.items():
        router_logits[key] = torch.stack(tensors)

    if len(router_logits) == 0:
        router_logits = None
        
    return act_dict, hidden_states, router_logits


@torch.no_grad()
def get_past_key_values(model,
                        input_ids: torch.LongTensor,
                        use_safetensors: bool = True):

    assert input_ids.size(0) == 1
    
    past_key_values = list()

    layers = get_layers(model)
    
    def cache_hook(m, x, y):
        # print(y)
        # kv = y[-1]
        # kv = (kv[0].cpu(), kv[1].cpu())
        past_key_values.append(y[-1])

    hooks = []

    for layer in layers:
        hook = layer.register_forward_hook(cache_hook)
    
    num_hooks = len(hooks)
    print(f'[get_past_key_values] register {num_hooks} hooks.')

    _ = inference_per_layers_disk(model, input_ids,
                                  use_safetensors=use_safetensors,
                                  use_cache=True)
    
    for hook in hooks:
        hook.remove()
    
    return past_key_values[-1].to_legacy_cache()
