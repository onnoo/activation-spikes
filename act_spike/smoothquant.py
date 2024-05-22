from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm

from .parsing_utils import get_linear_names
from .inference import inference_per_layers_disk


@torch.no_grad()
def get_sq_act_dict(model,
                    calib_dataset: torch.LongTensor,
                    offset: int = 0,
                    use_safetensors: bool = True):
    
    assert model.dtype == torch.float16

    act_dict = defaultdict(list)
    linear_names = get_linear_names(model)

    def calib_hook(m, x, y, name):
        x = x[0]
        hidden_dim = x.shape[-1]
        tensor = x.view(-1, hidden_dim).abs().detach()
        tensor = tensor[offset:, :]  # skip initial tokens
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()

        if name in act_dict:
            act_dict[name] = torch.max(act_dict[name], comming_max)
        else:
            act_dict[name] = comming_max
    

    hooks = defaultdict(list)
    for name, module in model.named_modules():
        if sum([ name.endswith(t) for t in linear_names ]):
            hook = module.register_forward_hook(partial(calib_hook, name=name))
            hooks['calib'].append(hook)
    
    num_hooks = len(hooks['calib'])
    print(f'[calibration] register {num_hooks} hooks.')

    _ = inference_per_layers_disk(model, calib_dataset,
                                  use_safetensors=use_safetensors)

    for hook in (hooks['hidden'] + hooks['calib']):
        hook.remove()

    return act_dict


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, GemmaRMSNorm))
    for fc in fcs:
        # assert isinstance(fc, (nn.Linear, QLinear))
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, act_dict, alpha=0.5):

    for name, module in model.named_modules():
        if isinstance(module, (LlamaDecoderLayer, GemmaDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = act_dict[name + '.self_attn.q_proj']

            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = act_dict[name + '.mlp.up_proj']

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
