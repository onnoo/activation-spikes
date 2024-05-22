import math
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.cache_utils import Cache
from transformers import MixtralForCausalLM, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from .parsing_utils import get_layers, get_pre_norm_name_pairs, expand_module_names, get_sibling_names
from .safetensor_utils import CachedTensor, split_state_dict, CPUTensor
from .smoothquant import smooth_ln_fcs_llama_like


class QLinear(nn.Module):
    def __init__(self, module, granul, act_granul, except_layer=False, dtype=torch.float16):
        super().__init__()
        self.weight = module.weight
        self.in_features = module.in_features
        self.qmax = torch.iinfo(torch.int8).max
        self.input_act_scale = None
        self.weight_scale = None
        self.granul = granul
        self.act_granul = act_granul
        self.except_layer = except_layer
        self.dtype = dtype
        # self.weight_quant()
        # self.dtype = module.weight.dtype
        # self.device = module.weight.device

    def weight_quant(self):
        #### weight quant
        if self.granul == 'per-tensor':
            self.weight_scale = self.weight.abs().amax() / self.qmax
            self.weight.data = self.weight.data.div(self.weight_scale).round().type(torch.int8)
            # weight = self.weight.data.div(self.weight_scale).round().type(torch.int8)

        elif self.granul == 'per-channel':
            self.weight_scale = self.weight.abs().amax(dim=-1, keepdims=True) / self.qmax
            self.weight.data = self.weight.data.div(self.weight_scale).round().type(torch.int8)
            # weight = self.weight.data.div(self.weight_scale).round().type(torch.int8)
        
        else:
            print('weight granularity error')
        
        # self.weight.data = self.weight.data.to('cuda')

    def weight_dequant(self):
        #### weight dequant
        # self.weight.data = self.weight.data.mul(self.weight_scale).to(torch.float16)
        return self.weight.data.mul(self.weight_scale).to(torch.float16)

    def forward(self, input_act):
    
        #### weight_quant
        if self.weight.dtype is torch.int8:
            pass
        else:
            #### quantize
            self.weight_quant()

        if self.except_layer:
            dq_weight = self.weight_dequant()
            output = torch.matmul(input_act, dq_weight.T)
            del dq_weight

            return output

        #### input_act quant (if necessary)
        # input_act = input_act.squeeze(dim=0) # batch 1일 때
        if self.input_act_scale is None:
            if self.act_granul == 'per-tensor':
                self.input_act_scale = input_act.abs().amax() / self.qmax
                input_act = input_act.div(self.input_act_scale).round().type(torch.int8)
            elif self.act_granul == 'per-token':
                self.input_act_scale = input_act.abs().amax(dim=-1, keepdims=True) / self.qmax
                input_act = input_act.div(self.input_act_scale).round().type(torch.int8)
            else:
                raise f'Unknown quantization scheme for activation: {self.act_granul}'
        elif self.act_granul == 'static' and input_act.dtype != torch.int8:  # 2번 할 수도 있다
            input_act = input_act.div(self.input_act_scale).round().type(torch.int8)

        #### int matmul
        input_shape = input_act.shape
        input_act = input_act.reshape(-1, input_shape[-1])
        if input_act.size(0) > 16:
            output = torch._int_mm(input_act, self.weight.T)
        else:
            output = torch.mm(input_act.to(torch.float32),
                              self.weight.to(torch.float32).T)

        output = output.reshape(*input_shape[:-1], -1)
        #### dequantize
        scale = (self.input_act_scale * self.weight_scale.T).to(torch.float32)
        output = output.mul(scale).to(self.dtype)

        if self.act_granul != 'static':
            self.input_act_scale = None
        
        return output


class QMixtralSparseMoeBlock(nn.Module):
    """
    tweak forward dtype casting
    """
    def __init__(self, module: MixtralSparseMoeBlock, dtype=torch.float16):
        super().__init__()
        self.hidden_dim = module.hidden_dim
        self.ffn_dim = module.ffn_dim
        self.num_experts = module.num_experts
        self.top_k = module.top_k
        self.dtype = dtype
        
        self.gate = module.gate
        self.experts = module.experts
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(self.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=self.dtype, device=hidden_states.device
        )
        
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(self.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QLlamaSdpaAttention(nn.Module):

    def __init__(self, module: LlamaSdpaAttention):
        super().__init__()
        self.config = module.config
        self.layer_idx = module.layer_idx
        
        self.attention_dropout = module.attention_dropout
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.max_position_embeddings = module.max_position_embeddings
        self.rope_theta = module.rope_theta
        self.is_causal = module.is_causal
        
        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj
        
        self.rotary_emb = module.rotary_emb
    
    # Adapted from LlamaSdpaAttention.forward 
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, 'past_key_value', past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {'sin': sin, cos: cos, 'cache_position': cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        ### BMM quantization
        qmax = self.q_proj.qmax
        query_scale = query_states.abs().amax() / qmax
        query_states = query_states.div(query_scale).round().mul(query_scale)
        
        key_scale = key_states.abs().amax() / qmax
        key_states = key_states.div(key_scale).round().mul(key_scale)

        value_scale = value_states.abs().amax() / qmax
        value_states = value_states.div(value_scale).round().mul(value_scale)
        
        # 1)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # 2)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        attn_weights = None
        
        return attn_output, attn_weights, past_key_value


def quantize_model(model,
                   granul: str,
                   act_granul: str = 'per-tensor',
                   except_layer: list = [],
                   input_scale_dict: dict = None,
                   quantize_bmm: bool = False,
                   dtype=torch.float16):
    """
    granul: weight quant scheme, (per-channel or per-tensor)
    """
    qmax = torch.iinfo(torch.int8).max
    decoder_prefix = model.base_model_prefix
    except_layer = expand_module_names(model, except_layer)  # NOTE: inputs are not expanded

    for name, p in model.named_parameters():
        p.requires_grad = False
    
    # quantize linear modules
    for name, m in model.named_modules():
        modules_names = name.split('.')
        
        if isinstance(m, nn.Linear) and name.startswith(decoder_prefix):

            parent_module_name = name[:name.rindex('.')]
            parent_module = model.get_submodule(parent_module_name)

            child_module = QLinear(m, granul, act_granul=act_granul, dtype=dtype)
            if name in except_layer:
                child_module.except_layer = True
            setattr(parent_module, modules_names[-1], child_module)
    
    if isinstance(model, MixtralForCausalLM):
        # quantize mixtral moe block
        for name, module in model.named_modules():
            if isinstance(module, MixtralSparseMoeBlock):
                parent_name = name[:name.rindex('.')]
                child_name = name.split('.')[-1]
                
                parent_module = model.get_submodule(parent_name)
                q_module = QMixtralSparseMoeBlock(module)
                setattr(parent_module, child_name, q_module)
    
    if quantize_bmm:
        for name, module in model.named_modules():
            
            if isinstance(module, LlamaSdpaAttention):
                parent_name = parent_name = name[:name.rindex('.')]
                child_name = name.split('.')[-1]

                parent_module = model.get_submodule(parent_name)
                q_module = QLlamaSdpaAttention(module)
                setattr(parent_module, child_name, q_module)

    if act_granul == 'static':
        assert input_scale_dict is not None

        for name, input_act_scale in input_scale_dict.items():
            
            siblings = get_sibling_names(model, name)
            for linear_name in siblings:
                linear = model.get_submodule(linear_name)
                linear.input_act_scale = input_act_scale

    # quantize LayerNorm outputs
    def layer_norm_hook(m, x, output_act, next_modules, output_act_scale):
        if act_granul == 'static':
            output_act = output_act.div(output_act_scale).round().type(torch.int8)
        elif act_granul == 'per-tensor':
            output_act_scale = output_act.abs().amax() / qmax
            output_act = output_act.div(output_act_scale).round().type(torch.int8)
        elif act_granul == 'per-token':
            output_act_scale = output_act.abs().amax(dim=-1, keepdims=True) / qmax
            output_act = output_act.div(output_act_scale).round().type(torch.int8)
        else:
            raise f'Unknown quantization scheme for activation: {act_granul}'
        
        for module in next_modules:
            module.input_act_scale = output_act_scale
        
        return output_act

    layer_cls = type(get_layers(model)[0])
    pre_norm_name_pairs = get_pre_norm_name_pairs(model)

    for layer_name, m in model.named_modules():
        if isinstance(m, layer_cls):
            for norm_name, next_module_names in pre_norm_name_pairs:
                # next_modules: list
                norm = getattr(m, norm_name)
                next_modules = [m.get_submodule(name) for name in next_module_names]
                if any([module.except_layer for module in next_modules]):
                    continue
                
                if act_granul == 'static':
                    for child_name in next_module_names:
                        scale_key = layer_name + '.' + child_name
                        if scale_key in input_scale_dict:
                            output_act_scale = input_scale_dict[scale_key]
                else:
                    output_act_scale = None

                hook_fn = partial(layer_norm_hook,
                                  next_modules=next_modules,
                                  output_act_scale=output_act_scale)
                norm.register_forward_hook(hook_fn)

    return model


def load_quantized_model(pretrained,
                         granul: str = 'per-channel',
                         act_granul: str = 'per-tensor',
                         except_layer: list = [],
                         dtype=torch.float16,
                         device='cuda',
                         sq_act_dict=None,  # for smoothquant
                         sq_alpha=0.8,  # for smoothquant
                         osp_scale_dict=None,  # for osp
                         quantize_bmm=False,
                         input_scale_dict=None,  # for static quantization, not expanded
                         use_safetensors=True):
    """
    Load model with quantization
    granul: weight quant scheme, (per-channel or per-tensor)
    """

    config = AutoConfig.from_pretrained(pretrained)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    
    model = quantize_model(model,
                           granul,
                           act_granul=act_granul,
                           except_layer=except_layer,
                           input_scale_dict=input_scale_dict,
                           quantize_bmm=quantize_bmm,
                           dtype=dtype)  # 먼저 QLinear로 바꾼 후 weight 불러오기 (이러면 gpu가 줄어듬)
    
    if use_safetensors:
        cached_tensor = CachedTensor(pretrained)
    else:
        cached_tensor = CPUTensor(pretrained)
    
    
    sub_info = split_state_dict(model)

    for key, param_names in sub_info.items():
        # key: 'model.layers.0'

        # none인 경우에만 dtype 따라서 weight 가져오기
        if key == 'none':
            sub_dict = cached_tensor.infer_state_dict(param_names, device=device, dtype=dtype)
        else:
            sub_dict = cached_tensor.infer_state_dict(param_names, device=device, dtype=torch.float16)
        model.load_state_dict(sub_dict, strict=False, assign=True)

        for name, p in model.named_parameters():
            p.requires_grad = False
        
        if key == 'none':  # non layer class
            continue
        
        layer = model.get_submodule(key)

        ### smoothquant
        if sq_act_dict is not None:
            layer_module = model.get_submodule(key)
            attn_ln = layer_module.input_layernorm  # attention forward norm
            qkv = [
                layer_module.self_attn.q_proj,
                layer_module.self_attn.k_proj,
                layer_module.self_attn.v_proj,
            ]

            qkv_input_scales = sq_act_dict[key + '.self_attn.q_proj']

            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, sq_alpha)

            ffn_ln = layer_module.post_attention_layernorm  # feed forward norm
            fcs = [layer_module.mlp.gate_proj, layer_module.mlp.up_proj]
            fcs_input_scales = sq_act_dict[key + '.mlp.up_proj']

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, sq_alpha)
        
        ### osp preprocess
        if osp_scale_dict is not None:
            prefix = key + '.{}'

            if prefix.format('input_layernorm') in osp_scale_dict:
                migration_scale = osp_scale_dict[prefix.format('input_layernorm')]
                layer.self_attn.q_proj.weight.data *= migration_scale
                layer.self_attn.k_proj.weight.data *= migration_scale
                layer.self_attn.v_proj.weight.data *= migration_scale
                layer.input_layernorm.weight.data /= migration_scale
            
            if prefix.format('post_attention_layernorm') in osp_scale_dict:
                migration_scale = osp_scale_dict[prefix.format('post_attention_layernorm')]
                layer.mlp.gate_proj.weight.data *= migration_scale
                layer.mlp.up_proj.weight.data *= migration_scale
                layer.post_attention_layernorm.weight.data /= migration_scale

            if prefix.format('mlp.up_proj') in osp_scale_dict:
                migration_scale = osp_scale_dict[prefix.format('mlp.up_proj')]
                layer.mlp.down_proj.weight.data *= migration_scale
        
        # weight quantization
        for p_name in param_names:
            m_name = p_name[:p_name.rindex('.')]

            module = model.get_submodule(m_name)
            if isinstance(module, QLinear):
                module.weight_quant()
        
        ### osp postprocess (mitigation into the weight scale factor)
        if osp_scale_dict is not None:
            prefix = key + '.{}'

            if prefix.format('mlp.up_proj') in osp_scale_dict:
                migration_scale = osp_scale_dict[prefix.format('mlp.up_proj')]
                layer.mlp.up_proj.weight_scale.data = \
                    (layer.mlp.up_proj.weight_scale.data / migration_scale.unsqueeze(1)).reshape(-1, 1)

    if model.config.tie_word_embeddings:
        model.tie_weights()
    
    # del sub_dict
    model = model.to(device)
    # torch.cuda.empty_cache()
    
    return model
