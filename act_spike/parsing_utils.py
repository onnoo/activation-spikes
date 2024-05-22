from transformers import AutoConfig
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    MixtralForCausalLM,
    StableLmForCausalLM,
    GemmaForCausalLM,
    OPTForCausalLM,
    MptForCausalLM,
    GPTNeoXForCausalLM,
    FalconForCausalLM,
    PhiForCausalLM
)

__all__ = [
    "get_linear_names",
    "get_prefix_ids",
    "get_decoder",
    "get_layers",
    "get_embeddings",
    "get_norm",
    "get_lm_head",
    "expand_module_names",
    "get_sibling_names",
    "get_pre_norm_name_pairs"
]


def get_linear_names(model):
    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM)):
        return ['q_proj', 'o_proj', 'up_proj', 'down_proj']
    elif isinstance(model, MixtralForCausalLM):
        target_modules = ['q_proj', 'o_proj', 'gate']
        for i in range(model.config.num_local_experts):
            target_modules.extend([f'experts.{i}.w2', f'experts.{i}.w3'])
        # w1: gate, w2: down, w3: up
        return target_modules
    elif isinstance(model, OPTForCausalLM):
        return ['q_proj', 'out_proj', 'fc1', 'fc2']
    elif isinstance(model, MptForCausalLM):
        return ['Wqkv', 'out_proj', 'up_proj', 'down_proj']
    elif isinstance(model, (GPTNeoXForCausalLM, FalconForCausalLM)):
        return ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h']
    elif isinstance(model, PhiForCausalLM):
        return ['q_proj', 'dense', 'fc1', 'fc2']


def get_pre_norm_name_pairs(model):
    if isinstance(model, (LlamaForCausalLM,
                          GemmaForCausalLM,
                          MistralForCausalLM)):
        return [('input_layernorm', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
                ('post_attention_layernorm', ['mlp.gate_proj', 'mlp.up_proj'])]
    elif isinstance(model, MixtralForCausalLM):
        pre_norm_name_pairs = [
            ('input_layernorm', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
        ]
        pre_norm_mlp_downstream = ['block_sparse_moe.gate']
        for i in range(model.config.num_local_experts):
            pre_norm_mlp_downstream.extend(
                [f'block_sparse_moe.experts.{i}.w1', f'block_sparse_moe.experts.{i}.w3'])
        pre_norm_name_pairs.append(('post_attention_layernorm', pre_norm_mlp_downstream))
        return pre_norm_name_pairs


def get_prefix_ids(tokenizer):
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    if bos_token_id:
        return [bos_token_id]
    else:
        return [eos_token_id]


def get_decoder(model):
    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          MixtralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM,
                          PhiForCausalLM)):
        return model.model
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder
    elif isinstance(model, (MptForCausalLM, FalconForCausalLM)):
        return model.transformer
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.gpt_neox
    

def get_layers(model):
    decoder = get_decoder(model)

    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          MixtralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM,
                          GPTNeoXForCausalLM,
                          PhiForCausalLM)):
        return decoder.layers
    elif isinstance(model, OPTForCausalLM):
        return decoder.layers
    elif isinstance(model, MptForCausalLM):
        return decoder.blocks
    elif isinstance(model, FalconForCausalLM):
        return decoder.h


def get_embeddings(model):
    decoder = get_decoder(model)

    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          MixtralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM,
                          PhiForCausalLM)):
        return [decoder.embed_tokens]
    elif isinstance(model, OPTForCausalLM):
        return [decoder.embed_tokens, decoder.embed_positions]
    elif isinstance(model, MptForCausalLM):
        return [decoder.wte]
    elif isinstance(model, GPTNeoXForCausalLM):
        return [decoder.embed_in]
    elif isinstance(model, FalconForCausalLM):
        return [decoder.word_embeddings]


def get_norm(model):
    decoder = get_decoder(model)

    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          MixtralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM)):
        return decoder.norm
    elif isinstance(model, (OPTForCausalLM,
                            GPTNeoXForCausalLM)):
        return decoder.final_layer_norm
    elif isinstance(model, MptForCausalLM):
        return decoder.norm_f
    elif isinstance(model, FalconForCausalLM):
        return decoder.ln_f
    elif isinstance(model, PhiForCausalLM):
        return decoder.final_layernorm


def get_lm_head(model):
    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          MixtralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM,
                          OPTForCausalLM,
                          MptForCausalLM,
                          FalconForCausalLM,
                          PhiForCausalLM)):
        return model.lm_head
    if isinstance(model, GPTNeoXForCausalLM):
        return model.embed_out


def expand_module_names(model, module_names):
    """
    For except layers. module_names contain list of full name
    """
    new_module_names = []
    for module_name in module_names:
        sibling_modules = get_sibling_names(model, module_name)
        new_module_names.extend(sibling_modules)
        
    return new_module_names


def get_sibling_names(model, name):
    module_name = name.split('.')[-1]

    if isinstance(model, (LlamaForCausalLM,
                          MistralForCausalLM,
                          StableLmForCausalLM,
                          GemmaForCausalLM)):
        sibling_dict = {
            'q_proj': ['q_proj', 'k_proj', 'v_proj'],
            'up_proj': ['up_proj', 'gate_proj']
        }
    elif isinstance(model, MixtralForCausalLM):
        sibling_dict = {
            'q_proj': ['q_proj', 'k_proj', 'v_proj'],
            'w3': ['w1', 'w3']
        }
    elif isinstance(model, OPTForCausalLM):
        sibling_dict = {
            'q_proj': ['q_proj', 'k_proj', 'v_proj']
        }
    elif isinstance(model, (MptForCausalLM,
                            GPTNeoXForCausalLM,
                            FalconForCausalLM)):
        sibling_dict = {}
    elif isinstance(model, PhiForCausalLM):
        sibling_dict = {
            'q_proj': ['q_proj', 'k_proj', 'v_proj'],
        }
    
    sibling_names = sibling_dict.get(module_name, [module_name])
    
    full_sibling_names = []
    for sibling_name in sibling_names:
        full_name = '.'.join(name.split('.')[:-1] + [sibling_name])
        full_sibling_names.append(full_name)
    
    return full_sibling_names
