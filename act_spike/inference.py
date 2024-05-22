from collections import OrderedDict

import torch

from tqdm.auto import tqdm

from .safetensor_utils import CachedTensor, split_state_dict, CPUTensor
from .parsing_utils import get_layers, get_embeddings, get_norm, get_lm_head


def _assign_var(model, var_path, value):
    parent_name = var_path[:var_path.rfind('.')]
    var_name = var_path[var_path.rfind('.') + 1:]
    parent = model.get_submodule(parent_name)
    
    setattr(parent, var_name, value)


@torch.no_grad()
def inference_per_layers_disk(model,
                              input_ids: torch.LongTensor,
                              device: str ='cuda',
                              verbose=True,
                              use_safetensors=True,
                              stop_layer=-1,
                              use_cache=False):

    # config = AutoConfig.from_pretrained(pretrained)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    pretrained = model.name_or_path
    if use_safetensors:
        cached_tensor = CachedTensor(pretrained)
    else:
        cached_tensor = CPUTensor(pretrained)
    
    sub_info = split_state_dict(model)
    sub_info = { k.split('.')[-1]: v for k, v in sub_info.items() }

    sub_dict = cached_tensor.infer_state_dict(sub_info['none'], device='cpu', dtype=model.dtype)
    model.load_state_dict(sub_dict, strict=False, assign=True)
    if model.config.tie_word_embeddings:
        model.tie_weights()

    ########
    
    n_samples, seqlen = input_ids.size()
    conf_use_cache = model.config.use_cache
    model.config.use_cache = use_cache

    layers = get_layers(model)
    embeddings = get_embeddings(model)

    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(device)
    
    dtype = next(model.parameters()).dtype
    inps = torch.zeros(
        (n_samples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache.update(kwargs)

            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in tqdm(range(n_samples), desc='Warmup', disable=not verbose):
        batch = input_ids[i, :].unsqueeze(0).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']

    for i in tqdm(range(model.config.num_hidden_layers), desc='Layer', disable=not verbose):
        layer = layers[i]
        sub_dict = cached_tensor.infer_state_dict(sub_info[str(i)], device=device, dtype=model.dtype)

        model.load_state_dict(sub_dict, strict=False, assign=True)

        for name, buffer in layer.named_buffers():
            _assign_var(layer, name, buffer.cuda())
        
        for name, m in layer.named_parameters():
            m.requires_grad = False

        for j in range(n_samples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                **cache
            )[0]
        
        for name, param in sub_dict.items():
            sub_dict[name] = param.data.to('meta')
        model.load_state_dict(sub_dict, strict=False, assign=True)

        inps, outs = outs, inps
        torch.cuda.empty_cache()

        if i == stop_layer:
            break

    ########

    if stop_layer == -1:
        norm = get_norm(model)
        if norm is not None:
            norm = norm.to(device)
        lm_head = get_lm_head(model)
        lm_head = lm_head.to(device)

        input_ids = input_ids.to(device)
        nlls = []

        for i in range(n_samples):
            hidden_states = inps[i].unsqueeze(0)
            if norm is not None:
                hidden_states = norm(hidden_states)
            lm_logits = lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[i, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen)).item()
        print('perplexity:', ppl)
    
    del inps
    del outs
    del cache

    state_dict = OrderedDict()
    for name, param in model.named_parameters():
        state_dict[name] = param.data.to('meta')
    model.load_state_dict(state_dict, strict=False, assign=True)

    torch.cuda.empty_cache()
    model.config.use_cache = conf_use_cache

    if stop_layer == -1:
        return { 'perplexity': ppl, 'nlls': nlls }
    else:
        return { 'perplexity': None, 'nlls': None }
