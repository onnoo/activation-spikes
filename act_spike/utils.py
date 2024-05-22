import torch
from .parsing_utils import get_layers


def load_past_key_values(model, path):
    kv_cache = torch.load(path)

    layers = get_layers(model)

    cuda_kv_cache = []

    for i in range(len(layers)):
        device = next(layers[i].parameters()).device
        key, value = kv_cache[i]
        cuda_kv_cache.append((key.to(device), value.to(device)))
    
    return tuple(cuda_kv_cache)