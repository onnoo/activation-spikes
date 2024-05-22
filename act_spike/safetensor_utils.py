from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from .parsing_utils import get_layers


def merge_name(name_string, incoming_name):
    return f'{name_string}.{incoming_name}'.lstrip('.')


def split_state_dict(model: nn.Module):
    state_dict = defaultdict(dict)
    layer_class = type(get_layers(model)[0])

    def _split_state_dict(module: nn.Module, name_string=''):
        children = list(module.named_children())
        
        if isinstance(module, layer_class):
            state_dict[name_string] = { merge_name(name_string, key): None for key in module.state_dict() }
            return
        
        if len(children) == 0:
            for name, param in module.named_parameters():
                state_dict['none'][merge_name(name_string, name)] = None
            return
        
        for name, module in module.named_children():
            _split_state_dict(module, merge_name(name_string, name))

    _split_state_dict(model)

    return state_dict


class CPUTensor:

    def __init__(self, pretrained: str):
        model = AutoModelForCausalLM.from_pretrained(pretrained,
                                                     torch_dtype=torch.float16,
                                                     device_map='cpu')
        self.state_dict = model.state_dict()

    def infer_state_dict(self, state_dict: dict, device='cuda', dtype=torch.float16):
        output_dict = {}
        
        for key in state_dict.keys():
            if key not in self.state_dict:
                continue
            output_dict[key] = self.state_dict[key].to(device)

        return output_dict


class CachedTensor:
    
    def __init__(self, pretrained: str):
        index_file = cached_file(pretrained, SAFE_WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False)

        if index_file is None:
            self.shard_files = [cached_file(pretrained, SAFE_WEIGHTS_NAME)]
            self.weight_map = {}
            with safe_open(self.shard_files[0], framework='pt', device='cpu') as f:
                for key in f.keys():
                    self.weight_map[key] = self.shard_files[0]
        else:  # is_sharded
            self.shard_files, self.meta = get_checkpoint_shard_files(pretrained, index_file)
            self.weight_map = {
                key: str(Path(index_file).parent.joinpath(fname)) for key, fname in self.meta['weight_map'].items()
            }
    
    def infer_state_dict(self, state_dict: dict, device='cuda', dtype=torch.float16):
        inverted_map = defaultdict(list)
        output_dict = {}

        for key in state_dict.keys():
            if key not in self.weight_map:
                # weight tying case
                continue
            shard_file = self.weight_map[key]
            inverted_map[shard_file].append(key)
        
        for shard_file, keys in inverted_map.items():
            with safe_open(shard_file, framework='pt', device=device) as f:
                for key in keys:
                    tensor = f.get_tensor(key).type(dtype)
                    output_dict[key] = tensor

        output_dict = { k: v for k, v in output_dict.items() if v is not None }

        return output_dict
