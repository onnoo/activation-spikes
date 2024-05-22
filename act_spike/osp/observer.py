import torch
import torch.nn as nn
import logging
from .util_quant import fake_quantize_per_tensor_affine, fake_quantize_per_channel_affine
logger = logging.getLogger('OS+')
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _transform_to_ch_axis(x, ch_axis):
    if ch_axis == -1:
        return x
    else:
        x_dim = x.size()
        new_axis_list = [i for i in range(len(x_dim))]
        new_axis_list[ch_axis] = 0
        new_axis_list[0] = ch_axis
        x_channel = x.permute(new_axis_list)
        y = torch.flatten(x_channel, start_dim=1)
        return y


class ObserverBase(nn.Module):

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(ObserverBase, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float).eps]))
        if self.symmetric:
            self.quant_min = -2 ** (self.bit - 1) + 1
            self.quant_max = 2 ** (self.bit - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** self.bit - 1
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def set_name(self, name):
        self.name = name

    def set_batch(self, batch):
        self.batch = batch

    def set_percentile(self, percentile):
        self.percentile = percentile

    def quantile_range(self, x, percentile):
        upper = torch.quantile(x.to(torch.float32).abs(), percentile)
        return -upper, upper

    def cac_thres(self, token_min, token_max):
        _, upper = self.quantile_range(token_max, self.percentile)
        lower, _ = self.quantile_range(token_min, self.percentile)
        indice_upper = torch.nonzero(token_max <= upper, as_tuple=True)[0]
        indice_lower = torch.nonzero(token_min >= lower, as_tuple=True)[0]
        return indice_lower, indice_upper

    def prune_token(self, value):   # try batch first
        if 'attention_probs' in self.name:
            return value
        token_max, _ = value.max(1)
        token_min, _ = value.min(1)
        indice_lower, indice_upper = self.cac_thres(token_min, token_max)
        upper = token_max[indice_upper].max()
        lower = token_min[indice_lower].min()
        value = torch.clip(value, max=upper, min=lower)
        return value

    def remove_padding(self, x, observation_mask, seq_pos):
        # assert the first dim is batch
        pos = list(range(len(x.shape)))
        shape = x.shape
        pos.remove(seq_pos)
        if len(pos) == 3:
            x = x.permute(pos[0], seq_pos, pos[1], pos[2]).reshape(shape[pos[0]], shape[seq_pos], -1)
        if len(pos) == 2:
            x = x.permute(pos[0], seq_pos, pos[1])
        return x[observation_mask == 1]

    def reshape_batch_embedding(self, x, seq_pos):
        # assert the first dim is batch
        pos = list(range(len(x.shape)))
        shape = x.shape
        pos.remove(seq_pos)
        if len(pos) == 3:
            x = x.permute(pos[0], seq_pos, pos[1], pos[2]).reshape(shape[pos[0]], shape[seq_pos], -1)
        if len(pos) == 2:
            x = x.permute(pos[0], seq_pos, pos[1])
        return x.reshape(shape[pos[0]] * shape[seq_pos], -1)

    @torch.jit.export
    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=min_val_neg.dtype, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int, device=device)
        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, bit=8, symmetric=False, ch_axis=-1):
        super(MinMaxObserver, self).__init__(bit=bit, symmetric=symmetric, ch_axis=ch_axis)

    def forward(self, x_orig, observation_mask=None, seq_pos=-1):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach()
        if observation_mask is not None:
            assert self.ch_axis == -1
            x = self.remove_padding(x, observation_mask, seq_pos)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            y = _transform_to_ch_axis(x, self.ch_axis)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        self.min_val = min_val_cur
        self.max_val = max_val_cur
        return min_val_cur, max_val_cur