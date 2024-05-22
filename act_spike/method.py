
import torch

from .calibration import get_act_dict
from .parsing_utils import get_prefix_ids


def generate_prefix(tokenizer, model, calib_dataset, act_dict, router_logits=None, k=3):

    # TODO: support Mixtral

    prefix_id = get_prefix_ids(tokenizer)[0]

    # get top ratio
    top_ratio = -1
    top_key = None

    for key, x in act_dict.items():
        med_val = torch.stack([ t.median() for t in x ])
        max_val = torch.stack([ t.max() for t in x ])

        ratio = (max_val / med_val).mean().item()

        if ratio > top_ratio:
            top_ratio = ratio
            top_key = key
    
    if router_logits is not None:
        router_key = top_key[:top_key.rindex('.experts')]
        expert_idx = int(top_key.split('.')[-2])
        selected_experts = router_logits[router_key].topk(2, dim=-1).indices

    # get candidates
    acts = act_dict[top_key]
    # values, indices = acts[:, :].topk(3)
    num_samples = len(acts)

    token_scales = {}

    for sample_idx in range(num_samples):
        values, indices = acts[sample_idx].topk(3)

        if router_logits is not None:
            input_pos = torch.where(selected_experts[sample_idx].T == expert_idx)[1]
            seq_pos = input_pos[indices]
        else:
            seq_pos = indices
        
        token_ids = calib_dataset[sample_idx][seq_pos]
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item()

            if token_id == prefix_id:  # except BOS token
                continue

            value = values[i].item()
            if token_id in token_scales:
                token_scales[token_id] = max(token_scales[token_id], value)
            else:
                token_scales[token_id] = value
    sorted_scales = sorted(token_scales.items(), key=lambda x: x[1], reverse=True)

    results = []
    for i in range(k):
        token_id, value = sorted_scales[i]
        # indices = torch.where(calib_dataset == token_id)
        token_ids, freq = torch.unique(calib_dataset[:, 1:], return_counts=True) # pop BOS token
        ctx_token_ids = token_ids[freq.topk(200)[1]]  # 계산 범위 좁히기 + 발견 확률? 혹은 전체 데이터셋에서 top 매겨도 됨
        # 자주 등장: 컨텍스트에 덜 영향을 미친다? (stop words?)
        
        input_ids = ctx_token_ids.unsqueeze(1).repeat(1, 5)
        input_ids[:, 0] = prefix_id
        input_ids[:, 2] = token_id
        input_ids[:, 4] = token_id
        
        stop_layer = -1
        for name in top_key.split('.'):
            if name.isnumeric():
                stop_layer = int(name)
                break
        
        test_act_dict, hidden_states, _ = get_act_dict(model, input_ids, stop_layer=stop_layer)
        if router_logits is not None:
            top_layer_key = '.'.join(top_key.split('.')[:3])
            test_acts = hidden_states[top_layer_key].abs()
        else:
            test_acts = test_act_dict[top_key]
        
        idx = (test_acts[:, 2] / test_acts[:, 4]).argmax()
        ctx_token_id = ctx_token_ids[idx].item()
        
        prefix_ids = [prefix_id, ctx_token_id, token_id]
        prefix_tokens = [ repr(tokenizer.decode(tid)) for tid in prefix_ids]
        
        data = {
            'prefix_tokens': prefix_tokens,
            'prefix_ids': prefix_ids,
            'max_act': value,
            'first_val': test_acts[idx, 2].item(),
            'second_val': test_acts[idx, 4].item(),
            'rate': test_acts[idx, 2].item() / test_acts[idx, 4].item()
        }
        results.append(data)
    
    final_prefix_ids = max(results, key=lambda x: x['first_val'] / x['second_val'])['prefix_ids']
    
    meta = {
        'top_key': top_key,
        'top_ratio': top_ratio,
        'prefix_ids': final_prefix_ids
    }

    return {'meta': meta, 'results': results}
