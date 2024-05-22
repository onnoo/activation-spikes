from typing import List, Tuple

import torch
import transformers
from tqdm import tqdm
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


eval_logger = utils.eval_logger


def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    # first_seq_len = min(max_seq_len, len(token_list))
    # yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    # predicted += first_seq_len

    # === without prefix token ===
    first_seq_len = min(max_seq_len, len(token_list))
    # eval_logger.debug(f'first_seq_len: {first_seq_len}')
    yield (token_list[: first_seq_len - 1], token_list[1:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len

    assert predicted == len(token_list)  # 사실, predicted에서 토큰 하나는 없다


class CacheHFLM(HFLM):

    def set_prefix_ids(self, prefix_ids):
        self.prefix_ids = prefix_ids
    
    def set_past_key_values(self, past_key_values):
        self.past_key_values = past_key_values

    @property
    def eot_token_id(self):
        # eval_logger.debug('Call: eos_token_id()')

        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id
    
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """

        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                add_special_tokens = False
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                add_special_tokens = True

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_call(self, inps, attn_mask=None, labels=None):
        # eval_logger.debug(f'Call: _model_call({inps}, {attn_mask}, {labels})')

        if hasattr(self, 'past_key_values'):
            with torch.no_grad():
                logits = self.model(inps, past_key_values=self.past_key_values, use_cache=True).logits
                return logits.to('cuda')
            
        elif hasattr(self, 'prefix_ids'):
            prefix_size = self.prefix_ids.size(1)
            inps = torch.cat([self.prefix_ids, inps], dim=1)

            with torch.no_grad():
                logits = self.model(inps, use_cache=True).logits[:, prefix_size:]
                return logits.to('cuda')

        with torch.no_grad():
            logits = self.model(inps, use_cache=True).logits
            return logits.to('cuda')

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        loglikelihoods = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):

            # eval_logger.debug(f'Original sequence {self.tok_encode(string)}')
            # eval_logger.debug(f'len_token: {len(self.tok_encode(string))}')

            token_list = self.tok_encode(string)
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            predicted = len(token_list) - 1  # we do not predict the first token in sequence
            
            loglikelihoods.append((string_nll, predicted))

        return loglikelihoods