import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self,
        text: list[str],
        predictions: list[str] | None = None,
        log_probs: Tensor | None = None,
        log_probs_length: Tensor | None = None,
        **kwargs,
    ):
        wers = []
        pred_texts: list[str]
        if predictions is None:
            assert (
                log_probs is not None and log_probs_length is not None
            ), "Provide predictions or log_probs"
            argmax = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            lengths = log_probs_length.detach().numpy()
            pred_texts = [
                self.text_encoder.ctc_decode(vec[:length])
                for vec, length in zip(argmax, lengths)
            ]
        else:
            pred_texts = predictions

        for pred_text, target_text in zip(pred_texts, text):
            target_text = self.text_encoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
