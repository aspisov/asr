import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, logits: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        cers = []
        logits_np = logits.detach().cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for logit, length, target_text in zip(logits_np, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(
                False, None, None, logit[:length], self.beam_size
            )
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchLMCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, logits: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        cers = []
        logits_np = logits.detach().cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for logit, length, target_text in zip(logits_np, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(
                True, None, None, logit[:length], self.beam_size
            )
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
