import re
from dataclasses import dataclass
from string import ascii_lowercase

import torch
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC, build_ctcdecoder


@dataclass
class BeamResult:
    text: str
    score: float | None = None


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        kenlm_model_path="saved/3-gram.arpa",
        vocab_path="saved/librispeech-vocab.txt",
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if kenlm_model_path is not None:
            with open(vocab_path) as f:
                unigrams = [line.strip() for line in f.readlines()]
            self.decoder_lm = build_ctcdecoder(
                labels=[""] + self.alphabet,
                kenlm_model_path=kenlm_model_path,
                unigrams=unigrams,
            )
        self.decoder_no_lm = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        res = []
        empty_i = self.char2ind[self.EMPTY_TOK]
        last = empty_i
        for ind in inds:
            idx = int(ind)
            if idx != empty_i and idx != last:
                res.append(self.ind2char[idx])
            last = idx
        return "".join(res)

    def ctc_beam_search_custom(self, log_probs: torch.Tensor, beam_size: int = 10):
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()

        time_steps, vocab_size = log_probs.shape
        blank_idx = self.char2ind[self.EMPTY_TOK]

        beam = {(): (0.0, float("-inf"))}

        for t in range(time_steps):
            new_beam = {}

            for prefix, (p_blank, p_non_blank) in beam.items():
                char_idx = blank_idx
                new_prefix = prefix
                log_prob = log_probs[t, char_idx]

                p_total = self._log_sum_exp(p_blank + log_prob, p_non_blank + log_prob)

                if new_prefix not in new_beam:
                    new_beam[new_prefix] = (p_total, float("-inf"))
                else:
                    old_p_blank, old_p_non_blank = new_beam[new_prefix]
                    new_beam[new_prefix] = (
                        self._log_sum_exp(old_p_blank, p_total),
                        old_p_non_blank,
                    )

                for char_idx in range(vocab_size):
                    if char_idx == blank_idx:
                        continue

                    log_prob = log_probs[t, char_idx]
                    new_prefix = prefix + (char_idx,)

                    if len(prefix) > 0 and prefix[-1] == char_idx:
                        p_total = p_blank + log_prob
                    else:
                        p_total = self._log_sum_exp(
                            p_blank + log_prob, p_non_blank + log_prob
                        )

                    if new_prefix not in new_beam:
                        new_beam[new_prefix] = (float("-inf"), p_total)
                    else:
                        old_p_blank, old_p_non_blank = new_beam[new_prefix]
                        new_beam[new_prefix] = (
                            old_p_blank,
                            self._log_sum_exp(old_p_non_blank, p_total),
                        )

            beam_items = []
            for prefix, (p_blank, p_non_blank) in new_beam.items():
                total_p = self._log_sum_exp(p_blank, p_non_blank)
                beam_items.append((total_p, prefix, p_blank, p_non_blank))

            beam_items.sort(reverse=True, key=lambda x: x[0])
            beam = {
                prefix: (p_blank, p_non_blank)
                for _, prefix, p_blank, p_non_blank in beam_items[:beam_size]
            }

        best_prefix = max(
            beam.items(), key=lambda x: self._log_sum_exp(x[1][0], x[1][1])
        )[0]

        result = "".join([self.ind2char[idx] for idx in best_prefix])
        return result.lower()

    @staticmethod
    def _log_sum_exp(a, b):
        """Numerically stable log(exp(a) + exp(b))"""
        import numpy as np

        if a == float("-inf") and b == float("-inf"):
            return float("-inf")
        if a > b:
            return a + np.log1p(np.exp(b - a))
        else:
            return b + np.log1p(np.exp(a - b))

    def ctc_beam_search(
        self,
        use_lm: bool,
        log_probs: torch.Tensor,
        probs: torch.Tensor,
        logits: torch.Tensor,
        beam_size: int,
    ):
        """
        Decode a single utterance using beam search, optionally with an LM.

        Note: Only the `logits` and `beam_size` arguments are used; other
        arguments are kept for backward compatibility with call sites.
        """
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits

        if use_lm and hasattr(self, "decoder_lm") and self.decoder_lm is not None:
            return self.decoder_lm.decode(logits_np, beam_size).lower()
        return self.decoder_no_lm.decode(logits_np, beam_size).lower()

    def decode_logits(
        self,
        logits: torch.Tensor,
        log_probs: torch.Tensor,
        length: int,
        beam_size: int = 1,
    ):
        logits = logits[:length]
        log_probs = log_probs[:length]
        argmax_inds = torch.argmax(log_probs, dim=-1)
        argmax_list = argmax_inds.detach().cpu().tolist()
        raw_prediction = self.decode(argmax_list)
        argmax_prediction = self.ctc_decode(argmax_list)

        beams: list[BeamResult] = []
        if beam_size > 0:
            beam_text = self.ctc_beam_search(
                hasattr(self, "decoder_lm"),
                log_probs,
                torch.exp(log_probs),
                logits,
                beam_size,
            )
            beams.append(BeamResult(text=beam_text))
        return argmax_prediction, raw_prediction, beams

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
