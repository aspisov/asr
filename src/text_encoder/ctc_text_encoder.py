import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from string import ascii_lowercase

import torch

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


@dataclass(slots=True)
class BeamSearchCandidate:
    """Single beam search hypothesis."""

    text: str
    score: float
    tokens: tuple[int, ...]


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        beam_size: int = 5,
        beam_cutoff: float | None = None,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
            beam_size (int): beam size to use for CTC beam search decoding.
            beam_cutoff (float | None): optional per-token cutoff in log-space.
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_size = beam_size
        self.beam_cutoff = beam_cutoff

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
        blank_ind = self.char2ind[self.EMPTY_TOK]
        decoded = []
        prev_ind = None

        for ind in inds:
            ind = int(ind)
            if ind == prev_ind:
                continue
            if ind != blank_ind:
                decoded.append(self.ind2char[ind])
            prev_ind = ind

        return "".join(decoded).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    @staticmethod
    def _log_sum_exp(*values: float) -> float:
        result = float("-inf")
        for value in values:
            if result == float("-inf"):
                result = value
            elif value == float("-inf"):
                continue
            else:
                max_val = max(result, value)
                result = max_val + math.log1p(math.exp(min(result, value) - max_val))
        return result

    def decode_logits(
        self,
        log_probs: torch.Tensor,
        length: int,
        beam_size: int | None = None,
    ) -> tuple[str, str, list[BeamSearchCandidate]]:
        slice_logits = log_probs[:length]
        argmax_tokens = torch.argmax(slice_logits, dim=-1).tolist()
        raw_argmax = self.decode(argmax_tokens)
        argmax_decoded = self.ctc_decode(argmax_tokens)
        beam_candidates = self.ctc_beam_search(slice_logits, beam_size=beam_size)
        return argmax_decoded, raw_argmax, beam_candidates

    def ctc_beam_search(
        self,
        log_probs: torch.Tensor,
        beam_size: int | None = None,
    ) -> list[BeamSearchCandidate]:
        if beam_size is None:
            beam_size = self.beam_size

        if log_probs.dim() != 2:
            raise ValueError("log_probs must have shape [time, vocab]")

        blank = self.char2ind[self.EMPTY_TOK]
        time_steps, vocab_size = log_probs.size()
        log_probs_np = log_probs.detach().cpu().tolist()

        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}

        for t in range(time_steps):
            frame = log_probs_np[t]
            next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

            cutoff = self.beam_cutoff
            if cutoff is not None:
                frame_iter: Iterable[tuple[int, float]] = (
                    (token, log_prob)
                    for token, log_prob in enumerate(frame)
                    if log_prob >= cutoff
                )
            else:
                frame_iter = enumerate(frame)

            top_tokens = sorted(frame_iter, key=lambda x: x[1], reverse=True)

            for prefix, (log_p_blank, log_p_non_blank) in beams.items():
                for token, token_log_prob in top_tokens:
                    if token == blank:
                        new_log_p_blank = self._log_sum_exp(
                            next_beams.get(prefix, (float("-inf"), float("-inf")))[0],
                            log_p_blank + token_log_prob,
                            log_p_non_blank + token_log_prob,
                        )
                        new_log_p_non_blank = next_beams.get(
                            prefix, (float("-inf"), float("-inf"))
                        )[1]
                        next_beams[prefix] = (new_log_p_blank, new_log_p_non_blank)
                        continue

                    end_char = prefix[-1] if prefix else None
                    new_prefix = prefix + (token,)

                    prev_beam = next_beams.get(
                        new_prefix, (float("-inf"), float("-inf"))
                    )

                    if token == end_char:
                        new_log_p_non_blank = self._log_sum_exp(
                            prev_beam[1],
                            log_p_blank + token_log_prob,
                        )
                    else:
                        new_log_p_non_blank = self._log_sum_exp(
                            prev_beam[1],
                            self._log_sum_exp(log_p_blank, log_p_non_blank)
                            + token_log_prob,
                        )

                    next_beams[new_prefix] = (prev_beam[0], new_log_p_non_blank)

                    if token == end_char:
                        continue

                    prev_same_prefix = next_beams.get(
                        prefix, (float("-inf"), float("-inf"))
                    )
                    new_same_non_blank = self._log_sum_exp(
                        prev_same_prefix[1],
                        log_p_non_blank + token_log_prob,
                    )
                    next_beams[prefix] = (prev_same_prefix[0], new_same_non_blank)

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: self._log_sum_exp(*item[1]),
                    reverse=True,
                )[: beam_size * 2]
            )

        candidates = []
        for tokens, (log_p_blank, log_p_non_blank) in beams.items():
            total_log_prob = self._log_sum_exp(log_p_blank, log_p_non_blank)
            text = self.decode(tokens)
            text = self.normalize_text(text)
            if text:
                candidates.append(
                    BeamSearchCandidate(
                        text=text,
                        score=total_log_prob,
                        tokens=tokens,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:beam_size]
