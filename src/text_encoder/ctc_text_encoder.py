import re
from dataclasses import dataclass
from string import ascii_lowercase

from pyctcdecode import build_ctcdecoder

import torch


@dataclass(slots=True)
class BeamSearchCandidate:
    text: str
    score: float
    tokens: tuple[int, ...]


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        beam_size: int = 5,
        kenlm_model_path: str | None = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        **kwargs,
    ):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.default_beam_size = beam_size
        labels = self.vocab
        self.decoder_no_lm = build_ctcdecoder(labels=labels)
        self.decoder_lm = None
        if kenlm_model_path:
            self.decoder_lm = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=kenlm_model_path,
                alpha=alpha,
                beta=beta,
            )
        self._uses_lm = self.decoder_lm is not None

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = [char for char in text if char not in self.char2ind]
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(sorted(set(unknown_chars)))}'"
            )

    def decode(self, inds) -> str:
        return "".join(self.ind2char[int(ind)] for ind in inds).strip()

    def ctc_decode(self, inds) -> str:
        blank_ind = self.char2ind[self.EMPTY_TOK]
        decoded = []
        prev_ind = blank_ind
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

    def decode_logits(
        self,
        log_probs: torch.Tensor,
        length: int,
        beam_size: int | None = None,
        use_lm: bool | None = None,
    ) -> tuple[str, str, list[BeamSearchCandidate]]:
        slice_logits = log_probs[:length]
        argmax_tokens = torch.argmax(slice_logits, dim=-1).tolist()
        raw_argmax = self.decode(argmax_tokens)
        argmax_decoded = self.ctc_decode(argmax_tokens)
        beam_size = beam_size if beam_size is not None else self.default_beam_size
        probs = slice_logits.exp().cpu().numpy()
        decoder = self._select_decoder(use_lm=use_lm)
        beam_results = decoder.decode_beams(probs, beam_width=beam_size)
        candidates = []
        for result in beam_results:
            text = self.normalize_text(result["text"])
            score = result["combined_score"]
            tokens = tuple(int(tok) for tok in result.get("tokens", ()))
            candidates.append(BeamSearchCandidate(text=text, score=score, tokens=tokens))
        return argmax_decoded, raw_argmax, candidates

    def _select_decoder(self, use_lm: bool | None):
        if use_lm is None:
            use_lm = self._uses_lm
        if use_lm and self.decoder_lm is not None:
            return self.decoder_lm
        return self.decoder_no_lm