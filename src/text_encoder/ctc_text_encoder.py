import re
from string import ascii_lowercase

import torch
from pyctcdecode.alphabet import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC, build_ctcdecoder


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
        for i in inds:
            if i != last and i != empty_i:
                res.append(self.ind2char[i])
                continue
            last = i
        return "".join(res)

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
        # Ensure numpy array [time, vocab]
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits

        if use_lm and hasattr(self, "decoder_lm") and self.decoder_lm is not None:
            return self.decoder_lm.decode(logits_np, beam_size).lower()
        # No-LM decoder
        return self.decoder_no_lm.decode(logits_np, beam_size).lower()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
