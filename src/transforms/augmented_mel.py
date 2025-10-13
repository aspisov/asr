import random

import torch
import torch.nn as nn
import torchaudio


class AugmentedMelSpectrogram(nn.Module):
    """Apply lightweight waveform augmentations, then compute Mel spectrogram.

    Augments: time shift, gain, additive noise (three simple waveform augs).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        # Mel params
        n_mels: int = 128,
        n_fft: int | None = None,
        win_length: int | None = None,
        hop_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        # Time shift
        shift_p: float = 0.5,
        max_shift_seconds: float = 0.1,
        # Gain
        gain_p: float = 0.5,
        min_gain_db: float = -6.0,
        max_gain_db: float = 6.0,
        # Noise
        noise_p: float = 0.5,
        snr_db_min: float = 10.0,
        snr_db_max: float = 30.0,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)

        # mel
        mel_kwargs: dict = {
            "sample_rate": self.sample_rate,
            "n_mels": int(n_mels),
            "f_min": float(f_min),
        }
        if f_max is not None:
            mel_kwargs["f_max"] = float(f_max)
        if n_fft is not None:
            mel_kwargs["n_fft"] = int(n_fft)
        if win_length is not None:
            mel_kwargs["win_length"] = int(win_length)
        if hop_length is not None:
            mel_kwargs["hop_length"] = int(hop_length)
        self.mel = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

        # augs
        self.shift_p = float(shift_p)
        self.max_shift_seconds = float(max_shift_seconds)

        self.gain_p = float(gain_p)
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)

        self.noise_p = float(noise_p)
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio expected shape: (1, T)
        y = audio

        # time shift
        if random.random() < self.shift_p and self.max_shift_seconds > 0:
            max_shift = int(self.max_shift_seconds * self.sample_rate)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    ch, length = y.shape[-2], y.shape[-1]
                    if shift > 0:
                        pad = torch.zeros(ch, shift, dtype=y.dtype, device=y.device)
                        y = torch.cat([pad, y], dim=-1)[..., :length]
                    else:
                        pad = torch.zeros(ch, -shift, dtype=y.dtype, device=y.device)
                        y = torch.cat([y, pad], dim=-1)[..., -length:]

        # gain
        if random.random() < self.gain_p:
            gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
            gain = 10.0 ** (gain_db / 20.0)
            y = (y * gain).clamp_(-1.0, 1.0)

        # noise
        if random.random() < self.noise_p:
            signal_power = y.pow(2).mean().item()
            if signal_power > 1e-12:
                snr_db = random.uniform(self.snr_db_min, self.snr_db_max)
                snr_linear = 10.0 ** (snr_db / 10.0)
                noise_power = max(signal_power / snr_linear, 1e-12)
                noise = torch.randn_like(y) * (noise_power ** 0.5)
                y = (y + noise).clamp_(-1.0, 1.0)

        # mel spectrogram (n_mels, time)
        spec = self.mel(y)
        return spec


