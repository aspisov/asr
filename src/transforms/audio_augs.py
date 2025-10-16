import random

import torch
import torch.nn as nn
import torchaudio


class RandomGain(nn.Module):
    """Randomly amplify/attenuate waveform in dB range.

    Args:
        min_gain_db: minimum gain in dB (negative attenuates)
        max_gain_db: maximum gain in dB
        p: probability to apply
    """

    def __init__(
        self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.5
    ) -> None:
        super().__init__()
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)
        self.p = float(p)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return audio
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain = 10.0 ** (gain_db / 20.0)
        return (audio * gain).clamp_(-1.0, 1.0)


class AdditiveGaussianNoise(nn.Module):
    """Add white Gaussian noise at a random SNR.

    Args:
        snr_db_min: lower bound SNR in dB
        snr_db_max: upper bound SNR in dB
        p: probability to apply
    """

    def __init__(
        self, snr_db_min: float = 10.0, snr_db_max: float = 30.0, p: float = 0.5
    ) -> None:
        super().__init__()
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.p = float(p)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return audio
        # audio: (1, T) expected
        signal_power = audio.pow(2).mean().item()
        if signal_power <= 1e-12:
            return audio
        snr_db = random.uniform(self.snr_db_min, self.snr_db_max)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(audio) * (noise_power**0.5)
        return (audio + noise).clamp_(-1.0, 1.0)


class RandomTimeShift(nn.Module):
    """Shift waveform left/right by up to max_shift_seconds, padding with zeros.

    Args:
        max_shift_seconds: maximum absolute shift in seconds
        sample_rate: sample rate to convert seconds to samples
        p: probability to apply
    """

    def __init__(
        self, max_shift_seconds: float = 0.1, sample_rate: int = 16000, p: float = 0.5
    ) -> None:
        super().__init__()
        self.max_shift_seconds = float(max_shift_seconds)
        self.sample_rate = int(sample_rate)
        self.p = float(p)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return audio
        max_shift = int(self.max_shift_seconds * self.sample_rate)
        if max_shift <= 0:
            return audio
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return audio
        ch, length = audio.shape[-2], audio.shape[-1]
        if shift > 0:
            pad = torch.zeros(ch, shift, dtype=audio.dtype, device=audio.device)
            audio = torch.cat([pad, audio], dim=-1)[..., :length]
        else:
            pad = torch.zeros(ch, -shift, dtype=audio.dtype, device=audio.device)
            audio = torch.cat([audio, pad], dim=-1)[..., -length:]
        return audio


class SimpleSpecMask(nn.Module):
    """Apply one freq mask and one time mask to a spectrogram."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 40,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return spec

        squeeze = False
        if spec.dim() == 3 and spec.size(0) == 1:
            spec = spec.squeeze(0)
            squeeze = True

        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)

        if squeeze:
            spec = spec.unsqueeze(0)
        return spec
