import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def forward(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        augmented = self._aug(x)
        return augmented.squeeze(1)


class AddColoredNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def forward(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        augmented = self._aug(x)
        return augmented.squeeze(1)


class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def forward(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        augmented = self._aug(x)
        return augmented.squeeze(1)


class Shift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Shift(*args, **kwargs)

    def forward(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        augmented = self._aug(x)
        return augmented.squeeze(1)


class PolarityInversion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PolarityInversion(*args, **kwargs)

    def forward(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        augmented = self._aug(x)
        return augmented.squeeze(1)
