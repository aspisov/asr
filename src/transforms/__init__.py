from __future__ import annotations

import torch


class Normalize1D(torch.nn.Module):
    """Normalize 1D tensors with configurable mean and std."""

    def __init__(self, mean: float = 0.0, std: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + self.eps)
