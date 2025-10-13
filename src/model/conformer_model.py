import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_feats: int, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        output_dim = self._calc_output_dim(n_feats)
        self.linear = nn.Linear(d_model * output_dim, d_model)

    @staticmethod
    def _apply_stride(lengths: torch.Tensor) -> torch.Tensor:
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        return lengths.clamp(min=1)

    @staticmethod
    def _calc_output_dim(dim: int) -> int:
        dim = (dim - 1) // 2 + 1
        dim = (dim - 1) // 2 + 1
        return max(dim, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x.unsqueeze(1))
        batch_size, channels, time_dim, feat_dim = x.shape
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, time_dim, channels * feat_dim)
        )
        x = self.linear(x)
        lengths = self._apply_stride(lengths)
        return x, lengths


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, dropout: float) -> None:
        super().__init__()
        hidden_dim = d_ff if d_ff is not None else math.ceil(d_model * 8 / 3)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_1 = nn.Linear(d_model, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(inputs)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_output, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(attn_output)


class ConvolutionModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        inner_channels = d_model * expansion_factor
        self.pointwise_conv_1 = nn.Conv1d(
            d_model, inner_channels, kernel_size=1, bias=False
        )
        self.glu = nn.GLU(dim=1)
        half_channels = inner_channels // 2
        self.depthwise_conv = nn.Conv1d(
            half_channels,
            half_channels,
            kernel_size=kernel_size,
            groups=half_channels,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(half_channels)
        self.pointwise_conv_2 = nn.Conv1d(
            half_channels, d_model, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(inputs)
        x = rearrange(x, "batch_size seq_len d_model -> batch_size d_model seq_len")
        x = self.pointwise_conv_1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv_2(x)
        x = rearrange(x, "batch_size d_model seq_len -> batch_size seq_len d_model")
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        num_heads: int,
        dropout: float,
        kernel_size: int,
        conv_expansion_factor: int,
    ) -> None:
        super().__init__()
        self.ff_scale = 0.5
        self.ffn_1 = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.conv_module = ConvolutionModule(
            d_model=d_model,
            kernel_size=kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout=dropout,
        )
        self.ffn_2 = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = x + self.ff_scale * self.ffn_1(x)
        x = x + self.self_attention(x, key_padding_mask)
        x = x + self.conv_module(x)
        x = x + self.ff_scale * self.ffn_2(x)
        return self.final_layer_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_feats: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None,
        dropout: float,
        kernel_size: int,
        conv_expansion_factor: int,
    ) -> None:
        super().__init__()
        self.subsampling = Conv2dSubsampling(n_feats=n_feats, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    conv_expansion_factor=conv_expansion_factor,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _lengths_to_padding_mask(
        lengths: torch.Tensor, max_len: int, device: torch.device
    ) -> torch.Tensor:
        lengths_on_device = lengths.to(device)
        positions = torch.arange(max_len, device=device)
        return positions.unsqueeze(0) >= lengths_on_device.unsqueeze(1)

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, output_lengths = self.subsampling(inputs, input_lengths)
        x = self.dropout(x)
        key_padding_mask = self._lengths_to_padding_mask(
            output_lengths, x.size(1), x.device
        )
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x, output_lengths


class Conformer(nn.Module):
    def __init__(
        self,
        n_feats: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        kernel_size: int = 31,
        conv_expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(
            n_feats=n_feats,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            kernel_size=kernel_size,
            conv_expansion_factor=conv_expansion_factor,
        )
        self.classifier = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, **batch):
        inputs = batch["spectrogram"]
        input_lengths = batch["spectrogram_length"]
        encoded, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.classifier(encoded)
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info