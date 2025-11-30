from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x, attn_weights if need_weights else None


class TransformerRegressor(nn.Module):
    """
    Lightweight transformer encoder for time-series regression.
    Returns a scalar normalized return per window.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (batch, seq, features)
        x = self.input_proj(x)
        x = self.positional(x)

        attn_to_return = None
        for layer in self.layers:
            x, attn = layer(x, need_weights=return_attn)
            if return_attn and attn is not None:
                attn_to_return = attn  # keep last layer's attention weights

        pooled = x[:, -1, :]
        preds = self.head(pooled).squeeze(-1)
        if return_attn:
            return preds, attn_to_return
        return preds
