from __future__ import annotations

import torch
from torch import nn

from .transformer_model import PositionalEncoding, TransformerBlock


class HybridRegressor(nn.Module):
    """
    Hybrid encoder: LSTM summarizes the window, transformer refines token representations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        transformer_dim: int = 128,
        nhead: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.bridge = nn.Linear(hidden_size, transformer_dim)
        self.positional = PositionalEncoding(transformer_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=transformer_dim,
                    nhead=nhead,
                    dim_feedforward=transformer_dim * 2,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        enc_out, _ = self.encoder(x)
        x = self.bridge(enc_out)
        x = self.positional(x)
        for layer in self.layers:
            x, _ = layer(x, need_weights=False)
        last_token = x[:, -1, :]
        return self.head(last_token).squeeze(-1)
