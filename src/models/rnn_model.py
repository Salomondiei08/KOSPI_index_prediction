from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """
    Simple LSTM-based regressor that consumes a window of features and predicts
    the next normalized return.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]  # use final timestep representation
        return self.head(last_hidden).squeeze(-1)
