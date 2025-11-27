from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch dataset wrapping time-series windows."""

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def build_dataloader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TimeSeriesDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(preds - targets))))


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def directional_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    if len(preds) < 2:
        return float("nan")
    pred_moves = np.sign(np.diff(preds))
    target_moves = np.sign(np.diff(targets))
    match = np.mean(pred_moves == target_moves)
    return float(match)


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_processed_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclass
class TrainingSummary:
    model_name: str
    best_val_loss: float
    epochs_trained: int

