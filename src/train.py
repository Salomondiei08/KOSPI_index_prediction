from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_preprocessing import PreprocessingConfig, preprocess_and_save
from src.models.hybrid_model import HybridRegressor
from src.models.rnn_model import LSTMRegressor
from src.models.transformer_model import TransformerRegressor
from src.utils import TrainingSummary, build_dataloader, load_processed_arrays


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 5e-4
    patience: int = 5
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    transformer_dim: int = 128
    nhead: int = 8
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    processed_npz: Path = Path("data/processed.npz")
    models_dir: Path = Path("models")


def ensure_processed_data(pre_cfg: PreprocessingConfig, train_cfg: TrainingConfig) -> Dict[str, np.ndarray]:
    if train_cfg.processed_npz.exists():
        return load_processed_arrays(train_cfg.processed_npz)
    return preprocess_and_save(pre_cfg)


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        if isinstance(preds, tuple):
            preds = preds[0]
        loss = criterion(preds, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(batch_x)
    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        if isinstance(preds, tuple):
            preds = preds[0]
        loss = criterion(preds, batch_y)
        total += loss.item() * len(batch_x)
    return total / len(loader.dataset)


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
) -> TrainingSummary:
    device = torch.device(cfg.device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_val = math.inf
    patience_counter = 0
    history: List[Dict[str, float]] = []

    model_path = cfg.models_dir / f"{model_name}.pt"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        tqdm.write(
            f"[{model_name}] Epoch {epoch}/{cfg.epochs} · Train {train_loss:.4f} · Val {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                tqdm.write(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    return TrainingSummary(model_name=model_name, best_val_loss=best_val, epochs_trained=len(history))


def run_training_pipeline(pre_cfg: PreprocessingConfig, train_cfg: TrainingConfig) -> List[TrainingSummary]:
    arrays = ensure_processed_data(pre_cfg, train_cfg)
    train_loader = build_dataloader(arrays["X_train"], arrays["y_train"], train_cfg.batch_size, shuffle=True)
    val_loader = build_dataloader(arrays["X_val"], arrays["y_val"], train_cfg.batch_size, shuffle=False)

    input_size = arrays["X_train"].shape[-1]

    jobs = {
        "rnn": LSTMRegressor(
            input_size=input_size,
            hidden_size=train_cfg.hidden_size,
            num_layers=train_cfg.num_layers,
            dropout=train_cfg.dropout,
        ),
        "transformer": TransformerRegressor(
            input_size=input_size,
            d_model=train_cfg.transformer_dim,
            nhead=train_cfg.nhead,
            num_layers=train_cfg.num_layers,
            dropout=train_cfg.dropout,
        ),
        "hybrid": HybridRegressor(
            input_size=input_size,
            hidden_size=train_cfg.hidden_size,
            transformer_dim=train_cfg.transformer_dim,
            dropout=train_cfg.dropout,
        ),
    }

    summaries: List[TrainingSummary] = []
    for name, model in jobs.items():
        summary = train_model(model, name, train_loader, val_loader, train_cfg)
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    summaries = run_training_pipeline(PreprocessingConfig(), TrainingConfig())
    for s in summaries:
        print(s)
