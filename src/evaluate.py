from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

# Force a non-interactive backend so evaluation works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_preprocessing import PreprocessingConfig, preprocess_and_save
from src.models.hybrid_model import HybridRegressor
from src.models.rnn_model import LSTMRegressor
from src.models.transformer_model import TransformerRegressor
from src.utils import (
    build_dataloader,
    directional_accuracy,
    load_processed_arrays,
    mae,
    rmse,
    save_json,
)


def load_model(model_name: str, input_size: int, cfg) -> torch.nn.Module:
    if model_name == "rnn":
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    elif model_name == "transformer":
        model = TransformerRegressor(
            input_size=input_size,
            d_model=cfg.transformer_dim,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    else:
        model = HybridRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            transformer_dim=cfg.transformer_dim,
            dropout=cfg.dropout,
        )
    state_path = cfg.models_dir / f"{model_name}.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    return model


def collect_predictions(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            if isinstance(output, tuple):
                output = output[0]
            preds.append(output.cpu().numpy())
    return np.concatenate(preds)


def save_plots(
    dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray, model_name: str, reports_dir: Path
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.title(f"{model_name} · Actual vs Predicted")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(reports_dir / f"{model_name}_actual_vs_pred.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    residuals = predicted - actual
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title(f"{model_name} · Residual Histogram")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_residuals.png")
    plt.close()


def save_attention_heatmap(model, sample_batch: torch.Tensor, model_name: str, reports_dir: Path, device: torch.device) -> None:
    if not hasattr(model, "forward"):
        return
    with torch.no_grad():
        output = model(sample_batch.to(device), return_attn=True)
    if isinstance(output, tuple):
        _, attn = output
    else:
        return
    if attn is None:
        return
    # attn shape: (batch, heads, seq, seq); average batch + heads for a 2D map
    weights = attn.mean(dim=(0, 1)).cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(f"{model_name} · Attention Heatmap")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_attention.png")
    plt.close()


def evaluate_models(pre_cfg: PreprocessingConfig, train_cfg) -> Dict[str, Dict[str, float]]:
    if not train_cfg.processed_npz.exists():
        preprocess_and_save(pre_cfg)
    arrays = load_processed_arrays(train_cfg.processed_npz)
    input_size = arrays["X_test"].shape[-1]
    target_mean = float(arrays.get("target_mean", np.array([0.0]))[0])
    target_std = float(arrays.get("target_std", np.array([1.0]))[0])
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    test_loader = build_dataloader(arrays["X_test"], arrays["y_test"], batch_size=train_cfg.batch_size, shuffle=False)

    metrics: Dict[str, Dict[str, float]] = {}
    predictions_records = []
    reports_dir = Path("reports")
    dates = arrays["test_dates"]
    base_close = arrays["base_close_test"]
    actual_close = arrays["target_close_test"]

    for name in ("rnn", "transformer", "hybrid"):
        model = load_model(name, input_size, train_cfg)
        model.to(device)
        preds = collect_predictions(model, test_loader, device)
        predicted_returns = preds * target_std + target_mean
        predicted_close = base_close * np.exp(predicted_returns)
        metrics[name] = {
            "RMSE": rmse(predicted_close, actual_close),
            "MAE": mae(predicted_close, actual_close),
            "DirectionalAccuracy": directional_accuracy(predicted_close, actual_close),
        }
        save_plots(dates, actual_close, predicted_close, name, reports_dir)
        if name == "transformer":
            batch_sample = torch.tensor(arrays["X_test"][:1], dtype=torch.float32)
            save_attention_heatmap(model, batch_sample, name, reports_dir, device)

        for d, a, p in zip(dates, actual_close, predicted_close):
            predictions_records.append(
                {
                    "date": pd.to_datetime(d).date().isoformat(),
                    "model": name,
                    "actual": float(a),
                    "predicted": float(p),
                    "split": "test",
                }
            )

    save_json(metrics, reports_dir / "evaluation_metrics.json")
    pd.DataFrame(predictions_records).to_csv(reports_dir / "predictions.csv", index=False)
    return metrics


if __name__ == "__main__":
    from src.train import TrainingConfig

    metrics = evaluate_models(PreprocessingConfig(), TrainingConfig())
    print(metrics)
