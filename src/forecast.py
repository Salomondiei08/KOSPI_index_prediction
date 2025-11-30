from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    PreprocessingConfig,
    apply_feature_engineering,
    load_raw_data,
)
from src.evaluate import load_model
from src.train import TrainingConfig
from src.utils import load_processed_arrays



@dataclass
class ForecastConfig:
    start_date: str = "2025-12-01"
    end_date: str = "2025-12-05"
    output_csv: Path = Path("reports/forecast_dec_2025.csv")


def _prepare_feature_frame(cfg: PreprocessingConfig) -> Dict:
    raw = load_raw_data(cfg)
    engineered, lower, upper = apply_feature_engineering(raw, cfg)
    return {
        "df": engineered,
        "outlier_lower": lower,
        "outlier_upper": upper,
    }


def _compute_norm(value: float, history: List[float], window: int) -> float:
    relevant = history[-window:] if window <= len(history) else history
    mean = float(np.mean(relevant))
    std = float(np.std(relevant))
    if std == 0 or math.isnan(std):
        std = 1.0
    return (value - mean) / std


def _ma_ratio(current_close: float, closes: List[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = float(np.mean(window))
    if mean == 0 or math.isnan(mean):
        return 0.0
    return (current_close / mean) - 1.0


def _momentum(closes: List[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    baseline = closes[-lookback - 1]
    if baseline == 0:
        return 0.0
    return (closes[-1] / baseline) - 1.0


def _prepare_initial_states(df: pd.DataFrame, cfg: PreprocessingConfig) -> Dict:
    window = df[FEATURE_COLUMNS].tail(cfg.window_size).to_numpy(dtype=np.float32)
    closes = df["Close"].tolist()
    volumes = df["Volume"].tolist()
    log_returns = df["log_return"].tolist()
    return {
        "window": window,
        "closes": closes,
        "volumes": volumes,
        "log_returns": log_returns,
    }


def _build_feature_row(
    predicted_close: float,
    previous_close: float,
    closes: List[float],
    volumes: List[float],
    log_returns: List[float],
    rolling_window: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    norm_close = _compute_norm(predicted_close, closes, rolling_window)
    norm_volume = _compute_norm(volumes[-1], volumes, rolling_window)
    new_log_return = math.log(predicted_close / previous_close)
    is_outlier = int(new_log_return <= lower or new_log_return >= upper)
    volatility_10 = float(np.std(log_returns[-10:])) if len(log_returns) >= 2 else 0.0
    volume_change = (
        (volumes[-1] / volumes[-2]) - 1.0 if len(volumes) >= 2 and volumes[-2] != 0 else 0.0
    )
    momentum_10 = _momentum(closes, 10)
    feature_values = {
        "norm_close": norm_close,
        "log_return": new_log_return,
        "norm_volume": norm_volume,
        "is_outlier": is_outlier,
        "ma_5_ratio": _ma_ratio(predicted_close, closes, 5),
        "ma_20_ratio": _ma_ratio(predicted_close, closes, 20),
        "volatility_10": volatility_10,
        "volume_change": volume_change,
        "momentum_10": momentum_10,
    }
    return np.array([feature_values[col] for col in FEATURE_COLUMNS], dtype=np.float32)


def generate_forecast(
    pre_cfg: PreprocessingConfig,
    train_cfg: TrainingConfig,
    forecast_cfg: ForecastConfig,
) -> pd.DataFrame:
    """
    Produce multi-day forecasts (default Dec 1-5, 2025) for each trained model and
    store the results under reports/.
    """
    processed_arrays = load_processed_arrays(train_cfg.processed_npz)
    if not processed_arrays:
        raise RuntimeError("Processed arrays missing. Run preprocessing first.")

    rolling_window = int(processed_arrays.get("rolling_window", np.array([60]))[0])
    target_mean = float(processed_arrays.get("target_mean", np.array([0.0]))[0])
    target_std = float(processed_arrays.get("target_std", np.array([1.0]))[0])
    feature_frame = _prepare_feature_frame(pre_cfg)
    df = feature_frame["df"]
    lower = feature_frame["outlier_lower"]
    upper = feature_frame["outlier_upper"]
    states = _prepare_initial_states(df, pre_cfg)

    forecast_dates = pd.bdate_range(forecast_cfg.start_date, forecast_cfg.end_date)
    if forecast_dates.empty:
        raise ValueError("No business days fall within the provided forecast window.")

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    closes_history = states["closes"]
    volumes_history = states["volumes"]
    log_returns_history = states["log_returns"]
    window = states["window"]

    all_records: List[Dict] = []
    input_size = window.shape[-1]

    for model_name in ("rnn", "transformer", "hybrid"):
        model = load_model(model_name, input_size, train_cfg)
        model.to(device)

        working_window = window.copy()
        closes = closes_history.copy()
        volumes = volumes_history.copy()
        log_returns = log_returns_history.copy()

        for target_date in forecast_dates:
            features_tensor = torch.tensor(working_window, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(features_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                normalized_return = float(output.squeeze().cpu().item())
            predicted_return = normalized_return * target_std + target_mean
            predicted_return = float(np.clip(predicted_return, lower, upper))
            previous_close = closes[-1]
            predicted_close = previous_close * math.exp(predicted_return)

            volumes.append(volumes[-1])  # hold last observed volume for short horizon
            closes.append(predicted_close)
            log_returns.append(predicted_return)

            new_feature_row = _build_feature_row(
                predicted_close=predicted_close,
                previous_close=previous_close,
                closes=closes,
                volumes=volumes,
                log_returns=log_returns,
                rolling_window=rolling_window,
                lower=lower,
                upper=upper,
            )
            working_window = np.vstack([working_window[1:], new_feature_row])

            all_records.append(
                {
                    "date": target_date.date().isoformat(),
                    "model": model_name,
                    "predicted_close": float(predicted_close),
                    "predicted_log_return": predicted_return,
                }
            )

    forecast_df = pd.DataFrame(all_records)
    forecast_cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_cfg.output_csv, index=False)
    return forecast_df


if __name__ == "__main__":
    from src.train import TrainingConfig

    generate_forecast(PreprocessingConfig(), TrainingConfig(), ForecastConfig())
