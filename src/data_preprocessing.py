from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_fetcher import YahooFinanceConfig, ensure_local_cache

FEATURE_COLUMNS = [
    "norm_close",
    "log_return",
    "norm_volume",
    "is_outlier",
    "ma_5_ratio",
    "ma_20_ratio",
    "volatility_10",
    "volume_change",
    "momentum_10",
]

@dataclass
class PreprocessingConfig:
    source_csv: Path = Path("data/kospi_data.csv")
    output_npz: Path = Path("data/processed.npz")
    window_size: int = 30
    rolling_window: int = 60
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    prefer_api: bool = True
    ticker: str = "^KS11"
    start_date: str = "1983-01-01"
    end_date: str | None = None


def load_raw_data(cfg: PreprocessingConfig) -> pd.DataFrame:
    df = None
    if cfg.prefer_api or not cfg.source_csv.exists():
        try:
            yahoo_cfg = YahooFinanceConfig(
                ticker=cfg.ticker,
                start=cfg.start_date,
                end=cfg.end_date,
                output_csv=cfg.source_csv,
            )
            df = ensure_local_cache(yahoo_cfg, force=False)
        except Exception as exc:  # pragma: no cover - network failures
            print(f"Yahoo Finance download failed ({exc}). Falling back to local CSV.")
    if df is None:
        if not cfg.source_csv.exists():
            raise FileNotFoundError(
                f"Local CSV not found at {cfg.source_csv} and API download disabled/unavailable."
            )
        df = pd.read_csv(cfg.source_csv, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.ffill().bfill()
    return df


def apply_feature_engineering(
    df: pd.DataFrame, cfg: PreprocessingConfig
) -> Tuple[pd.DataFrame, float, float]:
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lower, upper = df["log_return"].quantile([0.01, 0.99])
    df["is_outlier"] = ((df["log_return"] <= lower) | (df["log_return"] >= upper)).astype(int)
    df["log_return"] = df["log_return"].clip(lower, upper)

    close_roll_mean = df["Close"].rolling(cfg.rolling_window, min_periods=1).mean()
    close_roll_std = (
        df["Close"].rolling(cfg.rolling_window, min_periods=1).std().fillna(1.0).replace(0, 1)
    )
    df["norm_close"] = (df["Close"] - close_roll_mean) / close_roll_std
    df["norm_close"] = df["norm_close"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    vol_roll_mean = df["Volume"].rolling(cfg.rolling_window, min_periods=1).mean()
    vol_roll_std = (
        df["Volume"].rolling(cfg.rolling_window, min_periods=1).std().fillna(1.0).replace(0, 1)
    )
    df["norm_volume"] = (df["Volume"] - vol_roll_mean) / vol_roll_std
    df["norm_volume"] = df["norm_volume"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["ma_5_ratio"] = (df["Close"] / df["Close"].rolling(5, min_periods=1).mean()) - 1.0
    df["ma_20_ratio"] = (df["Close"] / df["Close"].rolling(20, min_periods=1).mean()) - 1.0
    df["volatility_10"] = df["log_return"].rolling(10, min_periods=1).std().fillna(0.0)
    df["volume_change"] = (
        df["Volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    df["momentum_10"] = (
        (df["Close"] / df["Close"].shift(10)) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["target_close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["target_close"]).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)
    return df, float(lower), float(upper)


def build_sequences(
    df: pd.DataFrame, cfg: PreprocessingConfig
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets_close = df["target_close"].to_numpy(dtype=np.float32)
    closes = df["Close"].to_numpy(dtype=np.float32)
    dates = df["Date"].to_numpy()

    X, y_returns, y_close, prev_close, target_dates = [], [], [], [], []
    total_sequences = len(df) - cfg.window_size
    for idx in range(total_sequences):
        slice_features = features[idx : idx + cfg.window_size]
        prev_close_value = closes[idx + cfg.window_size - 1]
        next_close_value = closes[idx + cfg.window_size]
        log_ret = float(np.log(next_close_value / prev_close_value))
        target_date = dates[idx + cfg.window_size]
        X.append(slice_features)
        y_returns.append(log_ret)
        y_close.append(next_close_value)
        prev_close.append(prev_close_value)
        target_dates.append(target_date)

    return (
        np.array(X),
        np.array(y_returns),
        np.array(y_close),
        np.array(prev_close),
        np.array(target_dates),
    )


def split_arrays(
    X: np.ndarray,
    y_returns: np.ndarray,
    y_close: np.ndarray,
    prev_close: np.ndarray,
    dates: np.ndarray,
    cfg: PreprocessingConfig,
) -> Dict[str, np.ndarray]:
    total = len(X)
    train_end = int(total * cfg.train_ratio)
    val_end = train_end + int(total * cfg.val_ratio)

    arrays = {
        "X_train": X[:train_end],
        "y_train": y_returns[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y_returns[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y_returns[val_end:],
        "target_close_train": y_close[:train_end],
        "target_close_val": y_close[train_end:val_end],
        "target_close_test": y_close[val_end:],
        "base_close_train": prev_close[:train_end],
        "base_close_val": prev_close[train_end:val_end],
        "base_close_test": prev_close[val_end:],
        "test_dates": dates[val_end:],
    }
    return arrays


def preprocess_and_save(cfg: PreprocessingConfig) -> Dict[str, np.ndarray]:
    df = load_raw_data(cfg)
    df, lower, upper = apply_feature_engineering(df, cfg)
    X, y_returns, y_close, prev_close, dates = build_sequences(df, cfg)
    arrays = split_arrays(X, y_returns, y_close, prev_close, dates, cfg)
    target_mean = float(np.mean(arrays["y_train"]))
    target_std = float(np.std(arrays["y_train"]))
    if target_std == 0 or np.isnan(target_std):
        target_std = 1.0
    for split in ("train", "val", "test"):
        arrays[f"y_{split}"] = (arrays[f"y_{split}"] - target_mean) / target_std

    arrays["target_mean"] = np.array([target_mean], dtype=np.float32)
    arrays["target_std"] = np.array([target_std], dtype=np.float32)
    arrays["outlier_lower"] = np.array([lower], dtype=np.float32)
    arrays["outlier_upper"] = np.array([upper], dtype=np.float32)
    arrays["rolling_window"] = np.array([cfg.rolling_window], dtype=np.int32)
    arrays["window_size"] = np.array([cfg.window_size], dtype=np.int32)
    cfg.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.output_npz, **arrays)
    return arrays


if __name__ == "__main__":
    preprocess_and_save(PreprocessingConfig())
