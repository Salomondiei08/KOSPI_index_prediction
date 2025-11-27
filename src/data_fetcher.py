from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - yfinance may not be installed
    yf = None


@dataclass
class YahooFinanceConfig:
    """Configuration for downloading KOSPI data via Yahoo Finance."""

    ticker: str = "^KS11"
    start: str = "1983-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    output_csv: Path = Path("data/kospi_data.csv")


def download_kospi_history(cfg: YahooFinanceConfig) -> pd.DataFrame:
    """
    Download KOSPI historical OHLCV data using Yahoo Finance and persist to CSV.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Install it (pip install yfinance) to fetch live data."
        )
    df = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=cfg.end or date.today().isoformat(),
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError("Yahoo Finance returned an empty dataframe for the requested range.")

    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "AdjClose",
            "Volume": "Volume",
        }
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False)
    return df


def ensure_local_cache(cfg: YahooFinanceConfig, force: bool = False) -> pd.DataFrame:
    """
    Fetch data if the cache is missing or force=True, else load from disk.
    """
    if force or not cfg.output_csv.exists():
        return download_kospi_history(cfg)
    return pd.read_csv(cfg.output_csv, parse_dates=["Date"])

