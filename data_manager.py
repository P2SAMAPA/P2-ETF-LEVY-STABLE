"""data_manager.py — Data loading for Lévy / Alpha-Stable engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ALL_TICKERS = sorted(set(
    config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS
))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet and return (log_returns, macro_df).

    Returns
    -------
    log_returns : DataFrame  shape (T, n_etfs)  — daily log returns
    macro_df    : DataFrame  shape (T, n_macro)  — VIX, DXY, T10Y2Y, TBILL_3M
    """
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )

    df = pd.read_parquet(file_path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    # ── ETF log returns ───────────────────────────────────────────────────────
    available   = [t for t in ALL_TICKERS if t in df.columns]
    prices      = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── Macro ─────────────────────────────────────────────────────────────────
    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df
