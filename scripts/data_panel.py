"""data_panel.py — Load parquets, build price/return panels."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import OPEN_BAR

# Data lives at project root / data /
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data"


def load_bars(
    years: int | list[int],
    regular_hours_only: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw bar parquets for given years."""
    if isinstance(years, int):
        years = [years]
    frames = []
    for y in years:
        path = RAW_DIR / f"bars_30m_{y}.parquet"
        read_cols = columns
        drop_flag = False
        if read_cols is not None:
            read_cols = list(dict.fromkeys(read_cols))
            if regular_hours_only and "is_regular_hours" not in read_cols:
                read_cols = read_cols + ["is_regular_hours"]
                drop_flag = True
        df = pd.read_parquet(path, columns=read_cols)
        if regular_hours_only:
            df = df.loc[df["is_regular_hours"]]
        if drop_flag:
            df = df.drop(columns=["is_regular_hours"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return out.reset_index(drop=True)


def build_price_panel(years: int | list[int], price_col: str = "close") -> pd.DataFrame:
    """Wide price panel: rows=timestamp_utc, cols=symbols, values=close."""
    if isinstance(years, int):
        years = [years]
    cols = ["symbol", "timestamp_utc", price_col]
    per_year = []
    for y in tqdm(years, desc="Loading years", unit="yr"):
        df = load_bars(y, regular_hours_only=True, columns=cols)
        df = df.drop_duplicates(["timestamp_utc", "symbol"], keep="last")
        wide = df.pivot(index="timestamp_utc", columns="symbol", values=price_col)
        per_year.append(wide)
    panel = pd.concat(per_year, axis=0) if len(per_year) > 1 else per_year[0]
    panel = panel.sort_index().reindex(sorted(panel.columns), axis=1)
    panel.index.name = "timestamp_utc"
    panel.columns.name = "symbol"
    return panel


def build_dollar_volume_panel(years: int | list[int]) -> pd.DataFrame:
    """Wide dollar-volume panel: rows=timestamp_utc, cols=symbols, values=close*volume."""
    if isinstance(years, int):
        years = [years]
    cols = ["symbol", "timestamp_utc", "close", "volume"]
    per_year = []
    for y in tqdm(years, desc="Loading dollar volume", unit="yr"):
        df = load_bars(y, regular_hours_only=True, columns=cols)
        df["dollar_vol"] = df["close"] * df["volume"]
        df = df.drop_duplicates(["timestamp_utc", "symbol"], keep="last")
        wide = df.pivot(index="timestamp_utc", columns="symbol", values="dollar_vol")
        per_year.append(wide)
    panel = pd.concat(per_year, axis=0) if len(per_year) > 1 else per_year[0]
    panel = panel.sort_index().reindex(sorted(panel.columns), axis=1)
    panel.index.name = "timestamp_utc"
    panel.columns.name = "symbol"
    return panel


def build_return_panel(
    price_panel: pd.DataFrame, drop_overnight: bool = True
) -> pd.DataFrame:
    """Bar-to-bar log returns. Overnight (09:30) rows set to NaN."""
    ret = np.log(price_panel / price_panel.shift(1))
    if drop_overnight:
        ny_time = price_panel.index.tz_convert("America/New_York").strftime("%H:%M")
        overnight_mask = np.asarray(ny_time) == OPEN_BAR
        ret.loc[overnight_mask, :] = np.nan
    return ret
