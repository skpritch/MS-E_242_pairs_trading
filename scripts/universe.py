"""universe.py — Per-fold liquid universe + coverage filter."""

from __future__ import annotations

import pandas as pd

from .data_panel import load_bars


def liquid_universe(
    years: int | list[int],
    top_n: int = 150,
    date_range: tuple[str, str] | None = None,
) -> list[str]:
    """Top-N symbols by median per-bar dollar volume."""
    if isinstance(years, int):
        years = [years]
    cols = ["symbol", "date_ny", "close", "volume"]
    parts = []
    for y in years:
        df = load_bars(y, regular_hours_only=True, columns=cols)
        if date_range is not None:
            start, end = date_range
            df = df.loc[(df["date_ny"] >= start) & (df["date_ny"] <= end)]
        if df.empty:
            continue
        dv = (df["close"] * df["volume"]).to_numpy()
        parts.append(pd.DataFrame({"symbol": df["symbol"].to_numpy(), "dollar_vol": dv}))
    if not parts:
        return []
    allbars = pd.concat(parts, ignore_index=True)
    med = allbars.groupby("symbol")["dollar_vol"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def filter_coverage(
    panel: pd.DataFrame, symbols: list[str], min_coverage: float = 0.95
) -> list[str]:
    """Keep symbols with >= min_coverage fraction of non-NaN bars."""
    sub = panel.reindex(columns=[s for s in symbols if s in panel.columns])
    coverage = sub.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    return keep
