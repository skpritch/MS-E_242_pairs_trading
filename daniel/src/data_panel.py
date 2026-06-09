"""
data_panel.py — MSE242 pairs-trading DATA FOUNDATION (Agent A).

This module is the single, reusable data layer that every downstream component
(encoder, pair-selection, signal, backtest) depends on. It contains NO trading,
ML, or backtest logic — only loading, aligning, and caching of 30-minute
regular-hours bars.

Raw inputs
----------
`financial_trading_data/bars_30m_YYYY.parquet` for years 2016..2026.
Schema (verified): symbol, timestamp_utc (datetime64[us, UTC]), timestamp_ny,
date_ny ('YYYY-MM-DD'), time_ny ('HH:MM'), session, is_regular_hours (bool),
open, high, low, close, volume, vwap, trades, source_minute_count,
interval_minutes, source_date_utc, source_dataset.

Verified data facts (see coverage_report / __main__ output)
-----------------------------------------------------------
* Regular hours = exactly 13 bars/day, time_ny in
  {09:30,10:00,...,15:00,15:30}. Each label is the bar START; the 15:30 bar
  covers 15:30-16:00, so the trading session is 09:30-16:00 inclusive.
* `close` is never NaN for regular-hours rows; `volume` is never NaN.
* `vwap` is frequently 100% NaN -> we ALWAYS use `close` as the price.
* No duplicate (symbol, timestamp_utc) rows in regular hours.
* 798 symbols are present in regular hours in every year 2016..2026
  (the "full-history" universe). The union is ~1801 symbols.

Price-panel design
------------------
`build_price_panel` returns a WIDE DataFrame:
    index   = sorted DatetimeIndex of distinct regular-hours bar timestamps
              (timestamp_utc, tz-aware UTC). One row per 30m bar.
    columns = symbol
    values  = price (default close)
A cell is NaN when that symbol had no bar at that timestamp (it did not trade,
was not listed, or that bar is missing). This is the ONLY source of NaN in the
price panel — closes themselves are never NaN.

Return-panel design
-------------------
`build_return_panel` computes bar-to-bar LOG returns r_t = ln(P_t / P_{t-1}).
The first regular bar of each trading day (time_ny == '09:30') spans the
overnight gap from the prior 16:00 close to today's 09:30. With
`drop_overnight=True` (default) those overnight returns are set to NaN so an
overnight gap is never treated as an intraday move. Intraday returns are the
12 transitions 09:30->10:00 ... 15:00->15:30 per day.

Run end-to-end:  `.venv/bin/python -m src.data_panel`
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "financial_trading_data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Expected regular-hours intraday grid (bar START labels in NY time).
REGULAR_TIME_GRID = [
    "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
    "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
]
BARS_PER_DAY = len(REGULAR_TIME_GRID)  # 13
OPEN_BAR = "09:30"  # first bar of the day == overnight gap

# Default cache target (documented in DATA_README.md).
DEFAULT_YEARS = list(range(2016, 2026))  # 2016..2025; hold 2026 as final OOS.
DEFAULT_TOP_N = 200


def _raw_path(year: int) -> str:
    return os.path.join(RAW_DIR, f"bars_30m_{year}.parquet")


def _normalize_years(years: Iterable[int] | int) -> list[int]:
    if isinstance(years, int):
        return [years]
    return list(years)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_bars(
    years: Iterable[int] | int,
    regular_hours_only: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate raw bar parquet files for the given year(s).

    Parameters
    ----------
    years : int or iterable of int
        Calendar years to load (each maps to one parquet file).
    regular_hours_only : bool, default True
        If True, keep only rows where ``is_regular_hours`` is True.
    columns : sequence of str, optional
        Subset of columns to read (memory-friendly). ``is_regular_hours`` is
        always read internally when filtering, then dropped if you did not
        request it.

    Returns
    -------
    DataFrame
        Concatenated rows for all requested years, row order = file order then
        on-disk order. Index is a fresh RangeIndex.
    """
    years = _normalize_years(years)

    read_cols = None
    drop_flag = False
    if columns is not None:
        read_cols = list(dict.fromkeys(columns))  # de-dupe, preserve order
        if regular_hours_only and "is_regular_hours" not in read_cols:
            read_cols = read_cols + ["is_regular_hours"]
            drop_flag = True

    frames = []
    for y in years:
        path = _raw_path(y)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raw file for {y}: {path}")
        df = pd.read_parquet(path, columns=read_cols)
        if regular_hours_only:
            df = df.loc[df["is_regular_hours"]]
        if drop_flag:
            df = df.drop(columns=["is_regular_hours"])
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Price panel (wide)
# --------------------------------------------------------------------------- #


def build_price_panel(
    years: Iterable[int] | int,
    price_col: str = "close",
) -> pd.DataFrame:
    """Build a WIDE price panel: one row per 30m regular-hours bar.

    Built year-by-year and pivoted incrementally to keep peak memory low
    (we never hold all raw long-form rows for all years at once).

    Parameters
    ----------
    years : int or iterable of int
    price_col : str, default 'close'
        Price column to use. ``close`` is recommended and never NaN; ``vwap``
        is frequently entirely NaN and only kept as a documented fallback.

    Returns
    -------
    DataFrame
        index   : sorted tz-aware DatetimeIndex (UTC) of bar timestamps,
                  named 'timestamp_utc'.
        columns : symbol (sorted), columns.name = 'symbol'.
        values  : price (float). NaN == symbol had no bar at that timestamp.
    """
    years = _normalize_years(years)
    cols = ["symbol", "timestamp_utc", price_col]

    per_year = []
    for y in years:
        df = load_bars(y, regular_hours_only=True, columns=cols)
        # Guard against any accidental dup (verified none, but be safe).
        df = df.drop_duplicates(["timestamp_utc", "symbol"], keep="last")
        wide = df.pivot(index="timestamp_utc", columns="symbol", values=price_col)
        per_year.append(wide)

    panel = pd.concat(per_year, axis=0) if len(per_year) > 1 else per_year[0]
    panel = panel.sort_index()
    panel = panel.reindex(sorted(panel.columns), axis=1)
    panel.index.name = "timestamp_utc"
    panel.columns.name = "symbol"
    return panel


# --------------------------------------------------------------------------- #
# Return panel
# --------------------------------------------------------------------------- #


def build_return_panel(
    price_panel: pd.DataFrame,
    drop_overnight: bool = True,
) -> pd.DataFrame:
    """Bar-to-bar LOG returns from a wide price panel.

    r_t = ln(P_t / P_{t-1}) computed down each symbol column on the shared
    bar grid (the panel index). Because the index is the regular-hours grid,
    consecutive rows within a day are 30m apart, but the transition from one
    day's 16:00 close (15:30 bar) to the next day's 09:30 bar is an OVERNIGHT
    gap.

    Parameters
    ----------
    price_panel : DataFrame
        Wide panel from :func:`build_price_panel` (tz-aware DatetimeIndex).
    drop_overnight : bool, default True
        If True, set every return whose row is the first bar of a trading day
        (NY time == '09:30') to NaN, so overnight gaps are not mistaken for
        intraday moves. If False, those rows keep the raw close-to-open return.

    Returns
    -------
    DataFrame
        Same index/columns as ``price_panel``. First overall row is all-NaN
        (no prior bar). NaN also where price was NaN at t or t-1, and (if
        requested) on overnight rows.
    """
    ret = np.log(price_panel / price_panel.shift(1))

    if drop_overnight:
        # NY local time of each bar; first bar of a session is 09:30.
        ny_time = price_panel.index.tz_convert("America/New_York").strftime("%H:%M")
        overnight_mask = np.asarray(ny_time) == OPEN_BAR
        ret.loc[overnight_mask, :] = np.nan

    return ret


# --------------------------------------------------------------------------- #
# Liquidity-filtered universe
# --------------------------------------------------------------------------- #


def liquid_universe(
    years: Iterable[int] | int,
    top_n: int = 200,
    date_range: tuple[str, str] | None = None,
) -> list[str]:
    """Top-N symbols by median per-bar dollar volume over a window.

    Dollar volume per bar = close * volume. We rank symbols by the MEDIAN of
    this quantity across all regular-hours bars in the window (median is robust
    to single huge-volume bars). This is the per-walk-forward universe filter:
    downstream code calls this with the train window's ``date_range`` so the
    tradeable universe is chosen WITHOUT lookahead into the test window.

    Parameters
    ----------
    years : int or iterable of int
        Years to scan (should cover ``date_range``).
    top_n : int, default 200
    date_range : (start, end) of 'YYYY-MM-DD' strings, optional, INCLUSIVE.
        If given, restrict to bars with date_ny in [start, end].

    Returns
    -------
    list[str]
        Symbols sorted by descending median dollar volume (most liquid first).
    """
    years = _normalize_years(years)
    cols = ["symbol", "date_ny", "close", "volume"]

    parts = []
    for y in years:
        df = load_bars(y, regular_hours_only=True, columns=cols)
        if date_range is not None:
            start, end = date_range
            df = df.loc[(df["date_ny"] >= start) & (df["date_ny"] <= end)]
        if df.empty:
            continue
        dv = (df["close"] * df["volume"]).rename("dollar_vol")
        parts.append(pd.DataFrame({"symbol": df["symbol"].to_numpy(),
                                    "dollar_vol": dv.to_numpy()}))

    if not parts:
        return []

    allbars = pd.concat(parts, ignore_index=True)
    med = allbars.groupby("symbol")["dollar_vol"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #


def coverage_report(years: Iterable[int] | int) -> pd.DataFrame:
    """Per-symbol coverage diagnostics over the given years.

    Returns one row per symbol with:
        n_bars        : regular-hours bars observed
        first_date    : earliest date_ny seen
        last_date     : latest date_ny seen
        n_days        : distinct trading days the symbol appears on
        expected_bars : (# distinct trading days in the WHOLE panel that fall
                         within [first_date, last_date]) * 13
        coverage_frac : n_bars / expected_bars (fraction of expected bars
                        present while the symbol was "alive").

    ``coverage_frac`` near 1.0 means the symbol trades essentially every bar of
    every day it is listed; lower values flag thin/intermittent names.
    """
    years = _normalize_years(years)
    cols = ["symbol", "date_ny"]

    parts = [load_bars(y, regular_hours_only=True, columns=cols) for y in years]
    allbars = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    all_days = np.sort(allbars["date_ny"].unique())  # global trading-day calendar

    g = allbars.groupby("symbol")
    n_bars = g.size().rename("n_bars")
    first_date = g["date_ny"].min().rename("first_date")
    last_date = g["date_ny"].max().rename("last_date")
    n_days = g["date_ny"].nunique().rename("n_days")

    rep = pd.concat([n_bars, first_date, last_date, n_days], axis=1)

    # Expected bars = trading days in the global calendar within the symbol's
    # alive window, times 13 bars/day.
    left = np.searchsorted(all_days, rep["first_date"].to_numpy(), side="left")
    right = np.searchsorted(all_days, rep["last_date"].to_numpy(), side="right")
    expected_days = right - left
    rep["expected_bars"] = expected_days * BARS_PER_DAY
    rep["coverage_frac"] = rep["n_bars"] / rep["expected_bars"]

    return rep.sort_values("n_bars", ascending=False)


# --------------------------------------------------------------------------- #
# Validation helpers (used by __main__)
# --------------------------------------------------------------------------- #


def _validate_one_year(year: int) -> dict:
    """Cheap structural validation on a single year. Returns a dict of facts."""
    df = load_bars(year, regular_hours_only=True,
                   columns=["symbol", "timestamp_utc", "date_ny", "time_ny", "close"])
    bpd = df.groupby(["symbol", "date_ny"]).size()
    grid_ok = sorted(df["time_ny"].unique()) == REGULAR_TIME_GRID
    return {
        "year": year,
        "regular_rows": len(df),
        "n_symbols": df["symbol"].nunique(),
        "time_grid_ok": grid_ok,
        "max_bars_per_day": int(bpd.max()),
        "pct_days_full_13": float((bpd == BARS_PER_DAY).mean()),
        "close_nan_frac": float(df["close"].isna().mean()),
        "dup_symbol_ts": int(df.duplicated(["symbol", "timestamp_utc"]).sum()),
    }


def full_history_universe(years: Iterable[int]) -> list[str]:
    """Symbols present in regular hours in EVERY one of ``years``."""
    years = _normalize_years(years)
    sets = []
    for y in years:
        df = load_bars(y, regular_hours_only=True, columns=["symbol"])
        sets.append(set(df["symbol"].unique()))
    return sorted(set.intersection(*sets)) if sets else []


# --------------------------------------------------------------------------- #
# CLI / artifact builder
# --------------------------------------------------------------------------- #


def _human(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def build_and_cache(
    years: list[int] = DEFAULT_YEARS,
    top_n: int = DEFAULT_TOP_N,
    out_dir: str = PROCESSED_DIR,
) -> None:
    """Build default artifacts and print a concise data-quality report.

    Artifacts written to ``out_dir``:
      * close_panel_full_2016_2025.parquet      (wide close, full-history syms)
      * logret_panel_full_2016_2025.parquet     (overnight-dropped log returns)
      * close_panel_liquid200_2016_2025.parquet (top-200 liquid subset)
      * logret_panel_liquid200_2016_2025.parquet
      * coverage_report.csv
      * liquid_universe_top200.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    span = f"{years[0]}_{years[-1]}"
    print("=" * 72)
    print(f"MSE242 data_panel — building default artifacts for {years[0]}..{years[-1]}")
    print("=" * 72)

    # --- 1. Structural validation (one cheap pass on a few representative years)
    print("\n[1] Structural validation")
    for y in (years[0], years[len(years) // 2], years[-1]):
        f = _validate_one_year(y)
        print(f"    {f['year']}: rows={f['regular_rows']:,} syms={f['n_symbols']} "
              f"grid_ok={f['time_grid_ok']} max_bars/day={f['max_bars_per_day']} "
              f"full13={f['pct_days_full_13']:.1%} close_nan={f['close_nan_frac']:.4f} "
              f"dups={f['dup_symbol_ts']}")

    # --- 2. Universes
    print("\n[2] Universes")
    full_syms = full_history_universe(years)
    print(f"    full-history symbols (present every year {years[0]}..{years[-1]}): "
          f"{len(full_syms)}")
    liquid = liquid_universe(years, top_n=top_n)
    print(f"    liquid top-{top_n} by median dollar volume: {len(liquid)} symbols")
    print(f"    most liquid 10: {liquid[:10]}")

    pd.Series(liquid, name="symbol").to_csv(
        os.path.join(out_dir, f"liquid_universe_top{top_n}.csv"), index=False)

    # --- 3. Full close panel + returns, restricted to full-history symbols
    print("\n[3] Building FULL-history close panel + log-return panel")
    close_full = build_price_panel(years, price_col="close")
    # Restrict columns to the full-history universe for a dense, aligned panel.
    close_full = close_full.reindex(columns=full_syms)
    ret_full = build_return_panel(close_full, drop_overnight=True)

    p_close_full = os.path.join(out_dir, f"close_panel_full_{span}.parquet")
    p_ret_full = os.path.join(out_dir, f"logret_panel_full_{span}.parquet")
    close_full.to_parquet(p_close_full)
    ret_full.to_parquet(p_ret_full)

    nan_close_full = float(close_full.isna().mean().mean())
    # Return NaN excluding the structurally-NaN rows (first overall row +
    # overnight rows have no defined intraday return).
    ny_time = ret_full.index.tz_convert("America/New_York").strftime("%H:%M")
    intraday_rows = (np.asarray(ny_time) != OPEN_BAR).copy()
    intraday_rows[0] = False  # first overall row never has a prior bar
    ret_intraday = ret_full.loc[intraday_rows]
    nan_ret_intraday = float(ret_intraday.isna().mean().mean())

    print(f"    close_panel_full : shape={close_full.shape} "
          f"NaN_frac={nan_close_full:.3%} -> {p_close_full} "
          f"({_human(os.path.getsize(p_close_full))})")
    print(f"    logret_panel_full: shape={ret_full.shape} "
          f"intraday_NaN_frac={nan_ret_intraday:.3%} -> {p_ret_full} "
          f"({_human(os.path.getsize(p_ret_full))})")

    # --- 4. Liquid-200 subset panels (dense, recommended default for selection)
    print("\n[4] Building LIQUID top-200 close + log-return panels")
    liquid_in_panel = [s for s in liquid if s in close_full.columns]
    extra = [s for s in liquid if s not in close_full.columns]
    close_liq = build_price_panel(years, price_col="close").reindex(columns=liquid)
    ret_liq = build_return_panel(close_liq, drop_overnight=True)

    p_close_liq = os.path.join(out_dir, f"close_panel_liquid{top_n}_{span}.parquet")
    p_ret_liq = os.path.join(out_dir, f"logret_panel_liquid{top_n}_{span}.parquet")
    close_liq.to_parquet(p_close_liq)
    ret_liq.to_parquet(p_ret_liq)

    nan_close_liq = float(close_liq.isna().mean().mean())
    ret_liq_intraday = ret_liq.loc[intraday_rows]
    nan_ret_liq = float(ret_liq_intraday.isna().mean().mean())
    print(f"    close_panel_liquid : shape={close_liq.shape} "
          f"NaN_frac={nan_close_liq:.3%} -> {p_close_liq} "
          f"({_human(os.path.getsize(p_close_liq))})")
    print(f"    logret_panel_liquid: shape={ret_liq.shape} "
          f"intraday_NaN_frac={nan_ret_liq:.3%} -> {p_ret_liq} "
          f"({_human(os.path.getsize(p_ret_liq))})")
    print(f"    liquid syms in full-history set: {len(liquid_in_panel)}/{top_n}; "
          f"{len(extra)} liquid syms are NOT full-history (e.g. {extra[:5]})")

    # --- 5. Coverage report
    print("\n[5] Coverage report")
    cov = coverage_report(years)
    p_cov = os.path.join(out_dir, "coverage_report.csv")
    cov.to_csv(p_cov)
    print(f"    {len(cov)} symbols -> {p_cov}")
    print(f"    coverage_frac quantiles: "
          f"p10={cov['coverage_frac'].quantile(.10):.3f} "
          f"p50={cov['coverage_frac'].median():.3f} "
          f"p90={cov['coverage_frac'].quantile(.90):.3f}")
    full_hist_cov = cov.loc[cov.index.isin(full_syms), "coverage_frac"]
    print(f"    full-history syms coverage_frac: "
          f"median={full_hist_cov.median():.3f} min={full_hist_cov.min():.3f}")

    # --- 6. Summary
    print("\n[6] SUMMARY")
    print(f"    date span: {close_full.index.min()} -> {close_full.index.max()}")
    print(f"    total 30m regular bars (rows): {len(close_full):,}")
    print(f"    bars/day verified: {BARS_PER_DAY} (grid {REGULAR_TIME_GRID[0]}"
          f"..{REGULAR_TIME_GRID[-1]})")
    print(f"    full panel NaN: {nan_close_full:.2%} | "
          f"liquid panel NaN: {nan_close_liq:.2%}")
    print("    DONE.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MSE242 data panels.")
    ap.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--out-dir", type=str, default=PROCESSED_DIR)
    args = ap.parse_args()
    build_and_cache(years=args.years, top_n=args.top_n, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
