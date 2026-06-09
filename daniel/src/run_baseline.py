"""
run_baseline.py — run the CLASSIC walk-forward pairs backtest end-to-end.

This is the CONTROL ARM of the project A/B (classic cointegration/distance pair
selection + classic z-score/OU signal + realistic costs, walk-forward, no
lookahead). It produces the headline OOS Sharpe number for the classic arm and
saves all artifacts under ``results/`` so the encoder arm can be compared on the
exact same harness.

Run (fast iteration on a short span):
    .venv/bin/python -m src.run_baseline --start 2016-06-01 --end 2018-12-31

Full 2016-2025 run:
    .venv/bin/python -m src.run_baseline

Outputs (results/):
    classic_equity_curve.csv / .parquet   stitched OOS equity + per-bar returns
    classic_trade_log.csv                  every closed trade across folds
    classic_fold_metrics.csv               per-fold metrics
    classic_metrics_summary.json           aggregate OOS metrics
    classic_equity_curve.png               equity plot (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import backtest as bt
from . import metrics as mx
from .data_panel import liquid_universe, PROCESSED_DIR, PROJECT_ROOT
from .selectors import make_classic_selector, ClassicSelectorConfig

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLOSE_PANEL = os.path.join(PROCESSED_DIR, "close_panel_full_2016_2025.parquet")
LOGRET_PANEL = os.path.join(PROCESSED_DIR, "logret_panel_full_2016_2025.parquet")


def _ny_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.tz_convert("America/New_York").normalize()).tz_localize(None)


def load_panels(start: str | None, end: str | None,
                universe_top_n: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached full panels, optionally restrict to a date span and a liquid
    universe over the WHOLE span (a cheap convenience pre-filter to bound runtime;
    the *real* per-window universe selection still happens inside each fold via
    the per-window NaN policy + correlation/cointegration filters)."""
    close = pd.read_parquet(CLOSE_PANEL)
    logret = pd.read_parquet(LOGRET_PANEL)

    if start is not None:
        day = _ny_day(close.index)
        close = close.loc[day >= pd.Timestamp(start)]
        logret = logret.loc[_ny_day(logret.index) >= pd.Timestamp(start)]
    if end is not None:
        day = _ny_day(close.index)
        close = close.loc[day <= pd.Timestamp(end)]
        logret = logret.loc[_ny_day(logret.index) <= pd.Timestamp(end)]

    if universe_top_n is not None:
        # Pre-filter columns to a liquid set over the span. We reuse the cached
        # whole-period top-200 list as a cheap proxy and intersect with the panel
        # (avoids re-scanning raw parquet); this only bounds N for runtime — the
        # economically-meaningful selection is still per-window. NOTE: this is a
        # convenience cap, not the per-window universe (which is applied in-fold).
        liq_csv = os.path.join(PROCESSED_DIR, "liquid_universe_top200.csv")
        if os.path.exists(liq_csv):
            liq = pd.read_csv(liq_csv)["symbol"].tolist()
        else:
            liq = liquid_universe(list(range(2016, 2026)), top_n=universe_top_n)
        cols = [c for c in liq if c in close.columns][:universe_top_n]
        close = close[cols]
        logret = logret[cols]

    return close, logret


def save_results(res: bt.BacktestResult, cfg: bt.WalkForwardConfig,
                 sel_cfg: ClassicSelectorConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    eq = pd.DataFrame({"oos_return": res.oos_returns,
                       "turnover": res.oos_turnover,
                       "equity": res.equity})
    eq.to_csv(os.path.join(out_dir, "classic_equity_curve.csv"))
    eq.to_parquet(os.path.join(out_dir, "classic_equity_curve.parquet"))

    res.trade_log.to_csv(os.path.join(out_dir, "classic_trade_log.csv"), index=False)
    res.fold_metrics.to_csv(os.path.join(out_dir, "classic_fold_metrics.csv"), index=False)

    summary = {
        "arm": "classic",
        "walk_forward": {
            "train_days": cfg.train_days, "test_days": cfg.test_days,
            "roll_days": cfg.roll_days, "n_folds": len(res.folds),
        },
        "signal": {
            "z_entry": cfg.z_entry, "z_exit": cfg.z_exit, "z_stop": cfg.z_stop,
            "use_rolling_z": cfg.use_rolling_z, "cost_per_side_bps": cfg.cost_per_side * 1e4,
        },
        "selector": {
            "pvalue_max": sel_cfg.pvalue_max, "hl_min": sel_cfg.hl_min,
            "hl_max": sel_cfg.hl_max, "min_spread_vol": sel_cfg.min_spread_vol,
            "max_pairs": sel_cfg.max_pairs, "corr_top_k": sel_cfg.corr_top_k,
            "min_corr": sel_cfg.min_corr, "max_universe": sel_cfg.max_universe,
        },
        "bars_per_year": cfg.bars_per_year,
        "aggregate": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in res.aggregate.items()},
    }
    with open(os.path.join(out_dir, "classic_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Optional equity plot (non-fatal if matplotlib missing).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        res.equity.plot(ax=ax[0], color="C0")
        ax[0].set_title("Classic pairs — OOS equity (net of costs), walk-forward stitched")
        ax[0].set_ylabel("equity (start=1.0)")
        ax[0].axhline(1.0, color="k", lw=0.6, ls="--")
        dd = res.equity / res.equity.cummax() - 1.0
        dd.plot(ax=ax[1], color="C3")
        ax[1].set_ylabel("drawdown")
        ax[1].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "classic_equity_curve.png"), dpi=110)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")


def _print_table(res: bt.BacktestResult) -> None:
    fm = res.fold_metrics
    cols = ["fold_id", "test_start", "test_end", "n_active_pairs",
            "sharpe_annualized", "return_total", "max_drawdown",
            "turnover_annualized", "avg_holding_bars"]
    cols = [c for c in cols if c in fm.columns]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== PER-FOLD OOS METRICS (classic arm) ===")
    print(fm[cols].to_string(index=False,
          float_format=lambda x: f"{x:,.3f}"))
    a = res.aggregate
    print("\n=== AGGREGATE OOS METRICS (stitched, net of costs) ===")
    for k in ["sharpe_annualized", "return_total", "return_annualized",
              "max_drawdown", "hit_rate_bars", "turnover_annualized",
              "n_trades", "trade_win_rate", "avg_holding_bars",
              "n_active_pairs", "n_bars", "bars_per_year"]:
        if k in a:
            print(f"  {k:22s} {a[k]:,.4f}" if isinstance(a[k], (int, float))
                  else f"  {k:22s} {a[k]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run classic pairs walk-forward backtest.")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD span start")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD span end")
    ap.add_argument("--train-days", type=int, default=252)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--roll-days", type=int, default=63)
    ap.add_argument("--z-entry", type=float, default=2.5)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=4.0)
    ap.add_argument("--cost-bps", type=float, default=5.0, help="per-side cost in bps")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--universe-top-n", type=int, default=150,
                    help="runtime cap on # symbols (per-window selection still in-fold)")
    ap.add_argument("--frozen-z", action="store_true",
                    help="use strict frozen-train z instead of causal rolling z")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    print("Loading panels...")
    close, logret = load_panels(args.start, args.end, args.universe_top_n)
    print(f"  panel: {close.shape[0]} bars x {close.shape[1]} symbols "
          f"({close.index.min()} .. {close.index.max()})")

    cfg = bt.WalkForwardConfig(
        train_days=args.train_days, test_days=args.test_days, roll_days=args.roll_days,
        z_entry=args.z_entry, z_exit=args.z_exit, z_stop=args.z_stop,
        cost_per_side=args.cost_bps * 1e-4, use_rolling_z=not args.frozen_z,
    )
    sel_cfg = ClassicSelectorConfig(max_pairs=args.max_pairs,
                                    max_universe=min(args.universe_top_n, 150))
    selector = make_classic_selector(sel_cfg)

    import time
    t0 = time.time()
    res = bt.run_walk_forward(close, logret, selector, cfg, verbose=True)
    runtime = time.time() - t0

    _print_table(res)
    print(f"\nruntime: {runtime:.1f}s over {len(res.folds)} folds "
          f"({runtime/max(len(res.folds),1):.1f}s/fold)")

    save_results(res, cfg, sel_cfg, args.out_dir)
    print(f"\nSaved artifacts to {args.out_dir}/")


if __name__ == "__main__":
    main()
