"""
run_ae.py — run the AUTOENCODER-selection walk-forward backtest end-to-end.

This is the ML ARM of the project A/B (autoencoder-embedding pair selection +
the SAME classic z-score/OU signal + the SAME costs/walk-forward/no-lookahead).
It plugs ``selector_ae.make_ae_selector`` into the SHARED ``src/backtest.py`` with
the identical WalkForwardConfig used by ``run_baseline.py`` (train 252 / test 63 /
roll 63, 5 bps/side, z 2.5/0.5/4.0, rolling-z), so the only difference vs the
classic arm is candidate-pair generation.

Run (fast iteration on a short span):
    .venv/bin/python -m src.run_ae --start 2018-01-01 --end 2020-12-31

Full 2016-2025 run:
    .venv/bin/python -m src.run_ae

Outputs (results/, ae_ prefix):
    ae_equity_curve.csv / .parquet   stitched OOS equity + per-bar returns
    ae_trade_log.csv                  every closed trade across folds
    ae_fold_metrics.csv               per-fold metrics
    ae_metrics_summary.json           aggregate OOS metrics
    ae_equity_curve.png               equity plot (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import backtest as bt
from .data_panel import liquid_universe, PROCESSED_DIR, PROJECT_ROOT
from .selector_ae import make_ae_selector, AESelectorConfig
from .encoder_ae import AEConfig

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLOSE_PANEL = os.path.join(PROCESSED_DIR, "close_panel_full_2016_2025.parquet")
LOGRET_PANEL = os.path.join(PROCESSED_DIR, "logret_panel_full_2016_2025.parquet")


def _ny_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.tz_convert("America/New_York").normalize()).tz_localize(None)


def load_panels(start: str | None, end: str | None,
                universe_top_n: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached full panels, optionally restrict to a date span and a liquid
    universe over the whole span (a cheap runtime cap; the real per-window
    universe selection still happens in-fold). Identical to run_baseline.py so the
    A/B starts from the same panel."""
    close = pd.read_parquet(CLOSE_PANEL)
    logret = pd.read_parquet(LOGRET_PANEL)

    if start is not None:
        close = close.loc[_ny_day(close.index) >= pd.Timestamp(start)]
        logret = logret.loc[_ny_day(logret.index) >= pd.Timestamp(start)]
    if end is not None:
        close = close.loc[_ny_day(close.index) <= pd.Timestamp(end)]
        logret = logret.loc[_ny_day(logret.index) <= pd.Timestamp(end)]

    if universe_top_n is not None:
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
                 sel_cfg: AESelectorConfig, runtime: float, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    eq = pd.DataFrame({"oos_return": res.oos_returns,
                       "turnover": res.oos_turnover,
                       "equity": res.equity})
    eq.to_csv(os.path.join(out_dir, "ae_equity_curve.csv"))
    eq.to_parquet(os.path.join(out_dir, "ae_equity_curve.parquet"))

    res.trade_log.to_csv(os.path.join(out_dir, "ae_trade_log.csv"), index=False)
    res.fold_metrics.to_csv(os.path.join(out_dir, "ae_fold_metrics.csv"), index=False)

    summary = {
        "arm": "autoencoder",
        "walk_forward": {
            "train_days": cfg.train_days, "test_days": cfg.test_days,
            "roll_days": cfg.roll_days, "n_folds": len(res.folds),
        },
        "signal": {
            "z_entry": cfg.z_entry, "z_exit": cfg.z_exit, "z_stop": cfg.z_stop,
            "use_rolling_z": cfg.use_rolling_z, "cost_per_side_bps": cfg.cost_per_side * 1e4,
        },
        "selector": {
            "method": "autoencoder_embedding_knn",
            "knn_k": sel_cfg.knn_k, "cand_cap": sel_cfg.cand_cap,
            "hl_min": sel_cfg.hl_min, "hl_max": sel_cfg.hl_max,
            "min_spread_vol": sel_cfg.min_spread_vol, "max_pairs": sel_cfg.max_pairs,
            "max_universe": sel_cfg.max_universe,
        },
        "encoder": {
            "type": "fc_autoencoder",
            "seq_len": sel_cfg.ae.seq_len, "emb_dim": sel_cfg.ae.emb_dim,
            "hidden": sel_cfg.ae.hidden, "epochs": sel_cfg.ae.epochs,
            "batch_size": sel_cfg.ae.batch_size, "lr": sel_cfg.ae.lr,
            "early_stop_patience": sel_cfg.ae.early_stop_patience,
            "train_window_policy": "rolling 252d train window, retrained per fold",
        },
        "bars_per_year": cfg.bars_per_year,
        "runtime_seconds": float(runtime),
        "aggregate": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in res.aggregate.items()},
    }
    with open(os.path.join(out_dir, "ae_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        res.equity.plot(ax=ax[0], color="C2")
        ax[0].set_title("Autoencoder pairs — OOS equity (net of costs), walk-forward stitched")
        ax[0].set_ylabel("equity (start=1.0)")
        ax[0].axhline(1.0, color="k", lw=0.6, ls="--")
        dd = res.equity / res.equity.cummax() - 1.0
        dd.plot(ax=ax[1], color="C3")
        ax[1].set_ylabel("drawdown")
        ax[1].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "ae_equity_curve.png"), dpi=110)
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
    print("\n=== PER-FOLD OOS METRICS (autoencoder arm) ===")
    print(fm[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
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
    ap = argparse.ArgumentParser(description="Run autoencoder pairs walk-forward backtest.")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--train-days", type=int, default=252)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--roll-days", type=int, default=63)
    ap.add_argument("--z-entry", type=float, default=2.5)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=4.0)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--universe-top-n", type=int, default=150)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--knn-k", type=int, default=8)
    ap.add_argument("--frozen-z", action="store_true")
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
    ae_cfg = AEConfig(emb_dim=args.emb_dim, epochs=args.epochs,
                      max_universe=min(args.universe_top_n, 150))
    sel_cfg = AESelectorConfig(max_pairs=args.max_pairs, knn_k=args.knn_k,
                               max_universe=min(args.universe_top_n, 150), ae=ae_cfg)
    selector = make_ae_selector(sel_cfg)

    t0 = time.time()
    res = bt.run_walk_forward(close, logret, selector, cfg, verbose=True)
    runtime = time.time() - t0

    _print_table(res)
    print(f"\nruntime: {runtime:.1f}s over {len(res.folds)} folds "
          f"({runtime/max(len(res.folds),1):.1f}s/fold)")

    save_results(res, cfg, sel_cfg, runtime, args.out_dir)
    print(f"\nSaved artifacts to {args.out_dir}/")


if __name__ == "__main__":
    main()
