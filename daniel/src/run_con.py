"""
run_con.py — run the CONTRASTIVE-ENCODER-selection walk-forward backtest.

This is the second ML ARM of the project A/B. It plugs
``selector_con.make_contrastive_selector`` into the SHARED ``src/backtest.py`` with
the IDENTICAL WalkForwardConfig used by ``run_baseline.py`` and ``run_ae.py``
(train 252 / test 63 / roll 63, 5 bps/side, z 2.5/0.5/4.0, rolling-z), so the only
difference vs the classic arm is candidate-pair generation (embedding kNN from a
contrastively-trained encoder instead of correlation + Engle-Granger cointegration).

Run (fast iteration on a short span):
    .venv/bin/python -m src.run_con --start 2018-01-01 --end 2020-12-31

Full 2016-2025 run:
    .venv/bin/python -m src.run_con

Outputs (results/, con_ prefix):
    con_equity_curve.csv / .parquet   stitched OOS equity + per-bar returns
    con_trade_log.csv                  every closed trade across folds
    con_fold_metrics.csv               per-fold metrics
    con_metrics_summary.json           aggregate OOS metrics
    con_equity_curve.png               equity plot (if matplotlib available)
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
from .selector_con import make_contrastive_selector, ContrastiveSelectorConfig
from .encoder_con import ContrastiveEncoderConfig

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CLOSE_PANEL = os.path.join(PROCESSED_DIR, "close_panel_full_2016_2025.parquet")
LOGRET_PANEL = os.path.join(PROCESSED_DIR, "logret_panel_full_2016_2025.parquet")


def _ny_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.tz_convert("America/New_York").normalize()).tz_localize(None)


def load_panels(start: str | None, end: str | None,
                universe_top_n: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached full panels, optionally restrict to a date span and a liquid
    universe over the whole span. Identical to run_baseline.py / run_ae.py so the
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
                 sel_cfg: ContrastiveSelectorConfig, runtime: float, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    eq = pd.DataFrame({"oos_return": res.oos_returns,
                       "turnover": res.oos_turnover,
                       "equity": res.equity})
    eq.to_csv(os.path.join(out_dir, "con_equity_curve.csv"))
    eq.to_parquet(os.path.join(out_dir, "con_equity_curve.parquet"))

    res.trade_log.to_csv(os.path.join(out_dir, "con_trade_log.csv"), index=False)
    res.fold_metrics.to_csv(os.path.join(out_dir, "con_fold_metrics.csv"), index=False)

    enc = sel_cfg.encoder or ContrastiveEncoderConfig()
    summary = {
        "arm": "contrastive",
        "walk_forward": {
            "train_days": cfg.train_days, "test_days": cfg.test_days,
            "roll_days": cfg.roll_days, "n_folds": len(res.folds),
        },
        "signal": {
            "z_entry": cfg.z_entry, "z_exit": cfg.z_exit, "z_stop": cfg.z_stop,
            "use_rolling_z": cfg.use_rolling_z, "cost_per_side_bps": cfg.cost_per_side * 1e4,
        },
        "selector": {
            "method": "contrastive_embedding_knn",
            "knn": sel_cfg.knn, "min_cos": sel_cfg.min_cos, "top_n": sel_cfg.top_n,
            "hl_min": sel_cfg.hl_min, "hl_max": sel_cfg.hl_max,
            "min_spread_vol": sel_cfg.min_spread_vol, "max_pairs": sel_cfg.max_pairs,
        },
        "encoder": {
            "type": "mlp_contrastive_ntxent",
            "emb_dim": enc.emb_dim, "hidden": enc.hidden, "window_len": enc.window_len,
            "epochs": enc.epochs, "batch_size": enc.batch_size, "lr": enc.lr,
            "temperature": enc.temperature, "jitter_std": enc.jitter_std,
            "mask_frac": enc.mask_frac, "patience": enc.patience,
            "n_views_eval": enc.n_views_eval,
            "train_window_policy": "rolling 252d train window, retrained per fold",
        },
        "bars_per_year": cfg.bars_per_year,
        "runtime_seconds": float(runtime),
        "aggregate": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in res.aggregate.items()},
    }
    with open(os.path.join(out_dir, "con_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        res.equity.plot(ax=ax[0], color="C0")
        ax[0].set_title("Contrastive pairs — OOS equity (net of costs), walk-forward stitched")
        ax[0].set_ylabel("equity (start=1.0)")
        ax[0].axhline(1.0, color="k", lw=0.6, ls="--")
        dd = res.equity / res.equity.cummax() - 1.0
        dd.plot(ax=ax[1], color="C3")
        ax[1].set_ylabel("drawdown")
        ax[1].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "con_equity_curve.png"), dpi=110)
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
    print("\n=== PER-FOLD OOS METRICS (contrastive arm) ===")
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
    ap = argparse.ArgumentParser(description="Run contrastive pairs walk-forward backtest.")
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
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--knn", type=int, default=8)
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
    enc_cfg = ContrastiveEncoderConfig(emb_dim=args.emb_dim, epochs=args.epochs)
    sel_cfg = ContrastiveSelectorConfig(
        max_pairs=args.max_pairs, knn=args.knn,
        top_n=min(args.universe_top_n, 150), encoder=enc_cfg,
    )
    selector = make_contrastive_selector(sel_cfg)

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
