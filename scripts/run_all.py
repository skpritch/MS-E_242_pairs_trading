"""run_all.py — Orchestrator: runs all arms, saves comparison.

Usage:
    python -m scripts.run_all                          # full run, all 3 arms
    python -m scripts.run_all --arms classic            # single arm
    python -m scripts.run_all --arms classic autoencoder
    python -m scripts.run_all --start 2018 --end 2020   # short test run
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import (
    AEConfig,
    ClassicConfig,
    EligibilityConfig,
    GRUConfig,
    SignalConfig,
    WalkForwardConfig,
)
from .data_panel import build_price_panel, build_return_panel, build_dollar_volume_panel
from .backtest import run_walk_forward
from .selector_classic import make_classic_selector
from .selector_ae import make_ae_selector
from .selector_gru import make_gru_selector
from .io_utils import save_json, json_serializable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def _save_arm_results(result, arm_name: str, output_dir: Path, config_snapshot: dict):
    """Save all result JSONs for one arm."""
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    save_json(result.aggregate, arm_dir / "aggregate_metrics.json")

    # Equity curve
    eq = result.equity
    save_json({
        "timestamps": [str(t) for t in eq.index],
        "equity": eq.tolist(),
    }, arm_dir / "equity_curve.json")

    # Fold metrics
    save_json(result.fold_metrics.to_dict(orient="records"),
              arm_dir / "fold_metrics.json")

    # Trade log
    if len(result.trade_log) > 0:
        tl = result.trade_log.copy()
        for col in ["entry_idx", "exit_idx"]:
            if col in tl.columns:
                tl[col] = tl[col].astype(str)
        save_json(tl.to_dict(orient="records"), arm_dir / "trade_log.json")
    else:
        save_json([], arm_dir / "trade_log.json")

    save_json(config_snapshot, arm_dir / "config.json")


def main():
    parser = argparse.ArgumentParser(description="Pairs trading backtest orchestrator")
    parser.add_argument("--arms", nargs="+", default=["classic", "autoencoder", "gru"],
                        choices=["classic", "autoencoder", "gru"],
                        help="Which arms to run")
    parser.add_argument("--start", type=int, default=2016, help="Start year (inclusive)")
    parser.add_argument("--end", type=int, default=2025, help="End year (inclusive)")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))

    # Shared configs
    wf_cfg = WalkForwardConfig()
    signal_cfg = SignalConfig()
    elig_cfg = EligibilityConfig()

    config_snapshot = {
        "walk_forward": asdict(wf_cfg),
        "signal": asdict(signal_cfg),
        "eligibility": asdict(elig_cfg),
        "years": years,
    }

    # ── Load panels once (shared across arms) ──
    print(f"Loading price panel for years {years[0]}-{years[-1]}...")
    t0 = time.time()
    close_panel = build_price_panel(years)
    logret_panel = build_return_panel(close_panel)
    dv_panel = build_dollar_volume_panel(years)
    print(f"  Loaded in {time.time() - t0:.1f}s: {close_panel.shape}")

    comparison = {}

    # ── Classic arm ──
    if "classic" in args.arms:
        print("\n" + "=" * 60)
        print("CLASSIC ARM")
        print("=" * 60)
        classic_cfg = ClassicConfig()
        config_snapshot["classic"] = asdict(classic_cfg)
        results_dir = str(OUTPUT_DIR / "classic")

        selector = make_classic_selector(classic_cfg, elig_cfg, signal_cfg, results_dir)
        t0 = time.time()
        result = run_walk_forward(
            close_panel, logret_panel, selector,
            wf_cfg=wf_cfg, signal_cfg=signal_cfg, elig_cfg=elig_cfg,
            dv_panel=dv_panel, results_dir=results_dir, arm_name="classic")
        elapsed = time.time() - t0

        print(f"\n  Aggregate Sharpe: {result.aggregate['sharpe_annualized']:.3f}")
        print(f"  Total return:     {result.aggregate['return_total']:.2%}")
        print(f"  Max drawdown:     {result.aggregate['max_drawdown']:.2%}")
        print(f"  Elapsed:          {elapsed:.1f}s")

        _save_arm_results(result, "classic", OUTPUT_DIR, config_snapshot)
        comparison["classic"] = result.aggregate

    # ── Autoencoder arm ──
    if "autoencoder" in args.arms:
        print("\n" + "=" * 60)
        print("AUTOENCODER ARM")
        print("=" * 60)
        ae_cfg = AEConfig()
        config_snapshot["autoencoder"] = asdict(ae_cfg)
        results_dir = str(OUTPUT_DIR / "autoencoder")

        selector = make_ae_selector(ae_cfg, elig_cfg, signal_cfg, results_dir)
        t0 = time.time()
        result = run_walk_forward(
            close_panel, logret_panel, selector,
            wf_cfg=wf_cfg, signal_cfg=signal_cfg, elig_cfg=elig_cfg,
            dv_panel=dv_panel, results_dir=results_dir, arm_name="autoencoder")
        elapsed = time.time() - t0

        print(f"\n  Aggregate Sharpe: {result.aggregate['sharpe_annualized']:.3f}")
        print(f"  Total return:     {result.aggregate['return_total']:.2%}")
        print(f"  Max drawdown:     {result.aggregate['max_drawdown']:.2%}")
        print(f"  Elapsed:          {elapsed:.1f}s")

        _save_arm_results(result, "autoencoder", OUTPUT_DIR, config_snapshot)
        comparison["autoencoder"] = result.aggregate

    # ── GRU arm ──
    if "gru" in args.arms:
        print("\n" + "=" * 60)
        print("GRU ARM")
        print("=" * 60)
        gru_cfg = GRUConfig()
        config_snapshot["gru"] = asdict(gru_cfg)
        results_dir = str(OUTPUT_DIR / "gru")

        selector = make_gru_selector(gru_cfg, elig_cfg, signal_cfg, results_dir)
        t0 = time.time()
        result = run_walk_forward(
            close_panel, logret_panel, selector,
            wf_cfg=wf_cfg, signal_cfg=signal_cfg, elig_cfg=elig_cfg,
            dv_panel=dv_panel, results_dir=results_dir, arm_name="gru")
        elapsed = time.time() - t0

        print(f"\n  Aggregate Sharpe: {result.aggregate['sharpe_annualized']:.3f}")
        print(f"  Total return:     {result.aggregate['return_total']:.2%}")
        print(f"  Max drawdown:     {result.aggregate['max_drawdown']:.2%}")
        print(f"  Elapsed:          {elapsed:.1f}s")

        _save_arm_results(result, "gru", OUTPUT_DIR, config_snapshot)
        comparison["gru"] = result.aggregate

    # ── Save comparison ──
    if comparison:
        save_json(comparison, OUTPUT_DIR / "comparison" / "summary.json")
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        metrics = ["sharpe_annualized", "return_total", "return_annualized",
                    "max_drawdown", "n_trades", "trade_win_rate", "n_active_pairs"]
        header = f"{'metric':>25s}"
        for arm in comparison:
            header += f"  {arm:>14s}"
        print(header)
        for m in metrics:
            row = f"{m:>25s}"
            for arm in comparison:
                val = comparison[arm].get(m, 0)
                if isinstance(val, float):
                    row += f"  {val:>14.4f}"
                else:
                    row += f"  {val:>14}"
                row
            print(row)

    print("\nDone. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
