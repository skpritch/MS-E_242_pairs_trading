"""
eda.py — exploratory data analysis + cross-arm diagnostics for the report.

Two groups of outputs, all written to results/ (and a compact JSON of numbers the
report quotes in-text):

A. DATA EDA (motivates the modeling choices)
   - universe coverage: histogram of per-symbol coverage fraction and the count of
     symbols clearing the 95% well-covered bar (data/processed/coverage_report.csv).
   - tradeable-universe size over time: active-symbol count per month from each
     symbol's [first_date, last_date] span.

B. RESULT DIAGNOSTICS (interpret the A/B)
   - per-fold OOS Sharpe time series for the three arms (regime dependence).
   - holding-period and per-trade PnL distributions per arm (trade logs).
   - cross-arm pair overlap: Jaccard of the unique traded pair sets (does the encoder
     pick DIFFERENT pairs than cointegration?), plus each arm's most-traded pairs.

Run:
    .venv/bin/python -m src.eda
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .data_panel import PROJECT_ROOT, PROCESSED_DIR

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
COVERAGE_CSV = os.path.join(PROCESSED_DIR, "coverage_report.csv")

ARMS = {
    "classic": ("Classic cointegration", "C1"),
    "ae": ("Autoencoder", "C2"),
    "con": ("Contrastive", "C0"),
}
ARM_FULL = {"classic": "classic", "ae": "autoencoder", "con": "contrastive"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def load_trade_log(arm: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{arm}_trade_log.csv"))
    df["pair"] = [f"{x}/{y}" for x, y in
                  (_pair_key(a, b) for a, b in zip(df.sym_a, df.sym_b))]
    return df


def load_fold_metrics(arm: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, f"{arm}_fold_metrics.csv"),
                       parse_dates=["test_start", "test_end"])


# ----------------------------------------------------------- A. data EDA
def coverage_eda(out_json: dict) -> None:
    cov = pd.read_csv(COVERAGE_CSV, parse_dates=["first_date", "last_date"])
    n95 = int((cov.coverage_frac >= 0.95).sum())
    out_json["universe"] = {
        "n_symbols_total": int(len(cov)),
        "n_symbols_ge_95pct_coverage": n95,
        "median_coverage_frac": float(cov.coverage_frac.median()),
        "n_symbols_full_history": int((cov.coverage_frac >= 0.99).sum()),
    }

    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(cov.coverage_frac, bins=40, color="C4", alpha=0.85)
    ax[0].axvline(0.95, color="k", ls="--", lw=0.8, label="95% threshold")
    ax[0].set_xlabel("per-symbol coverage fraction")
    ax[0].set_ylabel("# symbols")
    ax[0].set_title(f"Coverage of {len(cov)} symbols (2016–2025)")
    ax[0].legend(fontsize=8)

    # active symbols per month
    months = pd.date_range("2016-06-01", "2025-12-01", freq="MS")
    active = [int(((cov.first_date <= m) & (cov.last_date >= m)).sum()) for m in months]
    ax[1].plot(months, active, color="C4")
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("# active symbols")
    ax[1].set_title("Tradeable universe size over time")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_universe.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------- B. result diagnostics
def fold_sharpe_timeseries(out_json: dict) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    summ = {}
    for arm, (label, color) in ARMS.items():
        fm = load_fold_metrics(arm)
        ax.plot(fm.test_start, fm.sharpe_annualized, "-o", ms=3, label=label, color=color)
        summ[ARM_FULL[arm]] = {
            "pct_folds_positive": float((fm.sharpe_annualized > 0).mean()),
            "median_fold_sharpe": float(fm.sharpe_annualized.median()),
            "worst_fold_sharpe": float(fm.sharpe_annualized.min()),
            "best_fold_sharpe": float(fm.sharpe_annualized.max()),
        }
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("fold test-window start")
    ax.set_ylabel("annualized OOS Sharpe (per fold)")
    ax.set_title("Per-fold OOS Sharpe over time — regime dependence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_fold_sharpe.png"), dpi=120)
    plt.close(fig)
    out_json["fold_sharpe"] = summ


def trade_distributions(out_json: dict) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    stats = {}
    for arm, (label, color) in ARMS.items():
        tl = load_trade_log(arm)
        ax[0].hist(tl.holding_bars, bins=40, range=(0, 200), histtype="step",
                   lw=1.6, label=label, color=color)
        ax[1].hist(100 * tl.pnl, bins=60, range=(-3, 3), histtype="step",
                   lw=1.6, label=label, color=color)
        stats[ARM_FULL[arm]] = {
            "n_trades": int(len(tl)),
            "median_holding_bars": float(tl.holding_bars.median()),
            "win_rate": float((tl.pnl > 0).mean()),
            "median_trade_pnl_bps": float(10000 * tl.pnl.median()),
        }
    ax[0].set_xlabel("holding period (bars)")
    ax[0].set_ylabel("# trades")
    ax[0].set_title("Holding-period distribution")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("per-trade PnL (%)")
    ax[1].set_ylabel("# trades")
    ax[1].set_title("Per-trade PnL distribution")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eda_trade_dists.png"), dpi=120)
    plt.close(fig)
    out_json["trades"] = stats


def pair_overlap(out_json: dict) -> None:
    pair_sets = {}
    top_pairs = {}
    for arm in ARMS:
        tl = load_trade_log(arm)
        pair_sets[arm] = set(tl.pair.unique())
        top_pairs[ARM_FULL[arm]] = (
            tl.pair.value_counts().head(10).to_dict()
        )

    def jacc(a: str, b: str) -> float:
        sa, sb = pair_sets[a], pair_sets[b]
        return float(len(sa & sb) / max(len(sa | sb), 1))

    out_json["pair_overlap"] = {
        "n_unique_pairs": {ARM_FULL[a]: len(pair_sets[a]) for a in ARMS},
        "jaccard_contrastive_classic": jacc("con", "classic"),
        "jaccard_contrastive_ae": jacc("con", "ae"),
        "jaccard_classic_ae": jacc("classic", "ae"),
        "n_shared_con_classic": int(len(pair_sets["con"] & pair_sets["classic"])),
        "n_con_only_vs_classic": int(len(pair_sets["con"] - pair_sets["classic"])),
        "top_pairs": top_pairs,
    }


def main() -> None:
    out: dict = {}
    coverage_eda(out)
    fold_sharpe_timeseries(out)
    trade_distributions(out)
    pair_overlap(out)

    with open(os.path.join(RESULTS_DIR, "eda_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== UNIVERSE ===")
    print(json.dumps(out["universe"], indent=2))
    print("\n=== PER-FOLD SHARPE (fraction of folds positive) ===")
    for k, v in out["fold_sharpe"].items():
        print(f"  {k:12s} {v['pct_folds_positive']*100:4.0f}% positive  "
              f"median={v['median_fold_sharpe']:+.3f}  "
              f"worst={v['worst_fold_sharpe']:+.2f} best={v['best_fold_sharpe']:+.2f}")
    print("\n=== PAIR OVERLAP ===")
    po = out["pair_overlap"]
    print(f"  unique pairs: {po['n_unique_pairs']}")
    print(f"  Jaccard(contrastive, classic) = {po['jaccard_contrastive_classic']:.3f}")
    print(f"  shared con∩classic = {po['n_shared_con_classic']}, "
          f"con-only = {po['n_con_only_vs_classic']}")
    print(f"  contrastive top pairs: {list(po['top_pairs']['contrastive'].keys())[:6]}")
    print(f"  classic    top pairs: {list(po['top_pairs']['classic'].keys())[:6]}")
    print(f"\nSaved EDA artifacts (eda_*.png, eda_summary.json) to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
