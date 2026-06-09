"""
robustness.py — post-hoc robustness analyses on the three saved arm results.

Reads the stitched per-bar OOS equity curves (results/{classic,ae,con}_equity_curve.csv)
and produces, WITHOUT re-running any backtest:

1. COST SENSITIVITY. Each arm's per-bar net return is
       net = gross - turnover * cost_per_side,   with positions independent of cost
   (entries are z-threshold based, charged after positions are set). The saved
   curves are net at 5 bps/side with the per-bar turnover stored alongside, so the
   gross return is recovered exactly as  gross = net + turnover * 0.0005  and the
   net return at ANY cost c is  net_c = gross - turnover * c. We sweep c over a grid
   and report annualized Sharpe + annualized return per arm. This is identical to
   re-running each arm at each cost level (the backtester is not cost-aware).

2. BOOTSTRAP on the Sharpe GAP. Circular block bootstrap on the aligned per-bar net
   return series gives a sampling distribution for (i) each arm's annualized Sharpe
   and (ii) the contrastive-minus-classic Sharpe difference, yielding 95% CIs and a
   one-sided bootstrap p-value that the gap is > 0. Block length defaults to 50 bars
   (~4 trading days) to respect intraday autocorrelation and the ~56-bar mean hold.

Outputs (results/):
    robustness_cost_sensitivity.csv     arm x cost grid -> sharpe, ann_return
    robustness_cost_sensitivity.png
    robustness_bootstrap.json           per-arm Sharpe CIs + gap CI + p-value
    robustness_bootstrap.png            histogram of the bootstrapped Sharpe gap

Run:
    .venv/bin/python -m src.robustness
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import metrics as mx
from .data_panel import PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FITTED_COST = 0.0005  # 5 bps/side, the cost the saved curves were generated at

ARMS = {
    "classic": ("Classic cointegration", "classic_equity_curve.csv"),
    "autoencoder": ("Autoencoder", "ae_equity_curve.csv"),
    "contrastive": ("Contrastive", "con_equity_curve.csv"),
}

COST_GRID_BPS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]


def load_arm(fname: str) -> pd.DataFrame:
    """Load a saved equity curve with its per-bar net return + turnover, indexed by time."""
    path = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(path)
    tcol = df.columns[0]  # timestamp index column
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol)
    df["gross_return"] = df["oos_return"] + df["turnover"] * FITTED_COST
    return df[["oos_return", "turnover", "gross_return"]]


def net_at_cost(df: pd.DataFrame, cost_per_side: float) -> pd.Series:
    """Per-bar net return at an arbitrary per-side cost (analytic recompute)."""
    return df["gross_return"] - df["turnover"] * cost_per_side


# ---------------------------------------------------------------- cost sensitivity
def cost_sensitivity(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in arms.items():
        label = ARMS[key][0]
        for bps in COST_GRID_BPS:
            r = net_at_cost(df, bps * 1e-4)
            rows.append({
                "arm": key, "label": label, "cost_bps": bps,
                "sharpe_annualized": mx.annualized_sharpe(r),
                "return_annualized": mx.annualized_return(r),
                "return_total": mx.total_return(r),
            })
    return pd.DataFrame(rows)


def plot_cost_sensitivity(tbl: pd.DataFrame, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"classic": "C1", "autoencoder": "C2", "contrastive": "C0"}
    for key in ARMS:
        sub = tbl[tbl.arm == key].sort_values("cost_bps")
        ax[0].plot(sub.cost_bps, sub.sharpe_annualized, "-o",
                   label=ARMS[key][0], color=colors[key])
        ax[1].plot(sub.cost_bps, 100 * sub.return_annualized, "-o",
                   label=ARMS[key][0], color=colors[key])
    for a, ylab, ttl in [
        (ax[0], "Annualized Sharpe", "OOS Sharpe vs trading cost"),
        (ax[1], "Annualized return (%)", "OOS annualized return vs trading cost"),
    ]:
        a.axvline(5.0, color="k", lw=0.6, ls="--", alpha=0.6)
        a.axhline(0.0, color="k", lw=0.6, ls=":")
        a.set_xlabel("cost per side (bps)")
        a.set_ylabel(ylab)
        a.set_title(ttl)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- bootstrap
def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one circular-block-bootstrap resample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)[:n]


def bootstrap_sharpe(arms: dict[str, pd.DataFrame], cost_per_side: float = FITTED_COST,
                     block: int = 50, n_boot: int = 5000, seed: int = 0) -> dict:
    """Circular block bootstrap on aligned per-bar net returns.

    Returns per-arm Sharpe point + 95% CI, and the contrastive-minus-classic gap
    point + 95% CI + one-sided bootstrap p-value (P[gap <= 0])."""
    # align all arms on common timestamps
    nets = {k: net_at_cost(df, cost_per_side).rename(k) for k, df in arms.items()}
    panel = pd.concat(nets.values(), axis=1).dropna()
    n = len(panel)
    ann = np.sqrt(mx.BARS_PER_YEAR)
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * ann) if sd > 0 else 0.0

    arr = {k: panel[k].to_numpy() for k in panel.columns}
    point = {k: sharpe(arr[k]) for k in arr}
    gap_point = point["contrastive"] - point["classic"]

    boot_sharpe = {k: np.empty(n_boot) for k in arr}
    boot_gap = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_block_indices(n, block, rng)
        for k in arr:
            boot_sharpe[k][b] = sharpe(arr[k][idx])
        boot_gap[b] = boot_sharpe["contrastive"][b] - boot_sharpe["classic"][b]

    def ci(v: np.ndarray) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {
        "n_bars": int(n), "block_len": block, "n_boot": n_boot,
        "cost_per_side_bps": cost_per_side * 1e4,
        "per_arm": {k: {"sharpe": point[k], "ci95": ci(boot_sharpe[k])} for k in arr},
        "gap_contrastive_minus_classic": {
            "sharpe_gap": gap_point,
            "ci95": ci(boot_gap),
            "p_value_one_sided_gap_le_0": float(np.mean(boot_gap <= 0.0)),
        },
        "_boot_gap": boot_gap,  # popped before json dump, used for the plot
    }
    return out


def plot_bootstrap(boot: dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")
        return
    g = boot["_boot_gap"]
    gp = boot["gap_contrastive_minus_classic"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(g, bins=60, color="C0", alpha=0.8)
    ax.axvline(0.0, color="k", lw=1.0, ls="--", label="no difference")
    ax.axvline(gp["sharpe_gap"], color="C3", lw=1.5,
               label=f"point gap = {gp['sharpe_gap']:.3f}")
    lo, hi = gp["ci95"]
    ax.axvspan(lo, hi, color="C0", alpha=0.15, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("bootstrapped Sharpe gap (contrastive − classic)")
    ax.set_ylabel("frequency")
    ax.set_title(f"Block bootstrap of OOS Sharpe gap "
                 f"(p[gap≤0] = {gp['p_value_one_sided_gap_le_0']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    arms = {k: load_arm(v[1]) for k, v in ARMS.items()}
    aligned_n = pd.concat([df["oos_return"].rename(k) for k, df in arms.items()],
                          axis=1).dropna().shape[0]
    print(f"Loaded {len(arms)} arms; {aligned_n} aligned OOS bars.")

    # 1. cost sensitivity
    tbl = cost_sensitivity(arms)
    tbl.to_csv(os.path.join(RESULTS_DIR, "robustness_cost_sensitivity.csv"), index=False)
    plot_cost_sensitivity(tbl, os.path.join(RESULTS_DIR, "robustness_cost_sensitivity.png"))
    print("\n=== COST SENSITIVITY (annualized Sharpe) ===")
    piv = tbl.pivot(index="cost_bps", columns="label", values="sharpe_annualized")
    print(piv.to_string(float_format=lambda x: f"{x:6.3f}"))

    # 2. bootstrap
    boot = bootstrap_sharpe(arms)
    print("\n=== BOOTSTRAP (95% CI on annualized Sharpe, block=50, 5000 resamples) ===")
    for k, v in boot["per_arm"].items():
        print(f"  {k:12s} sharpe={v['sharpe']:+.3f}  CI95=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    gp = boot["gap_contrastive_minus_classic"]
    print(f"  GAP (con-classic) = {gp['sharpe_gap']:+.3f}  "
          f"CI95=[{gp['ci95'][0]:+.3f}, {gp['ci95'][1]:+.3f}]  "
          f"p(gap<=0)={gp['p_value_one_sided_gap_le_0']:.3f}")

    plot_bootstrap(boot, os.path.join(RESULTS_DIR, "robustness_bootstrap.png"))
    boot.pop("_boot_gap", None)
    with open(os.path.join(RESULTS_DIR, "robustness_bootstrap.json"), "w") as f:
        json.dump(boot, f, indent=2)
    print(f"\nSaved robustness artifacts to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
