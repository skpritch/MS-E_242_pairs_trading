"""selector_classic.py — Classic correlation + Breusch-Pagan pair selector.

Key differences from Daniel's classic arm:
- Breusch-Pagan homoskedasticity test (not Engle-Granger cointegration)
- Rank by training-period Sharpe (not p-value)
- Top 150 liquid symbols (not 200)
- Frozen z-score (not rolling)
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .config import (
    ClassicConfig,
    EligibilityConfig,
    Fold,
    PairParams,
    SignalConfig,
)
from . import signal as sig
from .universe import liquid_universe, filter_coverage
from .io_utils import save_json


def _candidate_pairs_by_corr(
    train_logret: pd.DataFrame,
    symbols: list[str],
    top_k: int = 300,
    min_corr: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Top-k most correlated pairs from train log-returns."""
    R = train_logret[symbols].dropna(how="all", axis=1)
    corr = R.corr(min_periods=50)
    cols = list(corr.columns)
    cm = corr.to_numpy()
    n = len(cols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = cm[i, j]
            if np.isfinite(c) and c >= min_corr:
                pairs.append((cols[i], cols[j], float(c)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _apply_eligibility(
    candidates: list[tuple[str, str, float]],
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
) -> list[tuple[float, PairParams, dict]]:
    """Filter candidates and rank by training-period Sharpe.

    Pipeline per candidate:
    1. Fit pair params (OLS hedge ratio, spread, half-life)
    2. Check positive beta
    3. Breusch-Pagan homoskedasticity (reject if p < bp_alpha)
    4. Check spread vol > min_spread_vol
    5. Compute training-period Sharpe (run pair_returns on train window)
    6. Rank by train Sharpe, select top max_pairs
    """
    scored = []

    for sym_a, sym_b, corr_val in candidates:
        # 1. Fit pair params
        params = sig.fit_pair_params(train_close, sym_a, sym_b)
        if params is None:
            continue

        # 2. Positive beta
        if params.beta <= 0:
            continue

        # 3. Breusch-Pagan test
        bp_result = sig.breusch_pagan_test(train_close, sym_a, sym_b)
        if bp_result is None:
            continue
        bp_stat, bp_pval = bp_result
        if bp_pval < elig_cfg.bp_alpha:
            continue  # reject heteroskedastic pairs

        # 4. Half-life filter (OU mean-reversion speed)
        hl = params.half_life
        if not np.isfinite(hl) or hl < 1 or hl > 150:
            continue

        # 5. Spread vol
        if not np.isfinite(params.sigma) or params.sigma < elig_cfg.min_spread_vol:
            continue

        # 5. Compute training-period Sharpe
        train_pr = sig.pair_returns(train_close, train_logret, params, signal_cfg)
        train_sharpe = float(pd.Series(train_pr["ret"]).dropna().pipe(
            lambda r: r.mean() / r.std(ddof=1) * np.sqrt(3024) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
        ))

        extra = {
            "train_corr": corr_val,
            "bp_stat": bp_stat,
            "bp_pval": bp_pval,
            "train_sharpe": train_sharpe,
            "half_life": params.half_life,
        }
        params.extra = extra

        scored.append((train_sharpe, params, extra))

    # 6. Rank by train Sharpe (descending), take top max_pairs
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:elig_cfg.max_pairs]


def _select_classic(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    fold: Fold,
    classic_cfg: ClassicConfig,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
    results_dir: str | None,
) -> list[PairParams]:
    """Classic selector: correlation pre-filter + BP + train Sharpe ranking."""
    symbols = list(train_close.columns)

    # Candidate pairs by correlation
    candidates = _candidate_pairs_by_corr(
        train_logret, symbols, classic_cfg.corr_top_k, classic_cfg.min_corr)

    # Apply eligibility filters + rank by train Sharpe
    scored = _apply_eligibility(candidates, train_close, train_logret, elig_cfg, signal_cfg)

    selected = [params for _, params, _ in scored]

    # Save intermediates
    if results_dir:
        save_json({
            "fold_id": fold.fold_id,
            "n_candidates": len(candidates),
            "n_selected": len(selected),
            "selected_pairs": [
                {"sym_a": p.sym_a, "sym_b": p.sym_b,
                 "beta": p.beta, "alpha": p.alpha,
                 "mu": p.mu, "sigma": p.sigma,
                 "half_life": p.half_life,
                 "train_sharpe": p.extra.get("train_sharpe", 0),
                 "train_corr": p.extra.get("train_corr", 0),
                 "bp_pval": p.extra.get("bp_pval", 0)}
                for p in selected
            ],
        }, f"{results_dir}/fold_{fold.fold_id:02d}_selected_pairs.json")

    return selected


def make_classic_selector(
    cfg: ClassicConfig | None = None,
    elig_cfg: EligibilityConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    results_dir: str | None = None,
) -> callable:
    """Return a selector callable matching the backtest's SelectorFn signature."""
    if cfg is None:
        cfg = ClassicConfig()
    if elig_cfg is None:
        elig_cfg = EligibilityConfig()
    if signal_cfg is None:
        signal_cfg = SignalConfig()
    return partial(
        _select_classic,
        classic_cfg=cfg,
        elig_cfg=elig_cfg,
        signal_cfg=signal_cfg,
        results_dir=results_dir,
    )
