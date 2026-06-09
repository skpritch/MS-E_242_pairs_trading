"""
selectors.py — CLASSIC pair-selection arm (the control arm of the A/B).

A *selector* is the pluggable component the backtest injects. Its EXACT contract
(which Agent D's encoder arm must match) is:

    select_pairs(
        train_close:  pd.DataFrame,
        train_logret: pd.DataFrame,
        train_window: tuple[str, str],
    ) -> list[signal.PairParams]

All selection, hedge-ratio fitting, and parameter freezing happen on TRAIN ONLY.
The returned PairParams are consumed verbatim by ``src/backtest.py`` on TEST.

Classic method
--------------
1. Per-window universe is already applied upstream (the panel passed in is the
   train-window slice restricted to the liquid universe). We additionally:
2. **Correlation pre-filter** (the O(N^2) shortcut): compute the pairwise
   correlation of train log-returns and only run the (expensive) cointegration
   test on the top-``corr_top_k`` most-correlated candidate pairs. This avoids
   running Engle-Granger on all ~C(150,2) ≈ 11k pairs every fold while keeping
   the economically-plausible (co-moving) candidates. Documented shortcut.
3. **Engle-Granger cointegration** (statsmodels ``coint``) on train log-prices;
   keep pairs with p-value <= ``pvalue_max``.
4. **Half-life filter**: OU half-life on the train spread must lie in
   [``hl_min``, ``hl_max``] bars (reject too-fast = noise/microstructure and
   too-slow = barely mean-reverting / capital-inefficient).
5. **Spread-volatility filter**: train spread std >= ``min_spread_vol`` so the
   z-score band corresponds to a tradeable move (not rounding noise).
6. **Cap & rank**: keep the top ``max_pairs`` by a selection score
   (lower coint p-value is better; ties broken by tighter half-life). This is the
   control arm; the encoder arm replaces steps 2-3 with embedding-distance
   candidate generation but keeps 4-6 (same signal-side filters) for a fair A/B.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from . import signal as sig


@dataclass
class ClassicSelectorConfig:
    corr_top_k: int = 300        # # candidate pairs kept after correlation pre-filter
    min_corr: float = 0.5        # candidate must have train log-ret corr >= this
    pvalue_max: float = 0.05     # Engle-Granger cointegration p-value cutoff
    hl_min: float = 10.0         # min OU half-life (bars) — reject microstructure noise
    hl_max: float = 150.0        # max OU half-life (bars) — reject barely-reverting
    min_spread_vol: float = 0.005  # min train spread std (log units)
    max_pairs: int = 20          # cap # pairs traded per fold
    max_universe: int = 150      # cap symbols before pairing (by train liquidity proxy)
    require_positive_beta: bool = True  # drop spurious negative hedge ratios
                                        # (two positively co-moving names should
                                        #  have beta>0; beta<0 makes the "spread" a
                                        #  trending sum, not a mean-reverting pair)


def _candidate_pairs_by_corr(
    train_logret: pd.DataFrame, top_k: int, min_corr: float
) -> list[tuple[str, str, float]]:
    """Top-k most positively-correlated symbol pairs from train log-returns.

    Correlation pre-filter (the documented O(N^2) shortcut): we compute the full
    correlation matrix once (cheap, vectorized) and only pass the most co-moving
    pairs to the expensive cointegration test.
    """
    r = train_logret.dropna(how="all", axis=1)
    # pairwise-complete correlation; min_periods guards thin overlaps
    corr = r.corr(min_periods=50)
    cols = corr.columns
    n = len(cols)
    cand: list[tuple[str, str, float]] = []
    cm = corr.to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            c = cm[i, j]
            if np.isfinite(c) and c >= min_corr:
                cand.append((cols[i], cols[j], float(c)))
    cand.sort(key=lambda x: x[2], reverse=True)
    return cand[:top_k]


def _restrict_universe(
    train_close: pd.DataFrame, max_universe: int
) -> pd.DataFrame:
    """Cap the universe to the ``max_universe`` highest-median-price*range names.

    The panel handed in is already the liquid universe slice; this is a secondary
    cap to bound the O(N^2) cost if the caller passes a wide panel. We rank by a
    cheap in-window liquidity/variability proxy: median price * realized log-range
    (proportional to a tradeable-notional × movement). Columns with too many NaNs
    were already dropped by the backtest's NaN policy.
    """
    if train_close.shape[1] <= max_universe:
        return train_close
    logp = np.log(train_close)
    rng = (logp.max() - logp.min())
    proxy = (train_close.median() * rng).sort_values(ascending=False)
    keep = proxy.head(max_universe).index
    return train_close[keep]


def select_pairs_classic(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    train_window: tuple[str, str],
    cfg: ClassicSelectorConfig | None = None,
) -> list[sig.PairParams]:
    """Classic cointegration+distance selector. See module docstring for contract.

    Returns a list of fully train-frozen ``signal.PairParams`` (capped at
    ``cfg.max_pairs``), each with OLS hedge ratio, spread mu/sigma, and OU
    half-life estimated on TRAIN only.
    """
    if cfg is None:
        cfg = ClassicSelectorConfig()

    train_close = _restrict_universe(train_close, cfg.max_universe)
    syms = list(train_close.columns)
    train_logret = train_logret.reindex(columns=syms)

    candidates = _candidate_pairs_by_corr(train_logret, cfg.corr_top_k, cfg.min_corr)

    logp = np.log(train_close)
    scored: list[tuple[float, float, sig.PairParams]] = []

    for a, b, corr in candidates:
        la = logp[a].dropna()
        lb = logp[b].dropna()
        common = la.index.intersection(lb.index)
        if len(common) < 60:
            continue
        la, lb = la.loc[common], lb.loc[common]

        # Engle-Granger cointegration on train log-prices.
        try:
            _, pval, _ = coint(la.to_numpy(), lb.to_numpy())
        except Exception:
            continue
        if not np.isfinite(pval) or pval > cfg.pvalue_max:
            continue

        params = sig.fit_pair_params(train_close, a, b)
        if params is None:
            continue
        # economic hedge-ratio sanity: reject spurious negative betas
        if cfg.require_positive_beta and params.beta <= 0:
            continue
        # half-life filter
        hl = params.half_life
        if not np.isfinite(hl) or hl < cfg.hl_min or hl > cfg.hl_max:
            continue
        # spread-vol filter
        if not np.isfinite(params.sigma) or params.sigma < cfg.min_spread_vol:
            continue

        params.extra = {"coint_pvalue": float(pval), "train_corr": float(corr)}
        # selection score: lower p-value best, tie-break tighter half-life
        scored.append((pval, hl, params))

    scored.sort(key=lambda x: (x[0], x[1]))
    selected = [p for _, _, p in scored[: cfg.max_pairs]]
    return selected


def make_classic_selector(cfg: ClassicSelectorConfig | None = None):
    """Return a selector callable matching the backtest's SelectorFn signature.

    Usage:
        selector = make_classic_selector(ClassicSelectorConfig(max_pairs=30))
        run_walk_forward(close, logret, selector, cfg)
    """
    return partial(select_pairs_classic, cfg=cfg or ClassicSelectorConfig())
