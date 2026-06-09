"""
selector_con.py — CONTRASTIVE-ENCODER pair-selection arm (the treatment arm of
the A/B). This is a drop-in ``select_pairs`` matching the EXACT backtest contract:

    select_pairs(
        train_close:  pd.DataFrame,
        train_logret: pd.DataFrame,
        train_window: tuple[str, str],
    ) -> list[signal.PairParams]

It differs from ``src/selectors.py`` (the classic control arm) in EXACTLY ONE
step — candidate generation. Everything downstream (the frozen-param fit via
``signal.fit_pair_params``, the positive-beta / half-life / spread-vol filters,
the cap-and-rank to ``max_pairs``, and the entire signal + cost + backtest) is the
SAME as classic. That isolation is what makes the headline A/B fair: any
difference in OOS performance is attributable to selection, not to the trading
rule.

Contrastive method
------------------
1. Per-window liquid universe via ``liquid_universe(date_range=train_window,
   top_n)`` — same call the classic arm relies on upstream — intersected with the
   panel handed in (which the backtest already restricted by the >=95% coverage
   NaN policy).
2. Train the contrastive encoder on TRAIN data only (``encoder_con.embed_window``)
   and embed every universe symbol into R^d.
3. **Candidate generation = embedding kNN**: for each symbol take its ``knn``
   nearest neighbors by COSINE SIMILARITY in embedding space; the union of those
   (sym, neighbor) pairs are the candidates. This replaces the classic
   correlation-prefilter + Engle-Granger cointegration test. Co-movers land near
   each other in the contrastive space, so kNN surfaces economically-plausible
   mean-reverting candidates without an explicit cointegration test.
4. Apply the SAME downstream filters as classic (steps 4-6 there):
     * fit frozen params with ``signal.fit_pair_params`` (OLS hedge ratio, spread
       mu/sigma, OU half-life — all TRAIN-frozen),
     * require positive beta,
     * half-life in [hl_min, hl_max] bars,
     * spread std >= min_spread_vol,
     * cap to ``max_pairs``, ranked by embedding cosine (closest first), with a
       half-life tie-break.

NO-LOOKAHEAD: the encoder + every fitted parameter sees TRAIN bars only (the
backtest slices TRAIN before calling us; ``encoder_con.embed_window`` re-asserts
it). The test window is never referenced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import signal as sig
from . import encoder_con as ec
from .data_panel import liquid_universe


@dataclass
class ContrastiveSelectorConfig:
    # universe
    top_n: int = 150             # per-window liquid universe size (matches classic max_universe)
    # candidate generation (the ONLY thing different from classic)
    knn: int = 8                 # nearest neighbors per symbol in embedding space
    min_cos: float = 0.0         # candidate must have embedding cosine >= this
    # SAME downstream filters as ClassicSelectorConfig
    hl_min: float = 10.0
    hl_max: float = 150.0
    min_spread_vol: float = 0.005
    max_pairs: int = 20
    require_positive_beta: bool = True
    # encoder
    encoder: ec.ContrastiveEncoderConfig | None = None


def _universe_for_window(
    train_close: pd.DataFrame, train_window: tuple[str, str], top_n: int
) -> list[str]:
    """Per-window liquid universe, intersected with the (coverage-filtered) panel.

    Uses ``liquid_universe(date_range=train_window)`` exactly like the classic arm
    relies on upstream, so both arms see the SAME tradeable universe. The years
    scanned are inferred from the train window. The panel passed in was already
    restricted by the backtest's >=95% coverage NaN policy, so we intersect to
    keep only names that are both liquid AND well-covered this window.
    """
    y0 = int(train_window[0][:4])
    y1 = int(train_window[1][:4])
    years = list(range(y0, y1 + 1))
    liq = liquid_universe(years, top_n=top_n, date_range=train_window)
    panel_syms = set(train_close.columns)
    uni = [s for s in liq if s in panel_syms]
    return uni


def _candidate_pairs_by_embedding(
    emb: pd.DataFrame, knn: int, min_cos: float
) -> list[tuple[str, str, float]]:
    """Union of each symbol's top-``knn`` cosine-nearest neighbors in embedding space.

    Returns unordered pairs (a, b, cosine) deduplicated, with cosine >= min_cos,
    sorted by descending cosine (closest co-movers first).
    """
    syms = list(emb.index)
    n = len(syms)
    if n < 2:
        return []
    E = emb.to_numpy().astype(np.float64)
    # embeddings are already L2-normalized; cosine = dot product.
    cos = E @ E.T
    np.fill_diagonal(cos, -np.inf)

    k = min(knn, n - 1)
    seen: dict[tuple[str, str], float] = {}
    for i in range(n):
        nbr = np.argpartition(-cos[i], k - 1)[:k]
        for j in nbr:
            c = float(cos[i, j])
            if not np.isfinite(c) or c < min_cos:
                continue
            a, b = syms[i], syms[int(j)]
            key = (a, b) if a < b else (b, a)
            # keep the max cosine if the pair shows up from both directions
            if key not in seen or c > seen[key]:
                seen[key] = c
    cand = [(a, b, c) for (a, b), c in seen.items()]
    cand.sort(key=lambda x: x[2], reverse=True)
    return cand


def select_pairs_contrastive(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    train_window: tuple[str, str],
    cfg: ContrastiveSelectorConfig | None = None,
) -> list[sig.PairParams]:
    """Contrastive-encoder selector. See module docstring for the contract.

    Returns a list of fully TRAIN-frozen ``signal.PairParams`` (capped at
    ``cfg.max_pairs``). Candidate generation is embedding-kNN; all downstream
    filtering/freezing is identical to the classic arm.
    """
    if cfg is None:
        cfg = ContrastiveSelectorConfig()

    # 1. per-window liquid universe (same source as classic)
    universe = _universe_for_window(train_close, train_window, cfg.top_n)
    if len(universe) < 2:
        return []

    # 2. train encoder on TRAIN-only data + embed (no-lookahead asserted inside)
    emb = ec.embed_window(train_close, train_logret, train_window, universe,
                          cfg.encoder or ec.ContrastiveEncoderConfig())
    if emb.shape[0] < 2:
        return []

    # 3. candidate pairs by embedding kNN (THE only difference from classic)
    candidates = _candidate_pairs_by_embedding(emb, cfg.knn, cfg.min_cos)

    # 4. SAME downstream filters + frozen-param fit as classic
    scored: list[tuple[float, float, sig.PairParams]] = []
    for a, b, cos in candidates:
        params = sig.fit_pair_params(train_close, a, b)
        if params is None:
            continue
        if cfg.require_positive_beta and params.beta <= 0:
            continue
        hl = params.half_life
        if not np.isfinite(hl) or hl < cfg.hl_min or hl > cfg.hl_max:
            continue
        if not np.isfinite(params.sigma) or params.sigma < cfg.min_spread_vol:
            continue
        params.extra = {"emb_cosine": float(cos)}
        # rank: closest co-movers first (highest cosine), tie-break tighter half-life
        scored.append((-cos, hl, params))

    scored.sort(key=lambda x: (x[0], x[1]))
    selected = [p for _, _, p in scored[: cfg.max_pairs]]
    return selected


def make_contrastive_selector(cfg: ContrastiveSelectorConfig | None = None):
    """Return a selector callable matching the backtest's SelectorFn signature.

    Usage:
        selector = make_contrastive_selector(ContrastiveSelectorConfig(max_pairs=20))
        run_walk_forward(close, logret, selector, wf_cfg)
    """
    return partial(select_pairs_contrastive, cfg=cfg or ContrastiveSelectorConfig())
