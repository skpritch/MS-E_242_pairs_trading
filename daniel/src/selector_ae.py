"""
selector_ae.py — AUTOENCODER pair-selection arm (the ML arm of the A/B).

Drop-in replacement for ``selectors.select_pairs_classic`` matching the EXACT
backtest contract:

    select_pairs(train_close, train_logret, train_window) -> list[signal.PairParams]

The ONLY methodological difference from the classic arm is candidate generation:
instead of correlation pre-filter -> Engle-Granger cointegration, we
1. train a per-fold autoencoder on TRAIN returns (``encoder_ae.embed_window``),
2. embed every (train-window liquid) symbol, and
3. propose candidate pairs by SMALL EMBEDDING DISTANCE (k nearest neighbours of
   each symbol in embedding space).

Every downstream filter is IDENTICAL to the classic arm so the A/B is fair:
* hedge ratio / spread mu,sigma / OU half-life via ``signal.fit_pair_params`` (TRAIN),
* require positive beta,
* half-life in [hl_min, hl_max] bars,
* train spread std >= min_spread_vol,
* cap to ``max_pairs`` via an analogous score.

The classic arm scores by cointegration p-value (lower better, tie-break tighter
half-life). The AE arm has no p-value, so we score by EMBEDDING DISTANCE (smaller
= more similar return-shape = better candidate), tie-broken by tighter half-life —
the direct analogue ("the selector's own notion of pair quality, then half-life").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from . import signal as sig
from . import encoder_ae as enc


@dataclass
class AESelectorConfig:
    # candidate generation
    knn_k: int = 8               # nearest neighbours per symbol in embedding space
    cand_cap: int = 300          # cap on # unique candidate pairs scored (≈ classic corr_top_k)
    # downstream filters — IDENTICAL defaults to ClassicSelectorConfig
    hl_min: float = 10.0
    hl_max: float = 150.0
    min_spread_vol: float = 0.005
    max_pairs: int = 20
    require_positive_beta: bool = True
    max_universe: int = 150
    ae: enc.AEConfig = field(default_factory=enc.AEConfig)


def _knn_candidate_pairs(
    emb: pd.DataFrame, k: int, cap: int
) -> list[tuple[str, str, float]]:
    """Candidate pairs = each symbol's k nearest neighbours in embedding space.

    Returns unique unordered pairs (a<b) with their euclidean embedding distance,
    sorted ascending (closest first), capped to ``cap`` pairs. This is the AE
    analogue of the classic correlation pre-filter: a cheap O(N^2) distance scan
    that proposes the most economically-plausible (similarly-behaving) candidates
    to the expensive per-pair fitting/filters."""
    syms = list(emb.index)
    n = len(syms)
    if n < 2:
        return []
    V = emb.to_numpy(dtype=np.float64)
    # pairwise squared euclidean distances
    sq = (V * V).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (V @ V.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)

    kk = min(k, n - 1)
    seen: dict[tuple[str, str], float] = {}
    for i in range(n):
        nn_idx = np.argpartition(d2[i], kk - 1)[:kk]
        for j in nn_idx:
            a, b = syms[i], syms[int(j)]
            key = (a, b) if a < b else (b, a)
            dist = float(np.sqrt(d2[i, int(j)]))
            # keep the (symmetric) distance; both directions give the same value
            if key not in seen or dist < seen[key]:
                seen[key] = dist
    cand = [(a, b, d) for (a, b), d in seen.items()]
    cand.sort(key=lambda x: x[2])
    return cand[:cap]


def select_pairs_ae(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    train_window: tuple[str, str],
    cfg: AESelectorConfig | None = None,
) -> list[sig.PairParams]:
    """Autoencoder-embedding pair selector. See module docstring for the contract.

    Returns train-frozen ``signal.PairParams`` (capped at ``cfg.max_pairs``).
    All param fitting is on TRAIN only (via signal.fit_pair_params); the encoder
    is trained on TRAIN only (via encoder_ae.embed_window)."""
    if cfg is None:
        cfg = AESelectorConfig()

    # 1) Embed every train-window liquid symbol with the per-fold AE (TRAIN ONLY).
    emb = enc.embed_window(
        train_close, train_logret, train_window,
        universe=None, cfg=cfg.ae,
    )
    if emb is None or len(emb) < 2:
        return []

    # 2) Candidate pairs by SMALL embedding distance (the AE's analogue of corr).
    candidates = _knn_candidate_pairs(emb, cfg.knn_k, cfg.cand_cap)
    if not candidates:
        return []

    # 3) SAME downstream filters as classic; score by embedding distance.
    scored: list[tuple[float, float, sig.PairParams]] = []
    for a, b, dist in candidates:
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
        params.extra = {"emb_distance": float(dist)}
        # selection score: smaller embedding distance best, tie-break tighter HL.
        scored.append((dist, hl, params))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in scored[: cfg.max_pairs]]


def make_ae_selector(cfg: AESelectorConfig | None = None):
    """Return a selector callable matching the backtest's SelectorFn signature."""
    return partial(select_pairs_ae, cfg=cfg or AESelectorConfig())
