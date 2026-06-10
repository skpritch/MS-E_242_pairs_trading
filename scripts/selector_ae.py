"""selector_ae.py — AE pair selector (KNN on embeddings).

Pipeline: train AE → extract embeddings → KNN candidates → eligibility + rank
by train Sharpe → top 20.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .config import AEConfig, EligibilityConfig, Fold, PairParams, SignalConfig
from . import encoder_ae as enc
from . import signal as sig
from .io_utils import save_json


def _knn_candidates(
    emb_df: pd.DataFrame, k: int = 8, cap: int = 300
) -> list[tuple[str, str, float]]:
    """Candidate pairs = each symbol's k nearest neighbours by Euclidean distance.

    Returns unique unordered pairs (a, b) with distance, sorted ascending, capped.
    """
    syms = list(emb_df.index)
    n = len(syms)
    if n < 2:
        return []

    V = emb_df.to_numpy(dtype=np.float64)
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
            if key not in seen or dist < seen[key]:
                seen[key] = dist

    cand = [(a, b, d) for (a, b), d in seen.items()]
    cand.sort(key=lambda x: x[2])
    return cand[:cap]


def _apply_eligibility_and_rank(
    candidates: list[tuple[str, str, float]],
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
) -> list[tuple[float, PairParams]]:
    """Same filters as classic (positive beta, BP, spread vol), rank by train Sharpe."""
    scored = []

    for sym_a, sym_b, emb_dist in candidates:
        params = sig.fit_pair_params(train_close, sym_a, sym_b)
        if params is None:
            continue
        if params.beta <= 0:
            continue

        bp_result = sig.breusch_pagan_test(train_close, sym_a, sym_b)
        if bp_result is None:
            continue
        bp_stat, bp_pval = bp_result
        if bp_pval < elig_cfg.bp_alpha:
            continue

        hl = params.half_life
        if not np.isfinite(hl) or hl < 1 or hl > 150:
            continue

        if not np.isfinite(params.sigma) or params.sigma < elig_cfg.min_spread_vol:
            continue

        train_pr = sig.pair_returns(train_close, train_logret, params, signal_cfg)
        r = pd.Series(train_pr["ret"]).dropna()
        train_sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(3024)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0

        params.extra = {
            "emb_distance": emb_dist,
            "bp_pval": bp_pval,
            "train_sharpe": train_sharpe,
        }
        scored.append((train_sharpe, params))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:elig_cfg.max_pairs]


def _select_ae(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    fold: Fold,
    ae_cfg: AEConfig,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
    results_dir: str | None,
) -> list[PairParams]:
    """AE selector: train AE → embed → KNN → eligibility → rank by train Sharpe."""
    symbols = list(train_close.columns)

    # Build windows and standardize
    X, symbols_kept, rows_per_sym = enc.build_daily_windows(
        train_logret, symbols, seq_len=ae_cfg.seq_len)
    if X.shape[0] < 10:
        return []

    X_std, mu, sigma = enc.standardize(X)

    # Train autoencoder (or load cached weights)
    wpath = f"{results_dir}/fold_{fold.fold_id:02d}_ae_model.pt" if results_dir else None
    model = enc.train_autoencoder(X_std, ae_cfg, seed=fold.fold_id, weights_path=wpath)

    # Extract embeddings
    emb_df = enc.extract_embeddings(model, X_std, symbols_kept, rows_per_sym)
    if len(emb_df) < 2:
        return []

    # Save embeddings
    if results_dir:
        save_json({
            "fold_id": fold.fold_id,
            "n_symbols": len(emb_df),
            "emb_dim": emb_df.shape[1],
            "symbols": list(emb_df.index),
            "embeddings": emb_df.to_dict(orient="index"),
        }, f"{results_dir}/fold_{fold.fold_id:02d}_embeddings.json")

    # KNN candidate pairs
    candidates = _knn_candidates(emb_df, k=ae_cfg.knn_k, cap=ae_cfg.cand_cap)
    if not candidates:
        return []

    # Apply eligibility + rank by train Sharpe
    scored = _apply_eligibility_and_rank(
        candidates, train_close, train_logret, elig_cfg, signal_cfg)

    selected = [params for _, params in scored]

    if results_dir:
        save_json({
            "fold_id": fold.fold_id,
            "n_candidates": len(candidates),
            "n_selected": len(selected),
            "selected_pairs": [
                {"sym_a": p.sym_a, "sym_b": p.sym_b,
                 "beta": p.beta, "train_sharpe": p.extra.get("train_sharpe", 0),
                 "emb_distance": p.extra.get("emb_distance", 0),
                 "bp_pval": p.extra.get("bp_pval", 0)}
                for p in selected
            ],
        }, f"{results_dir}/fold_{fold.fold_id:02d}_selected_pairs.json")

    return selected


def make_ae_selector(
    ae_cfg: AEConfig | None = None,
    elig_cfg: EligibilityConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    results_dir: str | None = None,
) -> callable:
    """Return a selector callable matching the backtest's SelectorFn signature."""
    if ae_cfg is None:
        ae_cfg = AEConfig()
    if elig_cfg is None:
        elig_cfg = EligibilityConfig()
    if signal_cfg is None:
        signal_cfg = SignalConfig()
    return partial(
        _select_ae,
        ae_cfg=ae_cfg,
        elig_cfg=elig_cfg,
        signal_cfg=signal_cfg,
        results_dir=results_dir,
    )
