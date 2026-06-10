"""selector_gru.py — GRU pair selector (KNN on GRU embeddings).

Pipeline: AR lookbacks → train GRU → extract embeddings → KNN candidates →
eligibility + rank by train Sharpe → top 20.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from .config import GRUConfig, EligibilityConfig, Fold, PairParams, SignalConfig
from . import encoder_gru as gru_enc
from . import signal as sig
from .selector_ae import _knn_candidates  # reuse KNN logic
from .io_utils import save_json


def _apply_eligibility_and_rank(
    candidates: list[tuple[str, str, float]],
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
) -> list[tuple[float, PairParams]]:
    """Same filters as classic/AE (positive beta, BP, spread vol), rank by train Sharpe."""
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


def _select_gru(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    fold: Fold,
    gru_cfg: GRUConfig,
    elig_cfg: EligibilityConfig,
    signal_cfg: SignalConfig,
    results_dir: str | None,
) -> list[PairParams]:
    """GRU selector: AR lookbacks → train GRU → embed → KNN → eligibility → rank."""
    symbols = list(train_close.columns)

    # 1. Compute AR lookbacks
    lookbacks = gru_enc.compute_all_lookbacks(train_logret, symbols, gru_cfg)

    if results_dir:
        save_json({
            "fold_id": fold.fold_id,
            "lookbacks": lookbacks,
        }, f"{results_dir}/fold_{fold.fold_id:02d}_ar_lookbacks.json")

    # 2. Build GRU training data
    X, y = gru_enc.build_gru_data(train_logret, symbols, lookbacks, gru_cfg.max_lags)
    if X.shape[0] < 50:
        return []

    # 3. Train GRU (or load cached weights)
    wpath = f"{results_dir}/fold_{fold.fold_id:02d}_gru_model.pt" if results_dir else None
    model = gru_enc.train_gru(X, y, gru_cfg, seed=fold.fold_id, weights_path=wpath)

    # 4. Extract embeddings
    emb_df = gru_enc.extract_gru_embeddings(
        model, train_logret, symbols, lookbacks, gru_cfg.max_lags)
    if len(emb_df) < 2:
        return []

    if results_dir:
        save_json({
            "fold_id": fold.fold_id,
            "n_symbols": len(emb_df),
            "emb_dim": emb_df.shape[1],
            "symbols": list(emb_df.index),
            "embeddings": emb_df.to_dict(orient="index"),
        }, f"{results_dir}/fold_{fold.fold_id:02d}_embeddings.json")

    # 5. KNN candidate pairs
    candidates = _knn_candidates(emb_df, k=gru_cfg.knn_k, cap=gru_cfg.cand_cap)
    if not candidates:
        return []

    # 6. Apply eligibility + rank by train Sharpe
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


def make_gru_selector(
    gru_cfg: GRUConfig | None = None,
    elig_cfg: EligibilityConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    results_dir: str | None = None,
) -> callable:
    """Return a selector callable matching the backtest's SelectorFn signature."""
    if gru_cfg is None:
        gru_cfg = GRUConfig()
    if elig_cfg is None:
        elig_cfg = EligibilityConfig()
    if signal_cfg is None:
        signal_cfg = SignalConfig()
    return partial(
        _select_gru,
        gru_cfg=gru_cfg,
        elig_cfg=elig_cfg,
        signal_cfg=signal_cfg,
        results_dir=results_dir,
    )
