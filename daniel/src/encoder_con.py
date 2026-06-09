"""
encoder_con.py — CONTRASTIVE per-symbol encoder for the encoder-selection arm
(Agent C2). This is the *novel* component of the contrastive A/B arm: it learns,
PER WALK-FORWARD FOLD ON TRAIN DATA ONLY, an embedding of each symbol's 30-minute
log-return series such that symbols that genuinely co-move land near each other in
embedding space. Candidate pairs are then drawn by small embedding distance (see
``src/selector_con.py``); the downstream signal + filters + backtest are byte-for-
byte the classic ones, so the ONLY thing that differs from the classic arm is
candidate generation. That isolation is what keeps the headline A/B fair.

Architecture
------------
A small MLP encoder over fixed-length, TIME-ALIGNED windows of each symbol's
standardized intraday log-return series:

    input  : (B, L)  a length-L block of bars (same calendar window for the batch)
    fc1    : Linear(L -> 256) + ReLU
    fc2    : Linear(256 -> 256) + ReLU
    proj   : Linear(256 -> emb_dim) ; embeddings are L2-normalized

Why an MLP and not a CNN: we deliberately do NOT pool/convolve away the temporal
positions, because for *pairs* the load-bearing signal is contemporaneous
co-movement — two symbols whose standardized returns line up bar-for-bar over a
shared window must map to nearby embeddings. A position-preserving MLP makes that
near-identity easy to represent; an empirical bake-off showed the MLP both trains
~15x faster on CPU (~1-1.5 s/fold vs ~19 s) AND recovers the return-correlation
neighbor structure markedly better (Spearman(cos, corr) ~0.5 vs ~0.4, top-5
neighbor hit-rate ~45-55% vs ~20%). ~110k params — still tiny.

Self-supervised contrastive objective (NT-Xent / InfoNCE)
---------------------------------------------------------
* A "sample" is one symbol. Per batch we pick ONE shared random time-crop offset
  and slice the SAME length-L block of TIME-ALIGNED bars for every symbol, so the
  encoder always sees a common market period. For each symbol we then build TWO
  augmented views of that same block (jitter + random temporal masking + a small
  shared sub-jitter of the crop). The two views of the SAME symbol form a POSITIVE
  pair; views of all OTHER symbols in the batch are NEGATIVES. This is SimCLR /
  NT-Xent with a temperature.
* The time-synchronized crop is the key design choice for *pairs*: because every
  symbol in a batch is encoded from the SAME calendar window, two symbols whose
  returns co-move over that window are pushed to similar embeddings (their views
  are hard negatives that the loss can only separate by leaning on idiosyncratic
  residual, leaving the common-factor part shared). Empirically this makes cosine
  similarity in embedding space track contemporaneous return correlation — exactly
  the co-movement structure pairs trading exploits. (Un-synchronized crops instead
  learn a symbol's *vol/dynamics fingerprint*, which does NOT recover co-movement;
  we verified that and switched to synchronized crops.) The emergent geometry is
  validated by the sanity check in ``__main__`` / the selector.

NO-LOOKAHEAD RAIL (hard requirement)
------------------------------------
* ``embed_window`` is handed the TRAIN-window panels only (the backtest slices
  TRAIN before calling the selector, which calls us). We never receive, load, or
  reference any test-window data.
* All standardization constants (per-symbol mean/std of returns) are computed on
  the TRAIN window only.
* The network is re-initialized and trained from scratch on each fold's TRAIN
  window; no state carries across folds (so no future fold can leak backward).
* ``assert_no_lookahead`` re-checks that every timestamp handed to the encoder is
  <= the train-window end date.

Determinism: a fixed seed (derived from the fold's train-end date) makes each
fold reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class ContrastiveEncoderConfig:
    emb_dim: int = 32            # embedding dimensionality (8-32 range)
    hidden: int = 256           # MLP hidden width
    window_len: int = 128        # length of a TIME-ALIGNED return window (~11 trading days)
    epochs: int = 30             # max epochs (early-stop usually halts sooner)
    batch_size: int = 128        # symbols per batch (each contributes 2 views)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.15    # NT-Xent temperature
    jitter_std: float = 0.5      # gaussian jitter (in std units) on standardized returns
    mask_frac: float = 0.2       # fraction of timesteps randomly zero-masked
    min_obs: int = 120           # min non-NaN returns required to embed a symbol
    patience: int = 8            # early-stop patience (epochs w/o loss improvement)
    n_views_eval: int = 30       # # shared time-crops averaged to form the embedding
    eval_stride_frac: float = 0.25  # stride (fraction of window_len) between eval crops
    seed: int = 0


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class _MLPEncoder(nn.Module):
    """Tiny MLP mapping a length-L (time-aligned) return window to an
    L2-normalized embedding. Position-preserving so contemporaneous co-movement
    maps to embedding proximity (see module docstring)."""

    def __init__(self, window_len: int, emb_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) -> (B, emb_dim), unit-norm
        return F.normalize(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
# Augmentations (build two positive views per symbol from its TRAIN series)
# --------------------------------------------------------------------------- #


def _augment_batch(
    block: np.ndarray, cfg: ContrastiveEncoderConfig, rng: np.random.Generator
) -> np.ndarray:
    """Augment a (B, L) block of TIME-ALIGNED standardized return windows.

    The same calendar window has already been cropped for every symbol (the crop
    is shared across the batch, see ``_train_encoder``). Here we add per-element
    stochasticity that PRESERVES which symbol each row is while breaking exact
    memorization:
      1. Additive Gaussian jitter (per element).
      2. Random temporal masking (zero a fraction of timesteps per row).
    """
    B, L = block.shape
    out = block.astype(np.float32).copy()
    out = out + rng.normal(0.0, cfg.jitter_std, size=(B, L)).astype(np.float32)
    if cfg.mask_frac > 0:
        m = rng.random((B, L)) < cfg.mask_frac
        out[m] = 0.0
    return out


def _prepare_matrix(
    train_logret: pd.DataFrame, universe: list[str], cfg: ContrastiveEncoderConfig
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """TIME-ALIGNED standardized return matrix (symbols x time) for the TRAIN window.

    Returns (syms, M, mask) where:
      * M is (n_syms, T) standardized log returns on a SHARED bar grid (intraday
        bars only; overnight rows dropped). Missing cells are filled with 0 (a
        neutral standardized return) and flagged in ``mask``.
      * mask is (n_syms, T) bool, True where the cell was observed (non-NaN).
    Each symbol is standardized by its OWN train mean/std over observed bars (no
    cross-symbol and no test leakage). A symbol is kept only if it has >= min_obs
    observed bars in the window.
    """
    # restrict to universe columns present in the panel
    cols = [s for s in universe if s in train_logret.columns]
    sub = train_logret[cols]
    # keep only intraday bars (rows with at least one finite return across syms);
    # overnight rows are all-NaN by construction (DATA_README) -> dropped.
    row_ok = sub.notna().any(axis=1).to_numpy()
    sub = sub.loc[row_ok]
    R = sub.to_numpy(dtype=np.float64)           # (T, n) time x symbols
    obs = np.isfinite(R)

    syms: list[str] = []
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for j, s in enumerate(cols):
        col = R[:, j]
        ok = obs[:, j]
        if ok.sum() < cfg.min_obs:
            continue
        mu = col[ok].mean()
        sd = col[ok].std()
        if not np.isfinite(sd) or sd <= 0:
            continue
        std_col = np.zeros_like(col, dtype=np.float32)
        std_col[ok] = ((col[ok] - mu) / sd).astype(np.float32)
        syms.append(s)
        rows.append(std_col)
        masks.append(ok)

    if not syms:
        return [], np.zeros((0, 0), np.float32), np.zeros((0, 0), bool)
    M = np.vstack(rows).astype(np.float32)       # (n_syms, T)
    mask = np.vstack(masks)                       # (n_syms, T)
    return syms, M, mask


# --------------------------------------------------------------------------- #
# No-lookahead guard
# --------------------------------------------------------------------------- #


def assert_no_lookahead(panel: pd.DataFrame, train_window: tuple[str, str]) -> None:
    """Hard-assert every bar handed to the encoder is within the TRAIN window.

    The encoder must NEVER see a timestamp at/after the test window. The backtest
    already slices TRAIN before calling the selector; this is a defensive double
    check so the no-lookahead claim is verifiable from this module alone.
    """
    if panel.shape[0] == 0:
        return
    ny = panel.index.tz_convert("America/New_York")
    last_day = pd.Timestamp(ny.normalize().max()).tz_localize(None).normalize()
    train_end = pd.Timestamp(train_window[1]).normalize()
    assert last_day <= train_end, (
        f"LOOKAHEAD in encoder: panel last bar {last_day} > train_end {train_end}")


# --------------------------------------------------------------------------- #
# Training + embedding
# --------------------------------------------------------------------------- #


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent (InfoNCE) loss for two batches of paired, L2-normalized embeddings.

    z1[i], z2[i] are the two views of symbol i (positives). Everything else in the
    2B batch is a negative. Standard SimCLR formulation.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # (2B, d)
    sim = z @ z.t() / temperature                # (2B, 2B) cosine sims (z is unit-norm)
    # mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))
    # positive index: i <-> i+B
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def _crop_offsets(T: int, L: int, rng: np.random.Generator) -> int:
    """Random start offset for a shared length-L time crop within T bars."""
    if T <= L:
        return 0
    return int(rng.integers(0, T - L + 1))


def _train_encoder(
    M: np.ndarray, cfg: ContrastiveEncoderConfig, seed: int
) -> _MLPEncoder:
    """Train the contrastive encoder on the TIME-ALIGNED TRAIN matrix M (n_syms, T).

    Each step picks ONE shared time-crop offset (so the whole batch sees the same
    calendar window), then builds two augmented views per symbol. Symbols that
    co-move within that window are pushed together by NT-Xent.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n, T = M.shape
    L = min(cfg.window_len, T)
    model = _MLPEncoder(L, cfg.emb_dim, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    batch = min(cfg.batch_size, n)
    if batch < 2 or n < 2:
        # Degenerate: not enough symbols to contrast. Return the random-init net;
        # the selector will fall back gracefully (few/no candidate pairs).
        return model

    # crops per epoch so every epoch sees several calendar windows
    crops_per_epoch = max(1, (T // L))

    best = float("inf")
    bad = 0
    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for _ in range(crops_per_epoch):
            off = _crop_offsets(T, L, rng)
            block_all = M[:, off:off + L]               # (n, L) shared window
            order = rng.permutation(n)
            for b0 in range(0, n, batch):
                idx = order[b0:b0 + batch]
                if len(idx) < 2:
                    continue
                blk = block_all[idx]                    # (b, L)
                v1 = _augment_batch(blk, cfg, rng)
                v2 = _augment_batch(blk, cfg, rng)
                z1 = model(torch.from_numpy(v1))
                z2 = model(torch.from_numpy(v2))
                loss = _nt_xent(z1, z2, cfg.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
        if n_batches == 0:
            break
        epoch_loss /= n_batches
        if epoch_loss < best - 1e-4:
            best = epoch_loss
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    return model


@torch.no_grad()
def _embed(
    model: _MLPEncoder, M: np.ndarray, cfg: ContrastiveEncoderConfig, seed: int
) -> np.ndarray:
    """Embed each symbol by averaging embeddings over many shared time-crops.

    We sweep STRIDED, evenly-spaced windows across the whole TRAIN period (no
    jitter/mask at eval) and encode ALL symbols on each shared window, then
    average. Tiling the full period makes the per-symbol embedding reflect the
    symbol's co-movement over the entire train window, not one lucky crop.
    Returned vectors are L2-normalized so cosine distance in the selector is
    well-defined.
    """
    model.eval()
    n, T = M.shape
    L = min(cfg.window_len, T)
    stride = max(1, int(L * cfg.eval_stride_frac))
    offs = list(range(0, max(1, T - L + 1), stride))[: cfg.n_views_eval]
    if not offs:
        offs = [0]
    out = np.zeros((n, cfg.emb_dim), dtype=np.float32)
    for off in offs:
        block = M[:, off:off + L].astype(np.float32)
        out += model(torch.from_numpy(block)).cpu().numpy()
    out /= len(offs)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed_window(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    train_window: tuple[str, str],
    universe: list[str],
    cfg: ContrastiveEncoderConfig | None = None,
) -> pd.DataFrame:
    """Embed each symbol in ``universe`` from its TRAIN-window return series.

    Parameters
    ----------
    train_close, train_logret : TRAIN-window panels (bars x symbols). The encoder
        uses ``train_logret`` (standardized log returns); ``train_close`` is
        accepted for interface symmetry / future use.
    train_window : ('YYYY-MM-DD','YYYY-MM-DD') inclusive NY train dates.
    universe : symbols to embed (the per-window liquid universe from the selector).
    cfg : ContrastiveEncoderConfig (defaults are fast + small).

    Returns
    -------
    pd.DataFrame indexed by symbol, columns = embedding dims (emb_dim columns).
    Only symbols with enough TRAIN observations appear (the rest are un-embeddable
    and thus un-pairable, which is the correct conservative behavior).
    """
    if cfg is None:
        cfg = ContrastiveEncoderConfig()

    # --- NO-LOOKAHEAD double check: only TRAIN bars may reach the encoder. ---
    assert_no_lookahead(train_logret, train_window)

    # deterministic per-fold seed from the train-end date (stable across runs)
    seed = cfg.seed + (int(pd.Timestamp(train_window[1]).strftime("%Y%m%d")) % 100000)

    syms, M, _mask = _prepare_matrix(train_logret, universe, cfg)
    if len(syms) == 0:
        return pd.DataFrame(columns=[f"e{i}" for i in range(cfg.emb_dim)])

    model = _train_encoder(M, cfg, seed)
    emb = _embed(model, M, cfg, seed)

    return pd.DataFrame(emb, index=pd.Index(syms, name="symbol"),
                        columns=[f"e{i}" for i in range(cfg.emb_dim)])


# --------------------------------------------------------------------------- #
# Standalone sanity check (run directly): does the embedding cluster co-movers?
# --------------------------------------------------------------------------- #


def _sanity_check() -> None:
    """Train on one TRAIN window and verify the embedding separates co-movers.

    We check that for each symbol, the nearest neighbor in EMBEDDING space tends
    to be the symbol it is most correlated with in RETURN space — i.e. the encoder
    recovers the co-movement structure it was never explicitly given.
    """
    import os
    from .data_panel import PROCESSED_DIR, liquid_universe

    close = pd.read_parquet(os.path.join(PROCESSED_DIR, "close_panel_full_2016_2025.parquet"))
    logret = pd.read_parquet(os.path.join(PROCESSED_DIR, "logret_panel_full_2016_2025.parquet"))

    tw = ("2018-01-01", "2018-12-31")
    ny = pd.DatetimeIndex(close.index.tz_convert("America/New_York").normalize()).tz_localize(None)
    mask = (ny >= pd.Timestamp(tw[0])) & (ny <= pd.Timestamp(tw[1]))
    tr_close = close.loc[mask]
    tr_logret = logret.loc[mask]

    uni = liquid_universe([2018], top_n=150, date_range=tw)
    uni = [s for s in uni if s in tr_close.columns]

    cfg = ContrastiveEncoderConfig()
    import time
    t0 = time.time()
    emb = embed_window(tr_close, tr_logret, tw, uni, cfg)
    dt = time.time() - t0
    print(f"[sanity] embedded {emb.shape[0]} symbols dim={emb.shape[1]} in {dt:.2f}s")

    syms = list(emb.index)
    E = emb.to_numpy()
    # cosine sim in embedding space
    cos = E @ E.T
    np.fill_diagonal(cos, -np.inf)
    nn_emb = {syms[i]: syms[int(np.argmax(cos[i]))] for i in range(len(syms))}

    # return-space correlation NN
    R = tr_logret[syms]
    corr = R.corr(min_periods=50).to_numpy().copy()
    np.fill_diagonal(corr, -np.inf)
    nn_corr = {syms[i]: syms[int(np.nanargmax(corr[i]))] for i in range(len(syms))}

    # agreement: embedding-NN == correlation-NN
    agree = np.mean([nn_emb[s] == nn_corr[s] for s in syms])
    # also: is embedding-NN among each symbol's top-5 correlated?
    top5 = {}
    for i, s in enumerate(syms):
        order = np.argsort(-np.nan_to_num(corr[i], nan=-np.inf))[:5]
        top5[s] = {syms[j] for j in order}
    in_top5 = np.mean([nn_emb[s] in top5[s] for s in syms])

    # rank correlation between embedding cosine and return correlation over all pairs
    try:
        from scipy.stats import spearmanr
        iu = np.triu_indices(len(syms), 1)
        cos_full = E @ E.T
        rho = spearmanr(cos_full[iu], np.nan_to_num(corr[iu], nan=0.0))[0]
        print(f"[sanity] Spearman(embedding-cosine, return-corr) over all pairs: {rho:.3f}")
    except Exception as e:
        print(f"[sanity] spearman skipped ({e})")

    print(f"[sanity] embedding-NN == correlation-NN exact agreement: {agree:.1%}")
    print(f"[sanity] embedding-NN within return top-5 correlated: {in_top5:.1%}")
    print(f"[sanity] examples (symbol -> emb-NN / corr-NN):")
    for s in syms[:12]:
        flag = "OK" if nn_emb[s] == nn_corr[s] else ("~" if nn_emb[s] in top5[s] else "x")
        print(f"    {s:6s} -> {nn_emb[s]:6s} / {nn_corr[s]:6s}  [{flag}]")


if __name__ == "__main__":
    _sanity_check()
