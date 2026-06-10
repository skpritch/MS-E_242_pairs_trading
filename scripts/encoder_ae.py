"""encoder_ae.py — Autoencoder model + embedding extraction.

Architecture: seq_len(12) -> 64(tanh) -> 16 -> 64(tanh) -> seq_len(12), MSE loss.
Per-symbol embedding = mean of per-window encoder codes, L2-normalized.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import AEConfig, GLOBAL_SEED


class AutoEncoder(nn.Module):
    """Fully-connected autoencoder over standardized return windows."""

    def __init__(self, seq_len: int, hidden: int, emb_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.Tanh(),
            nn.Linear(hidden, emb_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, seq_len),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_daily_windows(
    train_logret: pd.DataFrame,
    symbols: list[str],
    seq_len: int = 12,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Chop returns into non-overlapping seq_len-bar windows per symbol.

    Returns (X, symbols_kept, rows_per_sym) where X is (n_samples, seq_len).
    """
    X_rows = []
    rows_per_sym: dict[str, list[int]] = {s: [] for s in symbols}

    for s in symbols:
        if s not in train_logret.columns:
            continue
        r = train_logret[s].to_numpy(dtype=np.float64)
        n_win = len(r) // seq_len
        for w in range(n_win):
            seg = r[w * seq_len:(w + 1) * seq_len]
            valid = np.isfinite(seg)
            if valid.mean() < min_valid_frac:
                continue
            seg = np.where(valid, seg, 0.0)
            row_idx = len(X_rows)
            X_rows.append(seg.astype(np.float32))
            rows_per_sym[s].append(row_idx)

    if not X_rows:
        return np.empty((0, seq_len), np.float32), [], rows_per_sym

    X = np.stack(X_rows)
    symbols_kept = [s for s in symbols if len(rows_per_sym[s]) > 0]
    return X, symbols_kept, rows_per_sym


def standardize(
    X: np.ndarray, mu: float | None = None, sigma: float | None = None
) -> tuple[np.ndarray, float, float]:
    """Global standardization. Returns (X_std, mu, sigma)."""
    if mu is None or sigma is None:
        flat = X.ravel()
        flat = flat[np.isfinite(flat)]
        mu = float(np.mean(flat)) if flat.size else 0.0
        sigma = float(np.std(flat)) if flat.size else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
    X_std = (X - mu) / sigma
    X_std = np.where(np.isfinite(X_std), X_std, 0.0).astype(np.float32)
    return X_std, mu, sigma


def train_autoencoder(
    X: np.ndarray, cfg: AEConfig, seed: int = GLOBAL_SEED,
    weights_path: str | None = None,
) -> AutoEncoder:
    """Train AE with MSE + early stopping. Loads from weights_path if it exists."""
    model = AutoEncoder(cfg.seq_len, cfg.hidden, cfg.emb_dim)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = X.shape[0]
    Xt = torch.from_numpy(X)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * 0.15)) if n > 10 else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx = perm
    X_tr = Xt[tr_idx]
    X_val = Xt[val_idx] if n_val > 0 else Xt[tr_idx]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(X_tr.shape[0])
        for b in range(0, X_tr.shape[0], cfg.batch_size):
            idx = order[b:b + cfg.batch_size]
            xb = X_tr[idx]
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(X_val), X_val).item())
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)
    return model


def extract_embeddings(
    model: AutoEncoder,
    X: np.ndarray,
    symbols: list[str],
    rows_per_sym: dict[str, list[int]],
    min_windows: int = 5,
) -> pd.DataFrame:
    """Mean of per-window encoder codes, L2-normalized. Returns DataFrame indexed by symbol."""
    with torch.no_grad():
        codes = model.encode(torch.from_numpy(X)).numpy()

    rows = []
    kept_syms = []
    for s in symbols:
        idxs = rows_per_sym.get(s, [])
        if len(idxs) < min_windows:
            continue
        emb = codes[idxs].mean(axis=0)
        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        rows.append(emb)
        kept_syms.append(s)

    if not kept_syms:
        return pd.DataFrame()

    emb_dim = codes.shape[1]
    return pd.DataFrame(
        np.stack(rows), index=kept_syms,
        columns=[f"emb_{i}" for i in range(emb_dim)])
