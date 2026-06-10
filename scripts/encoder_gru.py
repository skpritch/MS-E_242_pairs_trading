"""encoder_gru.py — AR lookback + GRU model + embedding extraction.

Novel component: per-stock AR(15) determines effective lookback, then a shared
GRU+MLP is trained to predict next-bar returns. The 16-dim hidden layer before
the output is used as per-symbol embeddings.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import GRUConfig, GLOBAL_SEED


# ── AR lookback ───────────────────────────────────────────────────────────────

def fit_ar_lookback(
    returns: pd.Series,
    max_lags: int = 15,
    cutoff: float = 0.01,
) -> tuple[np.ndarray | None, int | None]:
    """Per-stock AR(max_lags) no intercept.

    Returns (coefficients, effective_lookback). Going backwards from lag max_lags,
    stop at first lag with |coeff| < cutoff.
    """
    r = returns.dropna()
    if len(r) < max_lags + 30:
        return None, None

    y = r.iloc[max_lags:].to_numpy()
    X = np.column_stack([r.shift(i).iloc[max_lags:].to_numpy() for i in range(1, max_lags + 1)])

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if valid.sum() < 30:
        return None, None
    y, X = y[valid], X[valid]

    try:
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return None, None

    effective = max_lags
    for lag in range(max_lags, 0, -1):
        if abs(coefs[lag - 1]) < cutoff:
            effective = lag - 1
        else:
            break
    effective = max(effective, 1)

    return coefs, effective


def compute_all_lookbacks(
    train_logret: pd.DataFrame,
    symbols: list[str],
    cfg: GRUConfig,
) -> dict[str, int]:
    """Per-symbol AR lookback. Defaults to 5 if AR fit fails."""
    lookbacks = {}
    for sym in symbols:
        if sym not in train_logret.columns:
            lookbacks[sym] = 5
            continue
        _, lb = fit_ar_lookback(train_logret[sym], cfg.max_lags, cfg.ar_cutoff)
        lookbacks[sym] = lb if lb is not None else 5
    return lookbacks


# ── GRU + MLP model ──────────────────────────────────────────────────────────

class GRUMLP(nn.Module):
    """GRU(input=1, hidden=64) -> Linear(64,16) -> ReLU -> Linear(16,1)."""

    def __init__(self, gru_hidden: int = 64, mlp_hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=gru_hidden, batch_first=True)
        self.fc1 = nn.Linear(gru_hidden, mlp_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full prediction: (batch, seq_len, 1) -> (batch, 1)."""
        _, h = self.gru(x)  # h: (1, batch, gru_hidden)
        h = h.squeeze(0)     # (batch, gru_hidden)
        z = self.relu(self.fc1(h))  # (batch, mlp_hidden)
        return self.fc2(z)          # (batch, 1)

    def extract_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """16-dim vector before output layer: (batch, seq_len, 1) -> (batch, mlp_hidden)."""
        _, h = self.gru(x)
        h = h.squeeze(0)
        return self.relu(self.fc1(h))


# ── Data construction ─────────────────────────────────────────────────────────

def build_gru_data(
    train_logret: pd.DataFrame,
    symbols: list[str],
    lookbacks: dict[str, int],
    max_lookback: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training data: per-stock rolling windows of length=lookback, target=next return.

    All windows padded to max_lookback (left-pad with 0). Pooled across stocks.
    Returns (X, y) where X is (n_samples, max_lookback, 1), y is (n_samples, 1).
    """
    X_all = []
    y_all = []

    for sym in symbols:
        if sym not in train_logret.columns:
            continue
        r = train_logret[sym].to_numpy(dtype=np.float64)
        lb = lookbacks.get(sym, 5)
        lb = min(lb, max_lookback)

        for i in range(lb, len(r)):
            target = r[i]
            if not np.isfinite(target):
                continue
            window = r[i - lb:i]
            if not np.all(np.isfinite(window)):
                continue
            # Left-pad to max_lookback
            padded = np.zeros(max_lookback, dtype=np.float32)
            padded[max_lookback - lb:] = window.astype(np.float32)
            X_all.append(padded)
            y_all.append(np.float32(target))

    if not X_all:
        return np.empty((0, max_lookback, 1), np.float32), np.empty((0, 1), np.float32)

    X = np.stack(X_all)[:, :, np.newaxis]  # (n, max_lookback, 1)
    y = np.array(y_all)[:, np.newaxis]      # (n, 1)
    return X, y


def train_gru(
    X: np.ndarray, y: np.ndarray, cfg: GRUConfig, seed: int = GLOBAL_SEED,
    weights_path: str | None = None,
) -> GRUMLP:
    """Train GRU+MLP with MSE loss, Adam, early stopping. Loads from weights_path if it exists."""
    model = GRUMLP(gru_hidden=cfg.gru_hidden, mlp_hidden=cfg.mlp_hidden)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        return model

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = X.shape[0]
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)

    perm = np.random.permutation(n)
    n_val = max(1, int(n * 0.15)) if n > 10 else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx = perm

    X_tr, y_tr = Xt[tr_idx], yt[tr_idx]
    X_val, y_val = (Xt[val_idx], yt[val_idx]) if n_val > 0 else (Xt[tr_idx], yt[tr_idx])

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
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(X_val), y_val).item())
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


def extract_gru_embeddings(
    model: GRUMLP,
    train_logret: pd.DataFrame,
    symbols: list[str],
    lookbacks: dict[str, int],
    max_lookback: int = 15,
) -> pd.DataFrame:
    """Mean of L2-normalized 16-dim hidden vectors per symbol.

    Returns DataFrame indexed by symbol, columns = emb_0..emb_15.
    """
    model.eval()
    rows = []
    kept_syms = []

    for sym in symbols:
        if sym not in train_logret.columns:
            continue
        r = train_logret[sym].to_numpy(dtype=np.float64)
        lb = lookbacks.get(sym, 5)
        lb = min(lb, max_lookback)

        windows = []
        for i in range(lb, len(r)):
            window = r[i - lb:i]
            if not np.all(np.isfinite(window)):
                continue
            padded = np.zeros(max_lookback, dtype=np.float32)
            padded[max_lookback - lb:] = window.astype(np.float32)
            windows.append(padded)

        if len(windows) < 5:
            continue

        X_sym = np.stack(windows)[:, :, np.newaxis]  # (n_windows, max_lookback, 1)
        with torch.no_grad():
            hidden = model.extract_hidden(torch.from_numpy(X_sym)).numpy()  # (n_windows, mlp_hidden)

        # L2-normalize each window's hidden vector, then average
        norms = np.linalg.norm(hidden, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        hidden_normed = hidden / norms
        emb = hidden_normed.mean(axis=0)

        # Final L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        rows.append(emb)
        kept_syms.append(sym)

    if not kept_syms:
        return pd.DataFrame()

    emb_dim = rows[0].shape[0]
    return pd.DataFrame(
        np.stack(rows), index=kept_syms,
        columns=[f"emb_{i}" for i in range(emb_dim)])
