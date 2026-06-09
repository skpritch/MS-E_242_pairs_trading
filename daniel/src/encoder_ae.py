"""
encoder_ae.py — sequence/return AUTOENCODER for per-symbol embeddings (Agent C1).

This is the ML novelty of the AUTOENCODER selection arm. For each walk-forward
fold we train a small autoencoder ON THE TRAIN WINDOW ONLY, encode every symbol
in the (train-window) liquid universe into a low-dimensional embedding, and hand
those embeddings to ``src/selector_ae.py`` which forms candidate pairs by small
embedding distance. Everything downstream of candidate generation (hedge-ratio
fit, spread mu/sigma, half-life, the z-score signal, costs) is the SHARED classic
machinery, so the headline A/B differs from the classic arm ONLY in how candidate
pairs are proposed.

NO-LOOKAHEAD RAIL (the load-bearing guarantee)
----------------------------------------------
``embed_window`` is handed the TRAIN-window panels by the backtest harness
(``backtest._slice_window`` already restricts to [train_start, train_end]). The
autoencoder is (re)trained from scratch on those train-window return series every
fold and never receives test-window rows. We additionally hard-ASSERT that the
panel handed in does not cross into any provided ``test_start``. The encoder
therefore cannot see the future: weights, the standardization stats, and the
embeddings are all functions of train data alone, and they are frozen before the
selector proposes any pair.

Design
------
* Per-symbol feature = the symbol's TRAIN-window intraday log-return vector,
  chopped into fixed-length non-overlapping windows of ``seq_len`` bars. Each
  window is one training sample; a symbol contributes many samples. This both
  augments the data and makes the embedding describe a symbol's *return-shape
  distribution* rather than one long vector.
* A symbol's embedding = the MEAN of its per-window encoder codes (a robust,
  order-free summary of how that symbol moves intraday over the train window).
* Architecture: a small fully-connected autoencoder
  ``seq_len -> 64 -> emb_dim -> 64 -> seq_len`` with tanh activations, trained to
  reconstruct the standardized return windows under MSE. Small + fast so it runs
  ~34 folds cheaply on CPU.
* Returns are standardized using TRAIN-window per-bar mean/std (frozen from train)
  before being fed to the net; missing returns are set to 0 only inside a window
  (we never fill across the overnight boundary because the log-return panel
  already NaNs the 09:30 rows, and we drop windows that are mostly NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class AEConfig:
    seq_len: int = 12          # one trading day of intraday returns per sample
    emb_dim: int = 16          # embedding dimensionality (8-32 reasonable)
    hidden: int = 64           # hidden width of encoder/decoder
    epochs: int = 40           # max epochs (early stop usually triggers earlier)
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 6   # stop if val loss does not improve this many epochs
    val_frac: float = 0.15     # fraction of windows held out (in-TRAIN) for early stop
    min_windows_per_symbol: int = 5    # drop symbols with too little train data
    min_valid_frac: float = 0.5        # a window must be >= this fraction non-NaN
    max_universe: int = 150    # cap symbols to bound runtime (matches classic arm)
    seed: int = 0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class _AutoEncoder(nn.Module):
    """Tiny fully-connected autoencoder over standardized return windows."""

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


# --------------------------------------------------------------------------- #
# Universe cap (mirror classic selectors._restrict_universe for a FAIR A/B)
# --------------------------------------------------------------------------- #


def _restrict_universe(train_close: pd.DataFrame, max_universe: int) -> list[str]:
    """Cap to the ``max_universe`` most liquid/variable names, exactly as the
    classic arm does (median price * realized log-range proxy). Keeping this
    identical to ``selectors._restrict_universe`` ensures the AE arm and the
    classic arm start from the SAME candidate universe, so the only difference is
    candidate-pair generation."""
    if train_close.shape[1] <= max_universe:
        return list(train_close.columns)
    logp = np.log(train_close)
    rng = (logp.max() - logp.min())
    proxy = (train_close.median() * rng).sort_values(ascending=False)
    return proxy.head(max_universe).index.tolist()


# --------------------------------------------------------------------------- #
# Feature construction (TRAIN-window log-returns -> standardized windows)
# --------------------------------------------------------------------------- #


def _build_samples(
    train_logret: pd.DataFrame,
    symbols: list[str],
    cfg: AEConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], np.ndarray, np.ndarray]:
    """Chop each symbol's train-window return series into ``seq_len`` windows.

    Returns
    -------
    X : (n_samples, seq_len) standardized return windows (float32).
    sym_of_row : (n_samples,) int index into ``symbols`` for each window.
    rows_per_sym : symbol -> list of row indices in X.
    mu, sd : (seq_len-agnostic) per-bar standardization stats from TRAIN only.
    """
    # Per-symbol-agnostic standardization: one mean/std over ALL train returns
    # (a single scalar pair). Robust and avoids per-symbol leakage of scale into
    # the embedding distance (we want shape, not magnitude, to dominate; magnitude
    # is partly normalized away). Computed on TRAIN data only.
    flat = train_logret[symbols].to_numpy(dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    mu = float(np.mean(flat)) if flat.size else 0.0
    sd = float(np.std(flat)) if flat.size else 1.0
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    L = cfg.seq_len
    X_rows: list[np.ndarray] = []
    sym_rows: list[int] = []
    rows_per_sym: dict[str, list[int]] = {s: [] for s in symbols}

    for si, s in enumerate(symbols):
        r = train_logret[s].to_numpy(dtype=np.float64)
        n_win = len(r) // L
        for w in range(n_win):
            seg = r[w * L:(w + 1) * L]
            valid = np.isfinite(seg)
            if valid.mean() < cfg.min_valid_frac:
                continue
            seg = np.where(valid, seg, 0.0)           # zero missing WITHIN window
            seg = (seg - mu) / sd                     # train-frozen standardization
            row_idx = len(X_rows)
            X_rows.append(seg.astype(np.float32))
            sym_rows.append(si)
            rows_per_sym[s].append(row_idx)

    if not X_rows:
        return (np.empty((0, L), np.float32), np.empty((0,), np.int64),
                rows_per_sym, np.array([mu]), np.array([sd]))

    X = np.stack(X_rows)
    sym_of_row = np.asarray(sym_rows, dtype=np.int64)
    return X, sym_of_row, rows_per_sym, np.array([mu]), np.array([sd])


# --------------------------------------------------------------------------- #
# Train + embed (per fold)
# --------------------------------------------------------------------------- #


def _train_autoencoder(X: np.ndarray, cfg: AEConfig) -> _AutoEncoder:
    """Train the AE on standardized train windows with MSE + early stopping.

    Early-stop validation is an IN-TRAIN random split of the windows; no test
    data is ever involved."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n = X.shape[0]
    Xt = torch.from_numpy(X)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * cfg.val_frac)) if n > 10 else 0
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if len(tr_idx) == 0:
        tr_idx = perm  # degenerate tiny case
    X_tr = Xt[tr_idx]
    X_val = Xt[val_idx] if n_val > 0 else Xt[tr_idx]

    model = _AutoEncoder(cfg.seq_len, cfg.hidden, cfg.emb_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0

    for _ in range(cfg.epochs):
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
            if bad >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def embed_window(
    train_close: pd.DataFrame,
    train_logret: pd.DataFrame,
    train_window: tuple[str, str],
    universe: list[str] | None = None,
    cfg: AEConfig | None = None,
    test_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Train the AE on the TRAIN window and return per-symbol embeddings.

    Parameters
    ----------
    train_close, train_logret : TRAIN-window panels (already sliced by the
        backtest harness to [train_start, train_end]).
    train_window : ('YYYY-MM-DD','YYYY-MM-DD') inclusive NY dates of the train
        window (used only for the no-lookahead assertion / diagnostics).
    universe : optional explicit symbol list; defaults to the columns of
        ``train_close`` capped via the classic liquidity proxy.
    cfg : AEConfig.
    test_start : optional Timestamp; if given we hard-assert the train panel ends
        strictly before it (belt-and-suspenders no-lookahead guard).

    Returns
    -------
    DataFrame indexed by symbol, columns = embedding dims (emb_0..emb_{d-1}).
    Symbols with too little train data are dropped.
    """
    if cfg is None:
        cfg = AEConfig()

    # --- NO-LOOKAHEAD ASSERTIONS -------------------------------------------- #
    # 1) The train panel must lie entirely within the declared train window.
    if len(train_close.index):
        ny = train_close.index.tz_convert("America/New_York")
        day = pd.DatetimeIndex(ny.normalize()).tz_localize(None)
        tw_lo = pd.Timestamp(train_window[0]).normalize()
        tw_hi = pd.Timestamp(train_window[1]).normalize()
        assert day.min() >= tw_lo and day.max() <= tw_hi, (
            f"LOOKAHEAD: train panel spans {day.min().date()}..{day.max().date()} "
            f"outside declared train window {train_window}")
        # 2) If a test_start is provided, the train data must end strictly before it.
        if test_start is not None:
            assert day.max() < pd.Timestamp(test_start).tz_localize(None).normalize(), (
                f"LOOKAHEAD: train data {day.max().date()} reaches test_start {test_start}")

    if universe is None:
        universe = _restrict_universe(train_close, cfg.max_universe)
    else:
        universe = [s for s in universe if s in train_close.columns]
        if len(universe) > cfg.max_universe:
            universe = _restrict_universe(train_close[universe], cfg.max_universe)

    if len(universe) < 2:
        return pd.DataFrame()

    X, sym_of_row, rows_per_sym, _, _ = _build_samples(train_logret, universe, cfg)
    if X.shape[0] < 10:
        return pd.DataFrame()

    model = _train_autoencoder(X, cfg)

    with torch.no_grad():
        codes = model.encode(torch.from_numpy(X)).numpy()  # (n_samples, emb_dim)

    # Symbol embedding = mean of its window codes (drop thin symbols).
    rows: list[np.ndarray] = []
    kept_syms: list[str] = []
    for s in universe:
        idxs = rows_per_sym.get(s, [])
        if len(idxs) < cfg.min_windows_per_symbol:
            continue
        rows.append(codes[idxs].mean(axis=0))
        kept_syms.append(s)

    if not kept_syms:
        return pd.DataFrame()

    emb = pd.DataFrame(np.stack(rows), index=kept_syms,
                       columns=[f"emb_{i}" for i in range(cfg.emb_dim)])
    emb.index.name = "symbol"
    return emb
