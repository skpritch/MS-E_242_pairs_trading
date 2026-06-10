"""signal.py — Hedge ratio, spread, z-score, positions, pair PnL.

Shared across all three arms (classic, autoencoder, GRU). Key design choices:
- OLS hedge ratio on log-prices (frozen from train)
- Rolling z-score: window = 2 * OU half-life, floored at 5 bars
- Breusch-Pagan homoskedasticity test (not Engle-Granger)
- State-machine position generation with entry/exit/stop thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from .config import PairParams, SignalConfig


# ── Hedge ratio / spread / half-life (TRAIN-side estimation) ──────────────────

def hedge_ratio_ols(logp_a: pd.Series, logp_b: pd.Series) -> tuple[float, float]:
    """OLS of logp_a on logp_b (with intercept). Returns (alpha, beta)."""
    df = pd.concat([logp_a, logp_b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 10:
        return np.nan, np.nan
    X = sm.add_constant(df["b"].to_numpy())
    res = sm.OLS(df["a"].to_numpy(), X).fit()
    return float(res.params[0]), float(res.params[1])


def build_spread(
    logp_a: pd.Series, logp_b: pd.Series, alpha: float, beta: float
) -> pd.Series:
    """Spread residual: s = logp_a - (alpha + beta * logp_b)."""
    return logp_a - (alpha + beta * logp_b)


def half_life_ou(spread: pd.Series) -> float:
    """OU/AR(1) half-life in bars. Returns np.nan if not mean-reverting."""
    s = pd.Series(spread).dropna()
    if len(s) < 20:
        return np.nan
    s_lag = s.shift(1)
    ds = (s - s_lag).dropna()
    s_lag = s_lag.loc[ds.index]
    X = sm.add_constant(s_lag.to_numpy())
    try:
        res = sm.OLS(ds.to_numpy(), X).fit()
    except Exception:
        return np.nan
    b = float(res.params[1])
    if b >= 0 or (1.0 + b) <= 0:
        return np.nan
    hl = -np.log(2.0) / np.log(1.0 + b)
    return float(hl) if np.isfinite(hl) and hl > 0 else np.nan


def fit_pair_params(
    train_close: pd.DataFrame, sym_a: str, sym_b: str
) -> PairParams | None:
    """Fit all train-frozen params for a single pair. Returns None on failure."""
    if sym_a not in train_close.columns or sym_b not in train_close.columns:
        return None
    logp_a = np.log(train_close[sym_a])
    logp_b = np.log(train_close[sym_b])
    alpha, beta = hedge_ratio_ols(logp_a, logp_b)
    if not np.isfinite(alpha) or not np.isfinite(beta) or beta == 0:
        return None
    spread = build_spread(logp_a, logp_b, alpha, beta).dropna()
    if len(spread) < 20:
        return None
    mu = float(spread.mean())
    sigma = float(spread.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    hl = half_life_ou(spread)
    return PairParams(sym_a, sym_b, alpha, beta, mu, sigma, hl)


# ── Z-score ──────────────────────────────────────────────────────────────────

def zscore_frozen(spread: pd.Series, mu: float, sigma: float) -> pd.Series:
    """z = (spread - mu_train) / sigma_train. No lookahead."""
    return (spread - mu) / sigma


def zscore_rolling(spread: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Causal rolling z-score (mean/std over trailing ``window`` bars)."""
    if min_periods is None:
        min_periods = max(window // 2, 5)
    m = spread.rolling(window, min_periods=min_periods).mean()
    sd = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    return (spread - m) / sd


# ── Position state machine ────────────────────────────────────────────────────

def generate_positions(
    z: pd.Series,
    z_entry: float = 2.5,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
) -> pd.Series:
    """Map z-score to per-bar spread position in {-1, 0, +1}.

    Enter SHORT spread when z >= +z_entry, LONG when z <= -z_entry.
    Exit to flat when |z| <= z_exit or |z| >= z_stop (hard stop).
    """
    z = pd.Series(z)
    pos = np.zeros(len(z), dtype=float)
    state = 0.0
    zv = z.to_numpy()
    for i in range(len(zv)):
        zi = zv[i]
        if not np.isfinite(zi):
            state = 0.0
        elif state == 0.0:
            if zi >= z_entry:
                state = -1.0
            elif zi <= -z_entry:
                state = 1.0
        else:
            if abs(zi) <= z_exit or abs(zi) >= z_stop:
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=z.index)


# ── Per-pair PnL on the TEST window ──────────────────────────────────────────

def pair_returns(
    test_close: pd.DataFrame,
    test_logret: pd.DataFrame,
    params: PairParams,
    signal_cfg: SignalConfig | None = None,
) -> dict:
    """Compute per-bar net returns & turnover for ONE pair over the TEST window.

    Dollar-neutral: +1 spread unit = +1/(1+|beta|) in A, -beta/(1+|beta|) in B.
    Uses frozen z-score (train mu/sigma). Returns dict with 'ret', 'gross_ret',
    'turnover', 'position', 'trades'.
    """
    if signal_cfg is None:
        signal_cfg = SignalConfig()

    a, b, beta = params.sym_a, params.sym_b, params.beta
    if a not in test_close.columns or b not in test_close.columns:
        idx = test_close.index
        zero = pd.Series(0.0, index=idx)
        return {"ret": zero, "gross_ret": zero, "turnover": zero,
                "position": zero, "trades": []}

    logp_a = np.log(test_close[a])
    logp_b = np.log(test_close[b])
    spread = build_spread(logp_a, logp_b, params.alpha, beta)

    if np.isfinite(params.half_life):
        win = int(max(round(params.half_life * 2), 5))
        z = zscore_rolling(spread, win)
    else:
        z = zscore_frozen(spread, params.mu, params.sigma)
    pos = generate_positions(z, signal_cfg.z_entry, signal_cfg.z_exit, signal_cfg.z_stop)

    denom = 1.0 + abs(beta)
    w_a_unit = 1.0 / denom
    w_b_unit = -beta / denom

    pos_in = pos.shift(1).fillna(0.0)
    w_a = pos_in * w_a_unit
    w_b = pos_in * w_b_unit

    r_a = test_logret[a].reindex(spread.index, fill_value=0.0) if a in test_logret.columns else pd.Series(0.0, index=spread.index)
    r_b = test_logret[b].reindex(spread.index, fill_value=0.0) if b in test_logret.columns else pd.Series(0.0, index=spread.index)
    r_a = r_a.fillna(0.0)
    r_b = r_b.fillna(0.0)

    gross_ret = w_a * r_a + w_b * r_b

    dw_a = w_a.diff().abs().fillna(w_a.abs())
    dw_b = w_b.diff().abs().fillna(w_b.abs())
    turnover = dw_a + dw_b
    cost = turnover * signal_cfg.cost_per_side
    net_ret = gross_ret - cost

    trades = _extract_trades(pos, gross_ret, cost, a, b)

    return {
        "ret": net_ret,
        "gross_ret": gross_ret,
        "turnover": turnover,
        "position": pos,
        "trades": trades,
    }


def _extract_trades(
    pos: pd.Series, gross_ret: pd.Series, cost: pd.Series, sym_a: str, sym_b: str
) -> list[dict]:
    """Turn a per-bar position series into a list of closed trades."""
    trades = []
    pv = pos.to_numpy()
    idx = pos.index
    gr = gross_ret.reindex(idx).fillna(0.0).to_numpy()
    cs = cost.reindex(idx).fillna(0.0).to_numpy()

    in_trade = False
    entry_i = None
    side = 0.0
    for i in range(len(pv)):
        p = pv[i]
        if not in_trade and p != 0.0:
            in_trade = True
            entry_i = i
            side = p
        elif in_trade and (p == 0.0 or p != side):
            lo = entry_i + 1
            hi = i + 1
            pnl = float(gr[lo:hi].sum() - cs[entry_i:hi].sum())
            trades.append({
                "sym_a": sym_a, "sym_b": sym_b, "side": side,
                "entry_idx": idx[entry_i], "exit_idx": idx[i],
                "holding_bars": i - entry_i, "pnl": pnl,
            })
            if p != 0.0:
                in_trade = True
                entry_i = i
                side = p
            else:
                in_trade = False
                side = 0.0
    if in_trade and entry_i is not None:
        i = len(pv) - 1
        lo = entry_i + 1
        hi = i + 1
        pnl = float(gr[lo:hi].sum() - cs[entry_i:hi].sum())
        trades.append({
            "sym_a": sym_a, "sym_b": sym_b, "side": side,
            "entry_idx": idx[entry_i], "exit_idx": idx[i],
            "holding_bars": i - entry_i, "pnl": pnl,
        })
    return trades


# ── Breusch-Pagan homoskedasticity test ───────────────────────────────────────

def breusch_pagan_test(
    train_close: pd.DataFrame, sym_a: str, sym_b: str
) -> tuple[float, float] | None:
    """OLS of log-prices + Breusch-Pagan test on residuals.

    Returns (bp_stat, bp_pval) or None on failure.
    """
    if sym_a not in train_close.columns or sym_b not in train_close.columns:
        return None
    logp_a = np.log(train_close[sym_a]).dropna()
    logp_b = np.log(train_close[sym_b]).dropna()
    common = logp_a.index.intersection(logp_b.index)
    if len(common) < 60:
        return None
    y = logp_a.loc[common].to_numpy()
    X = sm.add_constant(logp_b.loc[common].to_numpy())
    try:
        ols_result = sm.OLS(y, X).fit()
        bp_stat, bp_pval, _, _ = het_breuschpagan(ols_result.resid, X)
        return float(bp_stat), float(bp_pval)
    except Exception:
        return None
