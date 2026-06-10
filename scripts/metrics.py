"""metrics.py — Sharpe, equity curve, drawdown, trade stats.

Annualization: BARS_PER_YEAR = 12 intraday bars/day * 252 days = 3024.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BARS_PER_YEAR


def annualized_sharpe(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Annualized Sharpe ratio (risk-free = 0)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(bars_per_year))


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative equity (starts at 1.0)."""
    r = pd.Series(returns).fillna(0.0)
    return (1.0 + r).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return."""
    eq = equity_curve(returns)
    return float(eq.iloc[-1] - 1.0) if not eq.empty else 0.0


def annualized_return(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Geometric annualized return."""
    r = pd.Series(returns).fillna(0.0)
    n = len(r)
    if n == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (bars_per_year / n) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative fraction)."""
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    return float((eq / eq.cummax() - 1.0).min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero returns that are positive."""
    r = pd.Series(returns).dropna()
    r = r[r != 0.0]
    return float((r > 0).mean()) if not r.empty else 0.0


def trade_stats(trade_log: pd.DataFrame | None) -> dict:
    """Per-trade stats from a closed-trade log."""
    if trade_log is None or len(trade_log) == 0:
        return {"n_trades": 0, "trade_win_rate": 0.0,
                "avg_holding_bars": 0.0, "avg_trade_pnl": 0.0}
    return {
        "n_trades": int(len(trade_log)),
        "trade_win_rate": float((trade_log["pnl"] > 0).mean()),
        "avg_holding_bars": float(trade_log["holding_bars"].mean()),
        "avg_trade_pnl": float(trade_log["pnl"].mean()),
    }


def turnover_stats(turnover: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> dict:
    """Annualized turnover from a per-bar traded-notional series."""
    t = pd.Series(turnover).fillna(0.0)
    if t.empty:
        return {"turnover_annualized": 0.0, "turnover_per_bar": 0.0}
    return {
        "turnover_annualized": float(t.mean() * bars_per_year),
        "turnover_per_bar": float(t.mean()),
    }


def summarize(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    trade_log: pd.DataFrame | None = None,
    n_active_pairs: int | float | None = None,
    bars_per_year: int = BARS_PER_YEAR,
) -> dict:
    """One-stop metrics dict."""
    out = {
        "sharpe_annualized": annualized_sharpe(returns, bars_per_year),
        "return_total": total_return(returns),
        "return_annualized": annualized_return(returns, bars_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate_bars": hit_rate(returns),
        "n_bars": int(pd.Series(returns).shape[0]),
    }
    if turnover is not None:
        out.update(turnover_stats(turnover, bars_per_year))
    if trade_log is not None:
        out.update(trade_stats(trade_log))
    if n_active_pairs is not None:
        out["n_active_pairs"] = float(n_active_pairs)
    return out
