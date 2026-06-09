"""
metrics.py — performance metrics for the MSE242 mid-frequency pairs backtest.

All metrics operate on a per-bar OOS return series (net of costs) produced by
``src/backtest.py``. The cadence is 30-minute regular-hours bars with 12 intraday
return transitions per trading day (the 09:30 overnight bar carries no intraday
return; see DATA_README). Annualization therefore uses:

    BARS_PER_YEAR = 12 intraday returns/day * 252 trading days/year = 3024

This is the single annualization factor used everywhere; it must match the
mid-frequency cadence or the Sharpe is meaningless. (PLAN.md cites ~3,270
bars/year using all 13 bars; we trade on the 12 *intraday* return bars because
the overnight bar has no usable return, so 12*252 = 3024 is the honest figure.)

Returns here are arithmetic per-bar PnL on $1 of gross capital (the backtest
already converts the dollar-neutral spread PnL into a portfolio return), so
equity is the cumulative product (1+r) and Sharpe is mean/std * sqrt(BARS_PER_YEAR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mid-frequency annualization: 12 intraday return-bars/day * 252 days/year.
INTRADAY_BARS_PER_DAY = 12
TRADING_DAYS_PER_YEAR = 252
BARS_PER_YEAR = INTRADAY_BARS_PER_DAY * TRADING_DAYS_PER_YEAR  # 3024


def annualized_sharpe(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of a per-bar return series (risk-free = 0).

    Sharpe = mean(r)/std(r) * sqrt(bars_per_year). NaNs are dropped. Returns 0.0
    if there is no dispersion (degenerate / all-flat series).
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(bars_per_year))


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative equity (starts at 1.0) from per-bar arithmetic returns."""
    r = pd.Series(returns).fillna(0.0)
    return (1.0 + r).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the whole series."""
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    return float(eq.iloc[-1] - 1.0)


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
    """Maximum drawdown (as a negative fraction, e.g. -0.12) of the equity curve."""
    eq = equity_curve(returns)
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def turnover_stats(turnover: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> dict:
    """Annualized turnover from a per-bar traded-notional series.

    ``turnover`` = gross traded notional (sum of |Δweight| across legs) per bar,
    as a fraction of gross capital. Annualized = mean per-bar turnover *
    bars_per_year (i.e. how many times the book is turned over per year).
    """
    t = pd.Series(turnover).fillna(0.0)
    if t.empty:
        return {"turnover_annualized": 0.0, "turnover_per_bar": 0.0, "turnover_total": 0.0}
    return {
        "turnover_annualized": float(t.mean() * bars_per_year),
        "turnover_per_bar": float(t.mean()),
        "turnover_total": float(t.sum()),
    }


def hit_rate(returns: pd.Series) -> float:
    """Fraction of *non-zero* per-bar returns that are positive."""
    r = pd.Series(returns).dropna()
    r = r[r != 0.0]
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def trade_stats(trade_log: pd.DataFrame) -> dict:
    """Per-trade stats from a closed-trade log.

    Expects columns: 'pnl' (net trade PnL fraction) and 'holding_bars'
    (# bars the position was held). Returns count, win rate, avg holding bars,
    and avg PnL per trade.
    """
    if trade_log is None or len(trade_log) == 0:
        return {
            "n_trades": 0,
            "trade_win_rate": 0.0,
            "avg_holding_bars": 0.0,
            "avg_trade_pnl": 0.0,
        }
    tl = trade_log
    return {
        "n_trades": int(len(tl)),
        "trade_win_rate": float((tl["pnl"] > 0).mean()),
        "avg_holding_bars": float(tl["holding_bars"].mean()),
        "avg_trade_pnl": float(tl["pnl"].mean()),
    }


def summarize(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    trade_log: pd.DataFrame | None = None,
    n_active_pairs: int | float | None = None,
    bars_per_year: int = BARS_PER_YEAR,
) -> dict:
    """One-stop metrics dict for a per-bar OOS return series.

    Parameters
    ----------
    returns : per-bar net portfolio returns (arithmetic, on gross capital).
    turnover : per-bar gross traded notional (fraction of capital), optional.
    trade_log : closed-trade DataFrame with 'pnl' and 'holding_bars', optional.
    n_active_pairs : number of pairs traded (scalar or avg across folds), optional.
    """
    out = {
        "sharpe_annualized": annualized_sharpe(returns, bars_per_year),
        "return_total": total_return(returns),
        "return_annualized": annualized_return(returns, bars_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate_bars": hit_rate(returns),
        "n_bars": int(pd.Series(returns).shape[0]),
        "bars_per_year": bars_per_year,
    }
    if turnover is not None:
        out.update(turnover_stats(turnover, bars_per_year))
    if trade_log is not None:
        out.update(trade_stats(trade_log))
    if n_active_pairs is not None:
        out["n_active_pairs"] = float(n_active_pairs)
    return out
