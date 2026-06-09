"""
backtest.py — REUSABLE walk-forward pairs-trading engine (Agent B).

This is the shared harness the whole project plugs into. The classic-selection
arm (``src/selectors.py``) and the encoder-selection arm (Agent D) differ ONLY in
the injected ``select_pairs`` callable; the train/test split, spread construction,
signal, costs, and metrics are identical across arms, which is what makes the
headline A/B fair.

No-lookahead contract (a reviewer should be able to verify this by reading the
loop below)
--------------------------------------------------------------------------------
* For each fold we slice the panels into a TRAIN window and a disjoint, strictly
  later TEST window. ``_assert_no_overlap`` hard-asserts the boundary.
* The selector is called with TRAIN-ONLY panels and never sees the test window.
* Pair params (hedge ratio, alpha, spread mu/sigma, half-life) are produced from
  TRAIN data inside the selector (via signal.fit_pair_params) and FROZEN.
* On TEST we only build spreads/z-scores from those frozen params and the test
  prices; the z-score normalization constants come from train (signal.zscore_frozen).
* Equity is stitched continuously across folds (each fold's net returns are
  appended; the OOS curve is one unbroken series, no overlap, no gaps refilled
  from the future).

Pluggable selector signature (THE integration contract — Agent D must match it
EXACTLY)
--------------------------------------------------------------------------------
    select_pairs(
        train_close:  pd.DataFrame,   # TRAIN-window close panel (bars x symbols)
        train_logret: pd.DataFrame,   # TRAIN-window log-return panel, same index
        train_window: tuple[str, str] # ('YYYY-MM-DD','YYYY-MM-DD') inclusive NY dates
    ) -> list[signal.PairParams]

Each returned ``PairParams`` is a fully train-frozen, ready-to-trade pair. The
engine does NOT re-fit anything from it; it only consumes (sym_a, sym_b, beta,
alpha, mu, sigma, half_life). An alternative selector that returns this list type
slots in with zero engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import signal as sig
from . import metrics as mx

# A selector maps TRAIN panels + window -> list of frozen PairParams.
SelectorFn = Callable[[pd.DataFrame, pd.DataFrame, tuple[str, str]], list[sig.PairParams]]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardConfig:
    """Walk-forward + signal + cost configuration.

    Window sizes are in TRADING DAYS (converted to bars internally via the panel's
    day grid, so they are robust to short days).
    """

    train_days: int = 252      # 12 months
    test_days: int = 63        # 3 months OOS
    roll_days: int = 63        # roll quarterly

    z_entry: float = 2.5
    z_exit: float = 0.5
    z_stop: float = 4.0
    cost_per_side: float = 0.0005   # 5 bps/side, 10 bps round trip
    # Causal ROLLING z-score (trailing mean/std, window = 2*train half-life) is the
    # default. It is still strictly lookahead-free (only trailing test data) and,
    # unlike the frozen-train z, it adapts to the level-drift of intraday spreads.
    # Empirically the frozen-train z pins the signal to one side when the test
    # spread mean drifts (~0.5 sigma over a quarter), turning the strategy into a
    # structural loss generator with ~10x the turnover; rolling-z fixes both.
    use_rolling_z: bool = True

    bars_per_year: int = mx.BARS_PER_YEAR


# --------------------------------------------------------------------------- #
# Fold generation (trading-day aware)
# --------------------------------------------------------------------------- #


def _trading_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Sorted unique NY trading dates (as numpy datetime64[D]) for a bar index."""
    ny = index.tz_convert("America/New_York")
    days = pd.DatetimeIndex(ny.normalize()).tz_localize(None).normalize()
    return np.array(sorted(pd.unique(days)))


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_window(self) -> tuple[str, str]:
        return (self.train_start.strftime("%Y-%m-%d"),
                self.train_end.strftime("%Y-%m-%d"))

    def test_window(self) -> tuple[str, str]:
        return (self.test_start.strftime("%Y-%m-%d"),
                self.test_end.strftime("%Y-%m-%d"))


def generate_folds(index: pd.DatetimeIndex, cfg: WalkForwardConfig) -> list[Fold]:
    """Generate rolling walk-forward folds over the bar index.

    Each fold: TRAIN = ``train_days`` consecutive trading days, TEST = the next
    ``test_days`` trading days (disjoint, strictly later), then roll forward by
    ``roll_days``. Boundaries are on whole trading days so train/test never share
    a day.
    """
    days = _trading_days(index)
    folds: list[Fold] = []
    n = len(days)
    start = 0
    fid = 0
    while start + cfg.train_days + cfg.test_days <= n:
        tr_lo = start
        tr_hi = start + cfg.train_days - 1
        te_lo = start + cfg.train_days
        te_hi = start + cfg.train_days + cfg.test_days - 1
        folds.append(Fold(
            fold_id=fid,
            train_start=pd.Timestamp(days[tr_lo]),
            train_end=pd.Timestamp(days[tr_hi]),
            test_start=pd.Timestamp(days[te_lo]),
            test_end=pd.Timestamp(days[te_hi]),
        ))
        fid += 1
        start += cfg.roll_days
    return folds


# --------------------------------------------------------------------------- #
# Per-window slicing + NaN policy (DATA_README §3)
# --------------------------------------------------------------------------- #


def _slice_window(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice a panel to NY dates [start, end] inclusive (end-of-day inclusive)."""
    ny = panel.index.tz_convert("America/New_York")
    day = pd.DatetimeIndex(ny.normalize()).tz_localize(None)
    mask = (day >= start.normalize()) & (day <= end.normalize())
    return panel.loc[mask]


def _apply_nan_policy(
    close: pd.DataFrame, min_coverage: float = 0.95, ffill_limit: int = 3
) -> pd.DataFrame:
    """Per-window NaN policy on PRICES (DATA_README §3.3-3.5).

    * Drop symbols with < ``min_coverage`` non-NaN within the window.
    * Forward-fill remaining price gaps a bounded ``ffill_limit`` bars (prices
      only; returns are NEVER forward-filled, and never across 09:30 — the
      log-return panel already NaNs overnight rows, and we recompute returns from
      these ffilled prices only inside the window so any residual gap stays a 0
      return, never an imputed move).
    """
    cov = close.notna().mean()
    keep = cov[cov >= min_coverage].index
    out = close[keep].copy()
    out = out.ffill(limit=ffill_limit)
    return out


def _assert_no_overlap(fold: Fold) -> None:
    """Hard guard: test window must start strictly after train window ends."""
    assert fold.test_start > fold.train_end, (
        f"LOOKAHEAD: fold {fold.fold_id} test_start {fold.test_start} "
        f"<= train_end {fold.train_end}")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


@dataclass
class BacktestResult:
    oos_returns: pd.Series                 # stitched per-bar net OOS returns
    oos_turnover: pd.Series                # stitched per-bar turnover
    equity: pd.Series                      # cumulative equity (starts at 1.0)
    trade_log: pd.DataFrame                # all closed trades across folds
    fold_metrics: pd.DataFrame             # one row per fold
    aggregate: dict                        # whole-OOS metrics
    folds: list = field(default_factory=list)


def run_walk_forward(
    close_panel: pd.DataFrame,
    logret_panel: pd.DataFrame,
    select_pairs: SelectorFn,
    cfg: WalkForwardConfig | None = None,
    min_coverage: float = 0.95,
    verbose: bool = True,
) -> BacktestResult:
    """Run the full rolling walk-forward backtest with an injected selector.

    Parameters
    ----------
    close_panel, logret_panel : aligned wide panels (from src/data_panel.py).
    select_pairs : pluggable selector (see module docstring for the exact
        signature). Called on TRAIN-ONLY data; returns frozen PairParams.
    cfg : WalkForwardConfig (windows, z thresholds, costs).
    min_coverage : per-window minimum non-NaN fraction to keep a symbol.

    Returns
    -------
    BacktestResult with stitched OOS curve, per-fold + aggregate metrics, trades.
    """
    if cfg is None:
        cfg = WalkForwardConfig()

    folds = generate_folds(close_panel.index, cfg)
    if verbose:
        print(f"[backtest] {len(folds)} walk-forward folds "
              f"(train={cfg.train_days}d test={cfg.test_days}d roll={cfg.roll_days}d)")

    all_ret: list[pd.Series] = []
    all_turn: list[pd.Series] = []
    all_trades: list[dict] = []
    fold_rows: list[dict] = []

    for fold in folds:
        _assert_no_overlap(fold)  # no-lookahead guard

        # --- TRAIN slice (selection + param fitting happen ONLY here) ---
        tr_close = _slice_window(close_panel, fold.train_start, fold.train_end)
        tr_close = _apply_nan_policy(tr_close, min_coverage=min_coverage)
        tr_logret = logret_panel.reindex(columns=tr_close.columns)
        tr_logret = _slice_window(tr_logret, fold.train_start, fold.train_end)

        pairs = select_pairs(tr_close, tr_logret, fold.train_window())

        # --- TEST slice (frozen params only; no fitting) ---
        te_close = _slice_window(close_panel, fold.test_start, fold.test_end)
        te_close = te_close.ffill(limit=3)  # bounded price ffill, no return fill
        te_logret = _slice_window(logret_panel, fold.test_start, fold.test_end)

        test_idx = te_close.index
        fold_ret = pd.Series(0.0, index=test_idx)
        fold_turn = pd.Series(0.0, index=test_idx)
        n_pairs = len(pairs)
        fold_trades: list[dict] = []

        if n_pairs > 0:
            # Equal-weight pairs: each pair gets 1/n_pairs of gross capital so the
            # book targets gross exposure ~1.0 when all pairs are active.
            wgt = 1.0 / n_pairs
            for p in pairs:
                pr = sig.pair_returns(
                    te_close, te_logret, p,
                    z_entry=cfg.z_entry, z_exit=cfg.z_exit, z_stop=cfg.z_stop,
                    cost_per_side=cfg.cost_per_side, use_rolling_z=cfg.use_rolling_z,
                )
                fold_ret = fold_ret.add(pr["ret"].reindex(test_idx).fillna(0.0) * wgt,
                                        fill_value=0.0)
                fold_turn = fold_turn.add(pr["turnover"].reindex(test_idx).fillna(0.0) * wgt,
                                          fill_value=0.0)
                for t in pr["trades"]:
                    t = dict(t)
                    t["fold_id"] = fold.fold_id
                    t["weight"] = wgt
                    fold_trades.append(t)

        all_ret.append(fold_ret)
        all_turn.append(fold_turn)
        all_trades.extend(fold_trades)

        fm = mx.summarize(fold_ret, turnover=fold_turn,
                          trade_log=pd.DataFrame(fold_trades) if fold_trades else None,
                          n_active_pairs=n_pairs, bars_per_year=cfg.bars_per_year)
        fm = {"fold_id": fold.fold_id,
              "train_start": fold.train_start.date(), "train_end": fold.train_end.date(),
              "test_start": fold.test_start.date(), "test_end": fold.test_end.date(),
              **fm}
        fold_rows.append(fm)
        if verbose:
            print(f"  fold {fold.fold_id:2d} test {fold.test_start.date()}..{fold.test_end.date()} "
                  f"pairs={n_pairs:3d} sharpe={fm['sharpe_annualized']:6.2f} "
                  f"ret={fm['return_total']:+.3%} turn/yr={fm.get('turnover_annualized',0):.0f}")

    # --- stitch continuous OOS curve across folds ---
    oos_ret = pd.concat(all_ret).sort_index()
    oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
    oos_turn = pd.concat(all_turn).sort_index()
    oos_turn = oos_turn[~oos_turn.index.duplicated(keep="first")]

    trade_log = pd.DataFrame(all_trades) if all_trades else pd.DataFrame(
        columns=["sym_a", "sym_b", "side", "entry_idx", "exit_idx",
                 "holding_bars", "pnl", "fold_id", "weight"])

    fold_metrics = pd.DataFrame(fold_rows)
    avg_pairs = float(fold_metrics["n_active_pairs"].mean()) if len(fold_metrics) else 0.0
    aggregate = mx.summarize(
        oos_ret, turnover=oos_turn,
        trade_log=trade_log if len(trade_log) else None,
        n_active_pairs=avg_pairs, bars_per_year=cfg.bars_per_year)

    equity = mx.equity_curve(oos_ret)

    return BacktestResult(
        oos_returns=oos_ret, oos_turnover=oos_turn, equity=equity,
        trade_log=trade_log, fold_metrics=fold_metrics,
        aggregate=aggregate, folds=folds)
