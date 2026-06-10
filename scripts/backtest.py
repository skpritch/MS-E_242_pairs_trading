"""backtest.py — Walk-forward pairs-trading engine.

Pluggable selector contract (all 3 arms implement this):
    def selector(train_close, train_logret, fold) -> list[PairParams]

No-lookahead: test_start > train_end every fold; train params frozen before test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    BARS_PER_YEAR,
    EligibilityConfig,
    Fold,
    PairParams,
    SignalConfig,
    WalkForwardConfig,
)
from . import signal as sig
from . import metrics as mx
from .io_utils import save_json


SelectorFn = Callable[[pd.DataFrame, pd.DataFrame, Fold], list[PairParams]]


def _top_n_liquid(
    dv_slice: pd.DataFrame, symbols: list[str], top_n: int
) -> list[str]:
    """Keep top-N symbols by median dollar volume from the training-window slice."""
    common = [s for s in symbols if s in dv_slice.columns]
    if len(common) <= top_n:
        return common
    med = dv_slice[common].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


# ── Fold generation ───────────────────────────────────────────────────────────

def _trading_days(index: pd.DatetimeIndex) -> np.ndarray:
    """Sorted unique NY trading dates."""
    ny = index.tz_convert("America/New_York")
    days = pd.DatetimeIndex(ny.normalize()).tz_localize(None).normalize()
    return np.array(sorted(pd.unique(days)))


def generate_folds(index: pd.DatetimeIndex, wf_cfg: WalkForwardConfig) -> list[Fold]:
    """Generate rolling walk-forward folds. No overlap between train/test."""
    days = _trading_days(index)
    folds = []
    n = len(days)
    start = 0
    fid = 0
    while start + wf_cfg.train_days + wf_cfg.test_days <= n:
        tr_lo = start
        tr_hi = start + wf_cfg.train_days - 1
        te_lo = start + wf_cfg.train_days
        te_hi = start + wf_cfg.train_days + wf_cfg.test_days - 1
        folds.append(Fold(
            fold_id=fid,
            train_start=pd.Timestamp(days[tr_lo]),
            train_end=pd.Timestamp(days[tr_hi]),
            test_start=pd.Timestamp(days[te_lo]),
            test_end=pd.Timestamp(days[te_hi]),
        ))
        fid += 1
        start += wf_cfg.roll_days
    return folds


# ── Window slicing ────────────────────────────────────────────────────────────

def slice_window(
    panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Slice panel to NY date range [start, end] inclusive."""
    ny = panel.index.tz_convert("America/New_York")
    day = pd.DatetimeIndex(ny.normalize()).tz_localize(None)
    mask = (day >= start.normalize()) & (day <= end.normalize())
    return panel.loc[mask]


def apply_nan_policy(
    close: pd.DataFrame, min_coverage: float = 0.95, ffill_limit: int = 3
) -> pd.DataFrame:
    """Drop low-coverage symbols, forward-fill remaining price gaps."""
    cov = close.notna().mean()
    keep = cov[cov >= min_coverage].index
    out = close[keep].copy()
    out = out.ffill(limit=ffill_limit)
    return out


# ── Backtest result ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    oos_returns: pd.Series
    oos_turnover: pd.Series
    equity: pd.Series
    trade_log: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate: dict
    folds: list = field(default_factory=list)


# ── Main walk-forward engine ─────────────────────────────────────────────────

def run_walk_forward(
    close: pd.DataFrame,
    logret: pd.DataFrame,
    selector_fn: SelectorFn,
    wf_cfg: WalkForwardConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    elig_cfg: EligibilityConfig | None = None,
    dv_panel: pd.DataFrame | None = None,
    results_dir: str | None = None,
    arm_name: str = "arm",
    verbose: bool = True,
) -> BacktestResult:
    """Run the full rolling walk-forward backtest with an injected selector.

    The selector_fn receives (train_close, train_logret, fold) and returns
    a list of PairParams. The engine handles slicing, test-window PnL, stitching.
    """
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()
    if signal_cfg is None:
        signal_cfg = SignalConfig()
    if elig_cfg is None:
        elig_cfg = EligibilityConfig()

    folds = generate_folds(close.index, wf_cfg)
    if verbose:
        print(f"[{arm_name}] {len(folds)} walk-forward folds "
              f"(train={wf_cfg.train_days}d test={wf_cfg.test_days}d roll={wf_cfg.roll_days}d)")

    all_ret: list[pd.Series] = []
    all_turn: list[pd.Series] = []
    all_trades: list[dict] = []
    fold_rows: list[dict] = []

    fold_iter = tqdm(folds, desc=f"{arm_name} folds", unit="fold") if verbose else folds
    for fold in fold_iter:
        # No-lookahead guard
        assert fold.test_start > fold.train_end, (
            f"LOOKAHEAD: fold {fold.fold_id} test_start {fold.test_start} "
            f"<= train_end {fold.train_end}")

        # ── TRAIN slice ──
        tr_close = slice_window(close, fold.train_start, fold.train_end)
        tr_close = apply_nan_policy(tr_close, min_coverage=elig_cfg.min_coverage)

        # ── Liquidity filter: top-N by median dollar volume on train window ──
        if dv_panel is not None:
            tr_dv = slice_window(dv_panel, fold.train_start, fold.train_end)
            liquid_syms = _top_n_liquid(
                tr_dv, list(tr_close.columns), elig_cfg.top_n_liquid)
            tr_close = tr_close[liquid_syms]

        tr_logret = logret.reindex(columns=tr_close.columns)
        tr_logret = slice_window(tr_logret, fold.train_start, fold.train_end)

        # ── Select pairs (TRAIN only) ──
        pairs = selector_fn(tr_close, tr_logret, fold)

        # ── TEST slice (frozen params only) ──
        te_close = slice_window(close, fold.test_start, fold.test_end)
        te_close = te_close.ffill(limit=3)
        te_logret = slice_window(logret, fold.test_start, fold.test_end)

        test_idx = te_close.index
        fold_ret = pd.Series(0.0, index=test_idx)
        fold_turn = pd.Series(0.0, index=test_idx)
        n_pairs = len(pairs)
        fold_trades: list[dict] = []

        if n_pairs > 0:
            wgt = 1.0 / n_pairs
            for p in pairs:
                pr = sig.pair_returns(te_close, te_logret, p, signal_cfg)
                fold_ret = fold_ret.add(
                    pr["ret"].reindex(test_idx).fillna(0.0) * wgt, fill_value=0.0)
                fold_turn = fold_turn.add(
                    pr["turnover"].reindex(test_idx).fillna(0.0) * wgt, fill_value=0.0)
                for t in pr["trades"]:
                    t = dict(t)
                    t["fold_id"] = fold.fold_id
                    t["weight"] = wgt
                    fold_trades.append(t)

        all_ret.append(fold_ret)
        all_turn.append(fold_turn)
        all_trades.extend(fold_trades)

        fm = mx.summarize(
            fold_ret, turnover=fold_turn,
            trade_log=pd.DataFrame(fold_trades) if fold_trades else None,
            n_active_pairs=n_pairs)
        fm = {
            "fold_id": fold.fold_id,
            "train_start": str(fold.train_start.date()) if hasattr(fold.train_start, 'date') else str(fold.train_start),
            "train_end": str(fold.train_end.date()) if hasattr(fold.train_end, 'date') else str(fold.train_end),
            "test_start": str(fold.test_start.date()) if hasattr(fold.test_start, 'date') else str(fold.test_start),
            "test_end": str(fold.test_end.date()) if hasattr(fold.test_end, 'date') else str(fold.test_end),
            **fm,
        }
        fold_rows.append(fm)

        # Save per-fold metrics intermediate
        if results_dir:
            save_json(fm, f"{results_dir}/fold_{fold.fold_id:02d}_metrics.json")

        if verbose:
            fold_iter.set_postfix(
                pairs=n_pairs,
                sharpe=f"{fm['sharpe_annualized']:.2f}",
                ret=f"{fm['return_total']:+.3%}")

    # ── Stitch OOS curve ──
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
        n_active_pairs=avg_pairs)

    equity = mx.equity_curve(oos_ret)

    return BacktestResult(
        oos_returns=oos_ret, oos_turnover=oos_turn, equity=equity,
        trade_log=trade_log, fold_metrics=fold_metrics,
        aggregate=aggregate, folds=folds)
