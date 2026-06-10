"""config.py — Constants and configuration dataclasses for the pairs-trading backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Global constants ──────────────────────────────────────────────────────────

BARS_PER_DAY = 13          # 09:30, 10:00, ..., 15:30
INTRADAY_BARS = 12         # 12 intraday return transitions (09:30 bar has no return)
BARS_PER_YEAR = INTRADAY_BARS * 252  # 3024
GLOBAL_SEED = 42
OPEN_BAR = "09:30"


# ── Walk-forward configuration ────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    train_days: int = 252    # 12 months of trading days
    test_days: int = 63      # 3 months OOS
    roll_days: int = 63      # roll quarterly


# ── Signal / trading-rule configuration ───────────────────────────────────────

@dataclass
class SignalConfig:
    z_entry: float = 2.5
    z_exit: float = 0.5
    z_stop: float = 4.0
    cost_per_side: float = 0.0005  # 5 bps/side


# ── Eligibility / pair-filter configuration ───────────────────────────────────

@dataclass
class EligibilityConfig:
    min_spread_vol: float = 0.005
    max_pairs: int = 20
    min_coverage: float = 0.95
    top_n_liquid: int = 150
    bp_alpha: float = 0.05   # Breusch-Pagan significance level


# ── Classic correlation arm ───────────────────────────────────────────────────

@dataclass
class ClassicConfig:
    corr_top_k: int = 300
    min_corr: float = 0.5


# ── Autoencoder arm ───────────────────────────────────────────────────────────

@dataclass
class AEConfig:
    seq_len: int = 12
    hidden: int = 64
    emb_dim: int = 16
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 6
    knn_k: int = 8
    cand_cap: int = 300


# ── GRU arm ───────────────────────────────────────────────────────────────────

@dataclass
class GRUConfig:
    max_lags: int = 15
    ar_cutoff: float = 0.01
    gru_hidden: int = 64
    mlp_hidden: int = 16
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 6
    knn_k: int = 8
    cand_cap: int = 300


# ── Fold / PairParams dataclasses ─────────────────────────────────────────────

@dataclass
class Fold:
    fold_id: int
    train_start: object  # pd.Timestamp
    train_end: object
    test_start: object
    test_end: object


@dataclass
class PairParams:
    sym_a: str
    sym_b: str
    alpha: float
    beta: float
    mu: float
    sigma: float
    half_life: float
    extra: dict = field(default_factory=dict)
