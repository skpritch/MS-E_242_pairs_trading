# Embeddings-Based Pairs Trading

MS&E 242 project comparing three pairs-trading selection arms within a shared walk-forward backtest:

1. **Classic** — correlation pre-filter + Breusch-Pagan homoskedasticity test
2. **Autoencoder** — 12→64→16→64→12 autoencoder embeddings + KNN candidate pairs
3. **GRU** — per-stock AR(15) lookback + shared GRU-MLP embeddings + KNN candidate pairs

All arms share the same trading rule (rolling z-score with window = 2 × OU half-life, entry |z|>2.5, exit |z|<0.5, stop |z|>4), eligibility filters (positive beta, BP test, OU half-life 1–150 bars, spread vol > 0.005), and walk-forward structure (252-day train, 63-day test, quarterly roll). Pairs are ranked by training-period Sharpe and capped at 20 per fold. Model weights are cached per fold to `output/{arm}/fold_XX_{ae,gru}_model.pt`.

## Directory Structure

```
data/                  # 30-min bar parquets (bars_30m_YYYY.parquet)
scripts/               # All Python source modules + orchestrator
  config.py            # Constants and dataclass configs
  data_panel.py        # Load parquets, build price/return panels
  universe.py          # Liquid universe + coverage filter
  signal.py            # Hedge ratio, spread, z-score, positions, pair PnL
  metrics.py           # Sharpe, equity curve, drawdown, trade stats
  backtest.py          # Walk-forward engine
  selector_classic.py  # Classic correlation + BP selector
  encoder_ae.py        # Autoencoder model + embeddings
  selector_ae.py       # AE pair selector (KNN on embeddings)
  encoder_gru.py       # AR lookback + GRU model + embeddings
  selector_gru.py      # GRU pair selector (KNN on embeddings)
  io_utils.py          # JSON serialization helpers
  run_all.py           # Orchestrator CLI
notebooks/             # Jupyter notebooks (EDA, baseline)
output/                # JSON intermediates + final results (created at runtime)
results/               # Final figures (created at runtime)
daniel/                # Daniel's reference implementation
report_draft.md        # LaTeX report draft
```

## Usage

```bash
# Full run (all 3 arms, 2016–2025)
python -m scripts.run_all

# Single arm
python -m scripts.run_all --arms classic

# Short test run
python -m scripts.run_all --arms classic --start 2019 --end 2020
```

## Requirements

Python 3.10+ with: `numpy`, `pandas`, `statsmodels`, `torch`, `pyarrow`

## Data

30-minute OHLCV bars for US equities (June 2016 – December 2025), sourced from Polygon/Massive. ~1,749 symbols, 63M rows total. Not included in the repo.
