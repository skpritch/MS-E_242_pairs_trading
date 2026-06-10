# CLAUDE.md

## Project Overview

Pairs-trading backtest comparing three selection arms (classic correlation, autoencoder, GRU) within a shared walk-forward engine. Academic project for MS&E 242.

## Key Commands

```bash
# Run backtest (all arms)
python -m scripts.run_all

# Single arm, short date range
python -m scripts.run_all --arms classic --start 2019 --end 2020

# Import check
python -c "from scripts import config, signal, backtest, selector_classic, selector_ae, selector_gru"
```

## Architecture

All source code lives in `scripts/` as a single Python package. The three arms share:
- **Trading rule**: rolling z-score (window = 2 × OU half-life, floor 5 bars), entry |z|>2.5, exit |z|<0.5, stop |z|>4
- **Eligibility filters**: positive beta, Breusch-Pagan test, OU half-life 1–150 bars, spread vol > 0.005
- **Walk-forward**: 252-day train, 63-day test, quarterly roll
- **Ranking**: training-period Sharpe, top 20 pairs per fold

Arms differ only in candidate-pair generation:
- `selector_classic.py`: pairwise correlation pre-filter (top 300, ρ > 0.5)
- `selector_ae.py`: autoencoder embeddings → 8-NN per symbol
- `selector_gru.py`: GRU embeddings (with per-stock AR lookback) → 8-NN per symbol

## Data

Raw parquets in `data/bars_30m_YYYY.parquet`. Not in repo. 30-min bars, regular hours only (13 bars/day), 1,749 symbols.

## Key Design Choices (differ from daniel/ reference)

- Breusch-Pagan test (not Engle-Granger cointegration)
- Rolling z-score with window = 2 × OU half-life (same as Daniel's)
- 150 liquid stocks (not 200)
- Train Sharpe ranking (not p-value ranking)
- GRU model (not contrastive encoder)
- Model weights cached per fold (`output/{arm}/fold_XX_{ae,gru}_model.pt`); skips training on rerun

## Output

JSON intermediates saved to `output/{arm}/`. Per-fold files: `fold_XX_selected_pairs.json`, `fold_XX_metrics.json`. Final: `aggregate_metrics.json`, `equity_curve.json`, `fold_metrics.json`, `trade_log.json`. Cross-arm comparison: `output/comparison/summary.json`.
