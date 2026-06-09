# DATA_README — MSE242 pairs-trading data foundation (Agent A)

Single source of truth for the cleaned 30-minute panels. All downstream agents
(encoder, pair-selection, signal, backtest) should consume the cached artifacts
here or call `src/data_panel.py` directly. **No trading/ML logic lives in this
layer.**

Build/refresh everything:

```bash
.venv/bin/python -m src.data_panel          # default: 2016..2025, top-200 liquid
```

Runs in ~17 s, peak RSS ~4.4 GB (built year-by-year and pivoted incrementally;
only the needed columns are read from each parquet).

---

## 1. What gets built

### Price panel (`build_price_panel`)
WIDE table, **one row per 30-minute regular-hours bar**:

- `index`   = sorted, unique, tz-aware `DatetimeIndex` (UTC), named `timestamp_utc`.
- `columns` = `symbol` (sorted).
- `values`  = `close` price (float).

`close` is the price everywhere. **`vwap` is frequently 100% NaN** and is never
used (only available as a documented fallback arg).

A cell is NaN **iff** that symbol had no bar at that timestamp (not listed yet,
delisted, or a missing bar). Closes themselves are never NaN, so panel NaNs are
purely an *alignment/listing* phenomenon.

### Return panel (`build_return_panel`)
Bar-to-bar **log returns** `r_t = ln(P_t / P_{t-1})`, same index/columns.

**Overnight-gap handling (default `drop_overnight=True`):** the first bar of each
trading day is `09:30` NY, whose naive return spans the overnight gap from the
prior 16:00 close. Every `09:30` row is set to **NaN** so an overnight gap is
never mistaken for an intraday move. That leaves the **12 intraday transitions**
per day (09:30→10:00 … 15:00→15:30). The very first row of the whole panel is
also all-NaN (no prior bar).

### Liquid universe (`liquid_universe(years, top_n, date_range)`)
Ranks symbols by the **median per-bar dollar volume** (`close*volume`, median is
robust to volume spikes) over the window and returns the top-N. **Call it
per-walk-forward-window** with the *train* window's `date_range` so the tradeable
universe is chosen without lookahead into the test window.

### Coverage report (`coverage_report`)
Per-symbol `n_bars`, `first_date`, `last_date`, `n_days`, `expected_bars`,
`coverage_frac` = observed / (trading days in `[first,last]` × 13). Values near
1.0 = trades essentially every bar while listed.

---

## 2. Cached artifacts (`data/processed/`)

Default = **full-history symbols, 2016–2025, regular hours only**; 2026 is held
out as final OOS and is **not** in any cached panel (build it on demand with
`build_price_panel([2026])`).

| File | Shape (rows × cols) | NaN | Notes |
|---|---|---|---|
| `close_panel_full_2016_2025.parquet`     | 31238 × 798 | **0.70%** | full-history universe; dense, recommended default |
| `logret_panel_full_2016_2025.parquet`    | 31238 × 798 | 1.01% intraday | overnight rows dropped to NaN |
| `close_panel_liquid200_2016_2025.parquet`| 31238 × 200 | 13.73% | top-200 by median $-vol over full window |
| `logret_panel_liquid200_2016_2025.parquet`| 31238 × 200 | 13.73% intraday | overnight dropped |
| `coverage_report.csv`                    | 1749 rows | — | every symbol seen 2016–2025 |
| `liquid_universe_top200.csv`             | 200 rows | — | ranked, most-liquid first |

- **Rows = 31,238** distinct 30m regular bars; **13 bars/day** verified, grid
  `09:30..15:30` (each label = bar START; 15:30 covers 15:30–16:00).
- Date span: **2016-06-01 → 2025-12-31**.
- **798 full-history symbols** (present in regular hours every year 2016–2025).

---

## 3. Missingness — read this before modeling

- **Full-history panel: 0.70% cells NaN.** Tiny, mostly a few intermittent
  names + an occasional missing intra-day bar. Median per-symbol coverage 0.997.
  A handful of full-history symbols are weak (min coverage 0.596); drop them per
  window if a pair depends on one.
- **Liquid-200 panel: 13.73% NaN — do NOT read this as "the data is sparse".**
  It is an artifact of ranking liquidity over the *entire* 2016–2025 window: 53
  of the top-200 names are **recent IPOs/listings** (e.g. COIN, CRWV, CRCL,
  BMNR, MDLN listed 2025-12-17) that simply did not exist early on. Restricted
  to **2025 alone the liquid panel is only 1.5% NaN**, and the full panel 0.26%.
  Because `liquid_universe` is meant to be called **per train window**, the
  universe you actually trade in any window will be dense.

**Recommended NaN handling (walk-forward):**
1. Select the universe with `liquid_universe(date_range=train_window)`.
2. Slice the panel to that window + universe.
3. **Within the window**, require ≥ ~95% non-NaN per symbol; drop the rest.
4. For residual gaps, forward-fill prices a *small* bounded number of bars (or
   drop the bar). **Never forward-fill returns** — set missing returns to 0 only
   if a method needs a complete matrix, and prefer pairwise-complete stats
   otherwise. **Never fill across the overnight (09:30) boundary.**
5. Do not impute across a symbol's listing date — treat pre-listing as absent.

---

## 4. Alignment guarantees

- All panels share the **same index** (the global 30m regular-hours bar grid),
  so any column subset is automatically time-aligned — no reindexing needed to
  compare symbols or to join the close and return panels.
- Index is **sorted, unique, tz-aware UTC**. Use `.tz_convert("America/New_York")`
  for NY session logic; `09:30` marks each session open / overnight boundary.
- Log-return row `t` aligns to price rows `t` and `t-1`; the panels are designed
  so `shift(1)` is always the previous *bar* (and crosses days only at 09:30,
  which is exactly the row that is NaN'd out).

---

## 5. Caveats for downstream agents

- **2026 is intentionally excluded** from caches — final OOS. Build separately.
- **Re-rank liquidity per window** (`date_range`); the cached top-200 is a
  whole-period default for convenience, not a per-window universe.
- Survivorship: the full-history universe is, by construction, survivors of
  2016–2025. For a less biased universe, select per window from `coverage_report`
  / `liquid_universe` instead of the full-history set.
- `vwap` is unusable (often all-NaN); `close` only.
- `volume`/`close` are never 0 or NaN in regular hours; no duplicate
  `(symbol, timestamp)` rows. ~93–95% of symbol-days have the full 13 bars; the
  rest are short (early closes / thin names), which surfaces as panel NaNs.
