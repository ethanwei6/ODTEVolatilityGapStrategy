# 0DTE Volatility Gap Strategy

This repository contains a reusable backtesting framework for studying whether short-term implied-volatility shocks can improve timing decisions around a generic 0DTE volatility-carry strategy.

The core research idea:

> A strategy that earns volatility carry may perform well when implied volatility exceeds realized volatility, but it can become vulnerable after sharp short-term volatility repricing events. A volatility-index jump may therefore be useful as a risk-management signal.

This repo intentionally does **not** include proprietary data, private ticker names, or company-specific strategy details. The code is designed so a user can plug in their own local CSV file.

## What The Backtest Does

The framework tests a simple no-lookahead overlay:

1. Load daily strategy returns and a short-term volatility signal.
2. Compute whether the volatility signal exceeds an expanding historical percentile threshold.
3. If the signal triggers at close `t`, change exposure for the next close-to-close return from `t` to `t+1`.
4. Compare:
   - always-long baseline
   - de-risk overlay
   - optional short overlay

The intended interpretation is risk management, not a guaranteed prediction model.

## Input Format

Provide a local CSV with at least:

| Column | Description |
| --- | --- |
| `date` | Trading date |
| `strategy_return` | Daily close-to-close return of the strategy being tested |
| `vol_signal_change` | Daily point change or return change in a volatility signal |

Example:

```csv
date,strategy_return,vol_signal_change
2024-01-02,0.0012,-0.30
2024-01-03,-0.0041,2.15
2024-01-04,0.0008,-0.85
```

Keep private or licensed datasets outside the repository. The default `.gitignore` excludes `data/`, `outputs/`, CSV files, and Excel files.

## Run

```bash
pip install -r requirements.txt
python src/backtest_vol_gap.py \
  --input data/local_input.csv \
  --date-col date \
  --return-col strategy_return \
  --signal-col vol_signal_change \
  --output-dir outputs
```

The script writes:

- `performance_summary.csv`
- `signal_diagnostics.csv`
- `backtest_timeseries.csv`
- `cumulative_performance.svg`
- `drawdown_comparison.svg`

## Hedge Overlay Backtest

The repository also includes a generic hedge-overlay tester. It requires a local CSV with:

| Column | Description |
| --- | --- |
| `date` | Trading date |
| `strategy_return` | Daily return of the core strategy |
| `vol_signal_change` | Prior-day volatility signal used to activate the hedge |
| `hedge_return` | Daily return of the hedge instrument |

Run:

```bash
python src/backtest_hedge_overlay.py \
  --input data/local_hedge_input.csv \
  --date-col date \
  --strategy-return-col strategy_return \
  --signal-col vol_signal_change \
  --hedge-return-col hedge_return \
  --trigger-return 0.20 \
  --weights 0.025,0.05,0.10,0.15,0.20 \
  --output-dir outputs
```

The hedge overlay compares:

- unhedged baseline
- permanent funded hedge
- conditional funded hedge
- conditional overlay hedge

## Strategy Variants

### Always Long

Holds the strategy every day.

### De-Risk Overlay

Holds the strategy normally, but moves to cash for one day after a large volatility-signal jump.

### Short Overlay

Holds the strategy normally, but takes the opposite exposure for one day after a signal. This is included as an aggressive research variant and should be treated carefully.

## Why This Is Interesting

Volatility-carry strategies often have attractive average returns but can suffer during abrupt realized-volatility shocks. A short-term volatility signal may not forecast returns perfectly, but it can still be useful if it identifies periods when the distribution of next-day returns becomes worse.

The goal is not to overfit a magic rule. The goal is to build a clean framework for asking:

- Does a volatility shock signal reduce drawdowns?
- Does it improve return per unit of volatility?
- Is the signal useful only contemporaneously, or does it have forward-looking value?
- Is de-risking more robust than shorting?
- Is a conditional hedge more efficient than a permanent hedge?

## Research Hygiene

This implementation uses an expanding percentile threshold and applies the signal only to the following day. That avoids a common lookahead mistake where the same-day signal is allowed to explain or trade the same-day return.

## Limitations

- No transaction costs, financing costs, short borrow costs, taxes, or market impact are included.
- Results depend heavily on the input dataset and strategy being tested.
- A volatility index can be useful as a signal without being directly tradable.
- This is research code, not investment advice.
