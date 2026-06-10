# Methodology

## Research Question

Can a short-term volatility shock signal improve the timing of a generic volatility-carry allocation?

The project is built around the idea that volatility-carry strategies can be harmed when realized volatility exceeds the implied volatility premium being earned. A sharp volatility-signal increase may identify a regime where the next trading day has a less favorable risk profile.

## No-Lookahead Signal

For each date `t`, the code computes:

```text
threshold_t = expanding_percentile(signal history through t)
signal_t = signal_change_t >= threshold_t
```

The signal is observed at the close of day `t`.

The exposure change is applied to day `t+1`:

```text
exposure_{t+1} = 0 if signal_t else 1
```

This means the model never uses information from the return it is trying to trade.

## Tested Policies

### Always Long

The baseline strategy holds the target return stream every day.

### De-Risk Overlay

The strategy moves to cash for one day after a large volatility-signal jump.

### Short Overlay

The strategy takes negative exposure for one day after a signal. This is included for comparison, but it is less conservative than the de-risk overlay.

## Metrics

The script reports:

- annualized return
- annualized volatility
- return divided by volatility
- Sortino-style downside ratio
- max drawdown
- win rate
- final wealth
- signal frequency
- average next-day return after signal days

## Hedge Overlay Extension

The hedge overlay module applies the same no-lookahead logic to a generic hedge instrument.

The signal is observed at close `t`, then hedge exposure is applied to the next close-to-close return. This allows comparison between:

- permanent hedge exposure
- conditional hedge exposure
- conditional additive overlay exposure

The code deliberately expects user-supplied local data and writes outputs to an ignored directory. This keeps the public repository reusable without exposing proprietary datasets or instrument-specific research.

## Interpretation

A useful signal does not need to predict every bad day. It can still be valuable if it reduces exposure during a subset of high-risk regimes and improves the portfolio's drawdown profile.
