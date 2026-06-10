from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeBacktestConfig:
    input_path: Path
    output_dir: Path
    date_col: str
    strategy_return_col: str
    signal_col: str
    hedge_return_col: str
    trigger_return: float
    weights: tuple[float, ...]
    initial_capital: float = 1_000_000.0


def parse_weights(raw: str) -> tuple[float, ...]:
    weights = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not weights:
        raise ValueError("At least one hedge weight is required.")
    if any(w < 0 for w in weights):
        raise ValueError("Hedge weights must be nonnegative.")
    return weights


def parse_args() -> HedgeBacktestConfig:
    parser = argparse.ArgumentParser(description="Backtest a generic conditional hedge overlay.")
    parser.add_argument("--input", required=True, type=Path, help="Local CSV input path.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path, help="Directory for generated tables.")
    parser.add_argument("--date-col", default="date", help="Date column name.")
    parser.add_argument("--strategy-return-col", default="strategy_return", help="Daily return column for the core strategy.")
    parser.add_argument("--signal-col", default="vol_signal_change", help="Daily volatility signal change column.")
    parser.add_argument("--hedge-return-col", default="hedge_return", help="Daily return column for the hedge instrument.")
    parser.add_argument("--trigger-return", default=0.20, type=float, help="Prior-day signal threshold for activating the hedge.")
    parser.add_argument("--weights", default="0.025,0.05,0.10,0.15,0.20", help="Comma-separated hedge weights.")
    parser.add_argument("--initial-capital", default=1_000_000.0, type=float, help="Starting capital for wealth calculations.")
    args = parser.parse_args()
    return HedgeBacktestConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        date_col=args.date_col,
        strategy_return_col=args.strategy_return_col,
        signal_col=args.signal_col,
        hedge_return_col=args.hedge_return_col,
        trigger_return=args.trigger_return,
        weights=parse_weights(args.weights),
        initial_capital=args.initial_capital,
    )


def load_data(cfg: HedgeBacktestConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    required = {cfg.date_col, cfg.strategy_return_col, cfg.signal_col, cfg.hedge_return_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[[cfg.date_col, cfg.strategy_return_col, cfg.signal_col, cfg.hedge_return_col]].copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
    for col in [cfg.strategy_return_col, cfg.signal_col, cfg.hedge_return_col]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values(cfg.date_col).drop_duplicates(cfg.date_col, keep="last").reset_index(drop=True)
    if len(out) < 3:
        raise ValueError("Not enough observations for a prior-day signal backtest.")
    return out


def performance_stats(returns: pd.Series, initial_capital: float) -> dict[str, float]:
    r = returns.dropna()
    wealth = initial_capital * (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    annualized_return = (1 + r).prod() ** (252 / len(r)) - 1
    annualized_volatility = r.std() * np.sqrt(252)
    return {
        "days": float(len(r)),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "return_over_volatility": annualized_return / annualized_volatility if annualized_volatility else np.nan,
        "max_drawdown": drawdown.min(),
        "win_rate": (r > 0).mean(),
        "worst_day": r.min(),
        "best_day": r.max(),
        "final_wealth": wealth.iloc[-1],
    }


def add_strategy_returns(df: pd.DataFrame, cfg: HedgeBacktestConfig) -> pd.DataFrame:
    out = df.copy()
    prior_signal = out[cfg.signal_col].shift(1)
    out["hedge_active"] = prior_signal >= cfg.trigger_return
    out["unhedged_return"] = out[cfg.strategy_return_col]

    for weight in cfg.weights:
        label = f"{weight:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        out[f"permanent_funded_{label}"] = (1 - weight) * out[cfg.strategy_return_col] + weight * out[cfg.hedge_return_col]
        out[f"conditional_funded_{label}"] = np.where(
            out["hedge_active"],
            (1 - weight) * out[cfg.strategy_return_col] + weight * out[cfg.hedge_return_col],
            out[cfg.strategy_return_col],
        )
        out[f"conditional_overlay_{label}"] = np.where(
            out["hedge_active"],
            out[cfg.strategy_return_col] + weight * out[cfg.hedge_return_col],
            out[cfg.strategy_return_col],
        )
    return out


def build_summary(bt: pd.DataFrame, cfg: HedgeBacktestConfig) -> pd.DataFrame:
    rows = [{"strategy": "unhedged", "weight": 0.0, "type": "base", **performance_stats(bt["unhedged_return"], cfg.initial_capital)}]
    for weight in cfg.weights:
        label = f"{weight:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        specs = [
            (f"permanent_funded_{label}", "permanent_funded"),
            (f"conditional_funded_{label}", "conditional_funded"),
            (f"conditional_overlay_{label}", "conditional_overlay"),
        ]
        for col, kind in specs:
            rows.append({"strategy": col, "weight": weight, "type": kind, **performance_stats(bt[col], cfg.initial_capital)})
    return pd.DataFrame(rows)


def build_diagnostics(bt: pd.DataFrame, cfg: HedgeBacktestConfig) -> pd.DataFrame:
    signal_days = bt[bt["hedge_active"]].copy()
    non_signal_days = bt[~bt["hedge_active"]].copy()
    return pd.DataFrame(
        [
            {"metric": "hedge_active_days", "value": float(len(signal_days))},
            {"metric": "hedge_active_rate", "value": len(signal_days) / len(bt)},
            {"metric": "average_strategy_return_on_active_days", "value": signal_days[cfg.strategy_return_col].mean()},
            {"metric": "average_strategy_return_on_inactive_days", "value": non_signal_days[cfg.strategy_return_col].mean()},
            {"metric": "negative_strategy_return_rate_active_days", "value": (signal_days[cfg.strategy_return_col] < 0).mean()},
            {"metric": "negative_strategy_return_rate_inactive_days", "value": (non_signal_days[cfg.strategy_return_col] < 0).mean()},
            {"metric": "strategy_hedge_return_correlation", "value": bt[cfg.strategy_return_col].corr(bt[cfg.hedge_return_col])},
            {"metric": "max_abs_hedge_daily_return", "value": bt[cfg.hedge_return_col].abs().max()},
        ]
    )


def run_backtest(cfg: HedgeBacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(cfg)
    bt = add_strategy_returns(df, cfg)
    summary = build_summary(bt, cfg)
    diagnostics = build_diagnostics(bt, cfg)

    bt.to_csv(cfg.output_dir / "hedge_overlay_timeseries.csv", index=False)
    summary.to_csv(cfg.output_dir / "hedge_overlay_summary.csv", index=False)
    diagnostics.to_csv(cfg.output_dir / "hedge_overlay_diagnostics.csv", index=False)
    return bt, summary, diagnostics


def main() -> None:
    cfg = parse_args()
    _, summary, diagnostics = run_backtest(cfg)
    print("Hedge overlay summary")
    print(summary.to_string(index=False))
    print("\nDiagnostics")
    print(diagnostics.to_string(index=False))
    print(f"\nOutputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()

