from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    input_path: Path
    output_dir: Path
    date_col: str
    return_col: str
    signal_col: str
    percentile: float = 0.90
    min_history: int = 252
    initial_capital: float = 1_000_000.0


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(description="Backtest a volatility-shock timing overlay.")
    parser.add_argument("--input", required=True, type=Path, help="Local CSV input path.")
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path, help="Directory for generated tables and charts.")
    parser.add_argument("--date-col", default="date", help="Date column name.")
    parser.add_argument("--return-col", default="strategy_return", help="Daily strategy return column.")
    parser.add_argument("--signal-col", default="vol_signal_change", help="Volatility signal change column.")
    parser.add_argument("--percentile", default=0.90, type=float, help="Expanding percentile threshold, e.g. 0.90.")
    parser.add_argument("--min-history", default=252, type=int, help="Minimum observations before signals can trigger.")
    parser.add_argument("--initial-capital", default=1_000_000.0, type=float, help="Starting capital for wealth curves.")
    args = parser.parse_args()
    return BacktestConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        date_col=args.date_col,
        return_col=args.return_col,
        signal_col=args.signal_col,
        percentile=args.percentile,
        min_history=args.min_history,
        initial_capital=args.initial_capital,
    )


def load_data(cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_path)
    missing = {cfg.date_col, cfg.return_col, cfg.signal_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df[[cfg.date_col, cfg.return_col, cfg.signal_col]].copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df[cfg.return_col] = pd.to_numeric(df[cfg.return_col], errors="coerce")
    df[cfg.signal_col] = pd.to_numeric(df[cfg.signal_col], errors="coerce")
    df = df.dropna().sort_values(cfg.date_col).drop_duplicates(cfg.date_col, keep="last").reset_index(drop=True)
    if len(df) < cfg.min_history + 2:
        raise ValueError("Not enough rows for the requested min-history and next-day backtest.")
    return df


def add_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    bt = df.copy()
    bt["expanding_threshold"] = bt[cfg.signal_col].expanding(cfg.min_history).quantile(cfg.percentile)
    bt["signal_at_close"] = bt[cfg.signal_col] >= bt["expanding_threshold"]
    bt.loc[bt["expanding_threshold"].isna(), "signal_at_close"] = False

    # Signal is observed at close t and applied to the next close-to-close return.
    prior_signal = bt["signal_at_close"].shift(1, fill_value=False)
    bt["always_long_exposure"] = 1.0
    bt["derisk_exposure"] = np.where(prior_signal, 0.0, 1.0)
    bt["short_overlay_exposure"] = np.where(prior_signal, -1.0, 1.0)

    bt["always_long_return"] = bt[cfg.return_col]
    bt["derisk_return"] = bt[cfg.return_col] * bt["derisk_exposure"]
    bt["short_overlay_return"] = bt[cfg.return_col] * bt["short_overlay_exposure"]
    bt["forward_1d_return"] = bt[cfg.return_col].shift(-1)
    return bt


def add_wealth_and_drawdowns(bt: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = bt.copy()
    for col in ["always_long_return", "derisk_return", "short_overlay_return"]:
        stem = col.removesuffix("_return")
        wealth_col = f"{stem}_wealth"
        dd_col = f"{stem}_drawdown"
        out[wealth_col] = cfg.initial_capital * (1.0 + out[col]).cumprod()
        out[dd_col] = out[wealth_col] / out[wealth_col].cummax() - 1.0
    return out


def performance_stats(bt: pd.DataFrame, return_col: str, wealth_col: str, drawdown_col: str) -> dict[str, float]:
    returns = bt[return_col].dropna()
    wealth = bt[wealth_col].dropna()
    drawdown = bt[drawdown_col].dropna()
    annualized_return = (1.0 + returns).prod() ** (252 / len(returns)) - 1.0
    annualized_vol = returns.std() * np.sqrt(252)
    downside_vol = returns[returns < 0].std() * np.sqrt(252)
    return {
        "days": float(len(returns)),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "return_over_vol": annualized_return / annualized_vol if annualized_vol else np.nan,
        "sortino_proxy": annualized_return / downside_vol if downside_vol else np.nan,
        "max_drawdown": drawdown.min(),
        "win_rate": (returns > 0).mean(),
        "final_wealth": wealth.iloc[-1],
        "worst_day": returns.min(),
        "best_day": returns.max(),
    }


def build_summary(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Always Long", "always_long_return", "always_long_wealth", "always_long_drawdown"),
        ("De-Risk Overlay", "derisk_return", "derisk_wealth", "derisk_drawdown"),
        ("Short Overlay", "short_overlay_return", "short_overlay_wealth", "short_overlay_drawdown"),
    ]
    for name, ret_col, wealth_col, dd_col in specs:
        row = {"strategy": name}
        row.update(performance_stats(bt, ret_col, wealth_col, dd_col))
        rows.append(row)
    return pd.DataFrame(rows)


def build_signal_diagnostics(bt: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    signal_days = bt[bt["signal_at_close"]].copy()
    non_signal = bt[~bt["signal_at_close"]].copy()
    return pd.DataFrame(
        [
            {"metric": "signal_days", "value": float(len(signal_days))},
            {"metric": "signal_rate", "value": len(signal_days) / len(bt)},
            {"metric": "average_signal_change_on_signal_days", "value": signal_days[cfg.signal_col].mean()},
            {"metric": "average_next_day_return_after_signal", "value": signal_days["forward_1d_return"].mean()},
            {"metric": "average_next_day_return_non_signal", "value": non_signal["forward_1d_return"].mean()},
            {"metric": "negative_next_day_rate_after_signal", "value": (signal_days["forward_1d_return"] < 0).mean()},
            {"metric": "negative_next_day_rate_non_signal", "value": (non_signal["forward_1d_return"] < 0).mean()},
        ]
    )


def sx(i: int, n: int, left: int, width: int) -> float:
    return left + i * width / max(n - 1, 1)


def sy(value: float, ymin: float, ymax: float, top: int, height: int) -> float:
    return top + (ymax - value) * height / (ymax - ymin)


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    return " ".join([f"M {first[0]:.2f} {first[1]:.2f}", *[f"L {x:.2f} {y:.2f}" for x, y in rest]])


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#17202a}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#53606f}.axis{font-size:12px;fill:#53606f}.legend{font-size:13px;fill:#263342}</style>',
        f'<text x="72" y="42" class="title">{escape(title)}</text>',
        f'<text x="72" y="66" class="sub">{escape(subtitle)}</text>',
    ]


def add_year_ticks(svg: list[str], dates: pd.Series, left: int, chart_w: int, n: int, y1: int, y2: int, label_y: int) -> None:
    clean_dates = pd.Series(dates).reset_index(drop=True)
    for year in sorted(clean_dates.dt.year.dropna().unique()):
        idxs = clean_dates[clean_dates.dt.year == year].index
        if len(idxs) == 0:
            continue
        x = sx(int(idxs[0]), n, left, chart_w)
        svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y1}" y2="{y2}" stroke="#edf0f3"/>')
        svg.append(f'<text x="{x:.2f}" y="{label_y}" class="axis" text-anchor="middle">{int(year)}</text>')


def make_cumulative_chart(bt: pd.DataFrame, cfg: BacktestConfig) -> Path:
    width, height = 1260, 720
    left, right, top, chart_h = 86, 42, 105, 475
    chart_w = width - left - right
    svg = svg_header(width, height, "Cumulative Performance", "Baseline versus volatility-shock timing overlays")
    series = [
        ("Always Long", "always_long_wealth", "#253858"),
        ("De-Risk Overlay", "derisk_wealth", "#16705b"),
        ("Short Overlay", "short_overlay_wealth", "#9b3a46"),
    ]
    ymin = min(float(bt[col].min()) for _, col, _ in series) * 0.94
    ymax = max(float(bt[col].max()) for _, col, _ in series) * 1.05
    for val in np.linspace(ymin, ymax, 6):
        y = sy(float(val), ymin, ymax, top, chart_h)
        svg.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e6e9ed"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.2f}" class="axis" text-anchor="end">${val:,.0f}</text>')
    for label, col, color in series:
        pts = [(sx(i, len(bt), left, chart_w), sy(float(row[col]), ymin, ymax, top, chart_h)) for i, row in bt.iterrows()]
        svg.append(f'<path d="{line_path(pts)}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
    add_year_ticks(svg, bt[cfg.date_col], left, chart_w, len(bt), top, top + chart_h, top + chart_h + 31)
    for i, (label, _, color) in enumerate(series):
        x = left + i * 185
        y = height - 48
        svg.append(f'<line x1="{x}" x2="{x+30}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x+40}" y="{y+4}" class="legend">{escape(label)}</text>')
    svg.append(f'<text x="{width-right}" y="{height-18}" class="axis" text-anchor="end">Generated {escape(datetime.now().strftime("%Y-%m-%d"))}</text>')
    svg.append("</svg>")
    path = cfg.output_dir / "cumulative_performance.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def make_drawdown_chart(bt: pd.DataFrame, cfg: BacktestConfig) -> Path:
    width, height = 1260, 650
    left, right, top, chart_h = 86, 42, 105, 405
    chart_w = width - left - right
    svg = svg_header(width, height, "Drawdown Comparison", "Drawdown profile of baseline versus timing overlays")
    series = [
        ("Always Long", "always_long_drawdown", "#253858"),
        ("De-Risk Overlay", "derisk_drawdown", "#16705b"),
        ("Short Overlay", "short_overlay_drawdown", "#9b3a46"),
    ]
    ymin = min(float(bt[col].min()) for _, col, _ in series) * 1.12
    ymax = 0.01
    for val in np.linspace(ymin, ymax, 6):
        y = sy(float(val), ymin, ymax, top, chart_h)
        svg.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e6e9ed"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.2f}" class="axis" text-anchor="end">{val*100:.0f}%</text>')
    for label, col, color in series:
        pts = [(sx(i, len(bt), left, chart_w), sy(float(row[col]), ymin, ymax, top, chart_h)) for i, row in bt.iterrows()]
        svg.append(f'<path d="{line_path(pts)}" fill="none" stroke="{color}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>')
    add_year_ticks(svg, bt[cfg.date_col], left, chart_w, len(bt), top, top + chart_h, top + chart_h + 31)
    for i, (label, _, color) in enumerate(series):
        x = left + i * 185
        y = height - 48
        svg.append(f'<line x1="{x}" x2="{x+30}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x+40}" y="{y+4}" class="legend">{escape(label)}</text>')
    svg.append("</svg>")
    path = cfg.output_dir / "drawdown_comparison.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def run_backtest(cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(cfg)
    bt = add_signals(df, cfg)
    bt = add_wealth_and_drawdowns(bt, cfg)
    summary = build_summary(bt)
    diagnostics = build_signal_diagnostics(bt, cfg)
    bt.to_csv(cfg.output_dir / "backtest_timeseries.csv", index=False)
    summary.to_csv(cfg.output_dir / "performance_summary.csv", index=False)
    diagnostics.to_csv(cfg.output_dir / "signal_diagnostics.csv", index=False)
    make_cumulative_chart(bt, cfg)
    make_drawdown_chart(bt, cfg)
    return bt, summary, diagnostics


def main() -> None:
    cfg = parse_args()
    _, summary, diagnostics = run_backtest(cfg)
    print("Performance summary")
    print(summary.to_string(index=False))
    print("\nSignal diagnostics")
    print(diagnostics.to_string(index=False))
    print(f"\nOutputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()

