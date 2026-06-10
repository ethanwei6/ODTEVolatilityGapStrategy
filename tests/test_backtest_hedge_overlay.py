from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtest_hedge_overlay import HedgeBacktestConfig, run_backtest


def test_hedge_overlay_runs_on_synthetic_data() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dates = pd.bdate_range("2023-01-02", periods=300)
        t = np.arange(len(dates))
        strategy_return = 0.0005 + np.sin(t / 18) * 0.001
        signal_change = np.where(t % 37 == 0, 0.25, np.cos(t / 13) * 0.02)
        hedge_return = np.where(signal_change > 0.20, 0.03, -0.0004)
        df = pd.DataFrame(
            {
                "date": dates,
                "strategy_return": strategy_return,
                "vol_signal_change": signal_change,
                "hedge_return": hedge_return,
            }
        )
        input_path = root / "synthetic.csv"
        output_dir = root / "outputs"
        df.to_csv(input_path, index=False)

        cfg = HedgeBacktestConfig(
            input_path=input_path,
            output_dir=output_dir,
            date_col="date",
            strategy_return_col="strategy_return",
            signal_col="vol_signal_change",
            hedge_return_col="hedge_return",
            trigger_return=0.20,
            weights=(0.05, 0.10),
        )
        bt, summary, diagnostics = run_backtest(cfg)

        assert len(bt) == len(df)
        assert {"base", "permanent_funded", "conditional_funded", "conditional_overlay"}.issubset(set(summary["type"]))
        assert diagnostics.loc[diagnostics["metric"].eq("hedge_active_days"), "value"].iloc[0] > 0
        assert (output_dir / "hedge_overlay_summary.csv").exists()

