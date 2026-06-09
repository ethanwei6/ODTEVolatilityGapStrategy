from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtest_vol_gap import BacktestConfig, run_backtest


def test_backtest_runs_on_synthetic_data() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dates = pd.bdate_range("2022-01-03", periods=320)
        returns = np.full(len(dates), 0.001)
        signal = np.sin(np.arange(len(dates)) / 15) + np.where(np.arange(len(dates)) % 47 == 0, 4.0, 0.0)
        df = pd.DataFrame({"date": dates, "strategy_return": returns, "vol_signal_change": signal})
        input_path = root / "input.csv"
        output_dir = root / "outputs"
        df.to_csv(input_path, index=False)

        cfg = BacktestConfig(
            input_path=input_path,
            output_dir=output_dir,
            date_col="date",
            return_col="strategy_return",
            signal_col="vol_signal_change",
            percentile=0.90,
            min_history=60,
        )
        bt, summary, diagnostics = run_backtest(cfg)

        assert len(bt) == len(df)
        assert set(summary["strategy"]) == {"Always Long", "De-Risk Overlay", "Short Overlay"}
        assert diagnostics.loc[diagnostics["metric"].eq("signal_days"), "value"].iloc[0] > 0
        assert (output_dir / "performance_summary.csv").exists()
        assert (output_dir / "cumulative_performance.svg").exists()

