"""
Example showcasing the advanced backtesting engine with visualization and walk-forward analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from backtesting import (
    AdvancedBacktestEngine,
    PortfolioConstraints,
    SlippageModel,
    WalkForwardBacktester,
    WalkForwardWindow,
    plot_drawdown,
    plot_equity_curve,
    plot_risk_metrics,
    plot_trade_distribution,
)
from backtesting.engine import CommissionModel
from data import YFinanceDataProvider
from strategies.moving_average import MovingAverageCrossStrategy


def main() -> None:
    provider = YFinanceDataProvider()
    strategy = MovingAverageCrossStrategy(symbol="SPY", params={"short_window": 20, "long_window": 50})

    engine = AdvancedBacktestEngine(
        data_provider=provider,
        initial_capital=100_000,
        slippage_model=SlippageModel(bps=0.5),
        commission_model=CommissionModel(percentage=0.0005, minimum=1.0),
        constraints=PortfolioConstraints(max_leverage=1.5, max_position_weight=0.5, cash_reserve_pct=0.05),
    )

    end = pd.Timestamp(datetime.utcnow())
    start = end - pd.Timedelta(days=730)

    result = engine.run(
        strategy=strategy,
        symbols=["SPY"],
        start=start,
        end=end,
        primary_timeframe="1d",
        extra_timeframes=["1h"],
    )

    print("=== Performance Metrics ===")
    for key, value in result.metrics.items():
        print(f"{key:>25}: {value: .4f}")

    # Visualization
    figures = [
        plot_equity_curve(result),
        plot_drawdown(result),
        plot_trade_distribution(result.trades),
        plot_risk_metrics(result.metrics),
    ]
    for fig in figures:
        fig.show()

    # Walk-forward backtesting example (two slices)
    wf = WalkForwardBacktester(engine)
    midpoint = start + (end - start) / 2
    windows = [
        WalkForwardWindow(
            train_start=start,
            train_end=midpoint - timedelta(days=1),
            test_start=midpoint,
            test_end=end,
        ),
    ]

    def strategy_factory(training_data, window):
        # Training_data can be used to tune parameters; using fixed params for brevity.
        return MovingAverageCrossStrategy(symbol="SPY", params={"short_window": 20, "long_window": 50})

    walk_forward_results = wf.run(
        strategy_factory=strategy_factory,
        windows=windows,
        symbols=["SPY"],
        primary_timeframe="1d",
        extra_timeframes=["1h"],
    )

    print("\nWalk-forward segment metrics:")
    for idx, wf_result in enumerate(walk_forward_results, start=1):
        print(f"Segment {idx}: annualized_return={wf_result.metrics['annualized_return']:.4f}, "
              f"max_drawdown={wf_result.metrics['max_drawdown']:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
