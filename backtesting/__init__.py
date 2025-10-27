"""
Backtesting toolkit for evaluating strategies.
"""

from .engine import (
    AdvancedBacktestEngine,
    BacktestResult,
    BacktestTrade,
    CommissionModel,
    PortfolioConstraints,
    SlippageModel,
    WalkForwardBacktester,
    WalkForwardWindow,
)
from .performance import calculate_performance_metrics
from .visualization import (
    plot_drawdown,
    plot_equity_curve,
    plot_risk_metrics,
    plot_trade_distribution,
)

BacktestEngine = AdvancedBacktestEngine  # backwards compatibility alias

__all__ = [
    "AdvancedBacktestEngine",
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "SlippageModel",
    "CommissionModel",
    "PortfolioConstraints",
    "WalkForwardBacktester",
    "WalkForwardWindow",
    "calculate_performance_metrics",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_trade_distribution",
    "plot_risk_metrics",
]
