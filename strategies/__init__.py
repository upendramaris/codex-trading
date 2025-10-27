"""
Trading strategies and supporting abstractions.
"""

from .advanced import (
    MLSignalStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    PortfolioOptimizationStrategy,
)
from .base import Signal, Strategy
from .moving_average import MovingAverageCrossStrategy

__all__ = [
    "Strategy",
    "Signal",
    "MovingAverageCrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MLSignalStrategy",
    "PortfolioOptimizationStrategy",
]
