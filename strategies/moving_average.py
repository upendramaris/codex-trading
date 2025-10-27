"""
Example moving average crossover strategy.
"""

from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


class MovingAverageCrossStrategy(Strategy):
    """Generates buy/sell signals on moving average crossovers."""

    def generate_signals(self, data: pd.DataFrame, context=None) -> Signal:
        short_window = int(self.params.get("short_window", 10))
        long_window = int(self.params.get("long_window", 30))
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")

        history = data.tail(long_window + 1)
        if len(history) < long_window:
            raise ValueError("Not enough data to compute moving averages")

        history = history.assign(
            short_ma=history["close"].rolling(window=short_window).mean(),
            long_ma=history["close"].rolling(window=long_window).mean(),
        )
        latest = history.iloc[-1]
        prev = history.iloc[-2]

        if latest.short_ma > latest.long_ma and prev.short_ma <= prev.long_ma:
            return Signal(symbol=self.symbol, side="buy", confidence=1.0)
        if latest.short_ma < latest.long_ma and prev.short_ma >= prev.long_ma:
            return Signal(symbol=self.symbol, side="sell", confidence=1.0)
        return Signal(symbol=self.symbol, side="hold", confidence=0.0)
