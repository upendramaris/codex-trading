"""
Advanced strategy implementations covering mean reversion, momentum, ML signals, and portfolio optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ml import PredictionEngine
from strategies.base import Signal, Strategy


@dataclass
class MeanReversionParams:
    lookback: int = 20
    entry_zscore: float = 1.0
    exit_zscore: float = 0.25


class MeanReversionStrategy(Strategy):
    """Simple mean reversion strategy using z-score of close prices."""

    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None):
        super().__init__(symbol, params)
        merged = MeanReversionParams(**(params or {}))
        self.config = merged

    def generate_signals(self, data: pd.DataFrame, context=None) -> Signal:
        if len(data) < self.config.lookback + 1:
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        close = data["close"]
        rolling_mean = close.rolling(self.config.lookback).mean()
        rolling_std = close.rolling(self.config.lookback).std()
        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]
        if latest_std == 0 or np.isnan(latest_std):
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        zscore = (close.iloc[-1] - latest_mean) / latest_std
        prev_zscore = (close.iloc[-2] - rolling_mean.iloc[-2]) / rolling_std.iloc[-2]

        if zscore < -self.config.entry_zscore and prev_zscore >= -self.config.entry_zscore:
            return Signal(symbol=self.symbol, side="buy", confidence=min(abs(zscore), 2.0))
        if zscore > self.config.entry_zscore and prev_zscore <= self.config.entry_zscore:
            return Signal(symbol=self.symbol, side="sell", confidence=min(abs(zscore), 2.0))
        if abs(zscore) < self.config.exit_zscore:
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        return Signal(symbol=self.symbol, side="hold", confidence=0.0)


@dataclass
class MomentumParams:
    fast_window: int = 10
    slow_window: int = 40


class MomentumStrategy(Strategy):
    """Momentum strategy using moving average crossover with slope confirmation."""

    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None):
        super().__init__(symbol, params)
        merged = MomentumParams(**(params or {}))
        self.config = merged

    def generate_signals(self, data: pd.DataFrame, context=None) -> Signal:
        if len(data) < self.config.slow_window + 2:
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        close = data["close"]
        fast = close.rolling(self.config.fast_window).mean()
        slow = close.rolling(self.config.slow_window).mean()
        fast_slope = fast.diff().iloc[-1]
        slow_slope = slow.diff().iloc[-1]
        latest_fast = fast.iloc[-1]
        latest_slow = slow.iloc[-1]
        prev_fast = fast.iloc[-2]
        prev_slow = slow.iloc[-2]

        if latest_fast > latest_slow and prev_fast <= prev_slow and fast_slope > 0:
            return Signal(symbol=self.symbol, side="buy", confidence=float(min(1.0, fast_slope / abs(slow_slope or 1e-6))))
        if latest_fast < latest_slow and prev_fast >= prev_slow and fast_slope < 0:
            return Signal(symbol=self.symbol, side="sell", confidence=float(min(1.0, abs(fast_slope) / abs(slow_slope or 1e-6))))
        return Signal(symbol=self.symbol, side="hold", confidence=0.0)


class MLSignalStrategy(Strategy):
    """Wraps the prediction engine to generate ML-based signals."""

    def __init__(
        self,
        symbol: str,
        prediction_engine: PredictionEngine,
        feature_columns: List[str],
        confidence_threshold: float = 0.0,
    ):
        super().__init__(symbol)
        self.prediction_engine = prediction_engine
        self.feature_columns = feature_columns
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame, context=None) -> Signal:
        if len(data) < 10:
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        latest = data.tail(1)
        latest_features = latest[self.feature_columns] if all(col in latest.columns for col in self.feature_columns) else latest
        signals_df = self.prediction_engine.generate_signals(latest_features)
        signal_row = signals_df.iloc[-1]
        confidence = float(signal_row.confidence)
        if confidence <= self.confidence_threshold:
            return Signal(symbol=self.symbol, side="hold", confidence=0.0)
        side = "buy" if int(signal_row.signal) > 0 else "sell"
        return Signal(symbol=self.symbol, side=side, confidence=confidence)


class PortfolioOptimizationStrategy(Strategy):
    """
    Portfolio-level strategy that targets risk-parity weights across a basket.
    """

    def __init__(self, symbols: Iterable[str], lookback: int = 60, min_weight: float = 0.0):
        super().__init__(symbol="PORTFOLIO")
        self.symbols = [s.upper() for s in symbols]
        self.lookback = lookback
        self.min_weight = min_weight

    def generate_portfolio_signals(self, context: Dict) -> List[Signal]:
        close_df = pd.DataFrame(
            {
                symbol: ctx["primary"]["close"].tail(self.lookback)
                for symbol, ctx in context.items()
            }
        ).dropna()
        if close_df.empty or len(close_df) < self.lookback // 2:
            return []

        returns = close_df.pct_change().dropna()
        cov = returns.cov()
        vol = returns.std()
        inv_vol = 1 / vol.replace(0, np.nan)
        weights = inv_vol / inv_vol.sum()
        weights = weights.clip(lower=self.min_weight)
        weights = weights / weights.sum()

        signals = [
            Signal(symbol=symbol, side="buy" if weight >= 0 else "sell", target_weight=float(weight), confidence=float(abs(weight)))
            for symbol, weight in weights.items()
        ]
        return signals
