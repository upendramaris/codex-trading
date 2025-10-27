"""
Portfolio and risk management helpers for live and paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

class PositionSizer:
    """Base class for position sizing models."""

    def size_position(self, equity: float, price: float, **kwargs) -> float:
        raise NotImplementedError


@dataclass
class FixedFractionalSizer(PositionSizer):
    """
    Fixed fractional position sizing.

    Example: with fraction=0.02, allocate 2% of account equity to each trade.
    """

    fraction: float = 0.02

    def size_position(self, equity: float, price: float, **kwargs) -> float:
        if price <= 0:
            return 0.0
        allocation = equity * self.fraction
        return max(allocation / price, 0.0)


@dataclass
class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion sizing using estimated edge."""

    probability_of_win: float
    payoff_ratio: float
    max_fraction: float = 0.25

    def size_position(self, equity: float, price: float, **kwargs) -> float:
        if price <= 0:
            return 0.0
        p = np.clip(self.probability_of_win, 0.0, 1.0)
        b = max(self.payoff_ratio, 0.0)
        if b == 0 or p == 0:
            return 0.0
        kelly_fraction = (p * (b + 1) - 1) / b
        kelly_fraction = np.clip(kelly_fraction, 0.0, self.max_fraction)
        allocation = equity * kelly_fraction
        return max(allocation / price, 0.0)


@dataclass
class VolatilityAdjustedSizer(PositionSizer):
    """
    Sizes positions inversely proportional to recent volatility.

    allocation = (risk_fraction * equity) / max(volatility, epsilon)
    """

    risk_fraction: float = 0.02
    min_volatility: float = 1e-4

    def size_position(self, equity: float, price: float, **kwargs) -> float:
        volatility = float(kwargs.get("volatility") or 0.0)
        atr = float(kwargs.get("atr") or 0.0)
        effective_vol = max(volatility, atr, self.min_volatility)
        allocation = equity * self.risk_fraction / effective_vol
        if price <= 0:
            return 0.0
        return max(allocation / price, 0.0)


@dataclass
class CorrelationSettings:
    max_average_correlation: float = 0.85
    penalty_exponent: float = 1.5
    min_scaling: float = 0.25


@dataclass
class RiskSettings:
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 4.0
    min_stop_loss_pct: float = 0.01
    max_stop_loss_pct: float = 0.05
    max_position_risk: float = 0.02  # max % of equity at risk per trade


class RiskManager:
    """Derives stop-loss and take-profit levels and enforces per-position risk."""

    def __init__(self, settings: Optional[RiskSettings] = None):
        self.settings = settings or RiskSettings()

    def compute_levels(
        self,
        entry_price: float,
        atr: Optional[float],
        equity: float,
        quantity: float,
    ) -> Dict[str, float]:
        if entry_price <= 0 or quantity <= 0:
            return {}

        atr_value = atr or entry_price * 0.01
        stop_loss_dist = max(
            self.settings.min_stop_loss_pct * entry_price,
            min(self.settings.max_stop_loss_pct * entry_price, self.settings.stop_loss_atr_multiple * atr_value),
        )
        take_profit_dist = self.settings.take_profit_atr_multiple * atr_value

        stop_loss = entry_price - stop_loss_dist
        take_profit = entry_price + take_profit_dist

        # ensure risk per trade within limits
        position_risk = (entry_price - stop_loss) * quantity
        max_risk_value = equity * self.settings.max_position_risk
        if position_risk > max_risk_value and (entry_price - stop_loss) > 0:
            adjusted_dist = max_risk_value / quantity
            stop_loss = entry_price - adjusted_dist

        return {
            "stop_loss": max(stop_loss, 0.0),
            "take_profit": max(take_profit, 0.0),
        }

    def should_exit(self, price: float, levels: Dict[str, float]) -> Optional[str]:
        stop_loss = levels.get("stop_loss")
        take_profit = levels.get("take_profit")
        if stop_loss and price <= stop_loss:
            return "stop_loss"
        if take_profit and price >= take_profit:
            return "take_profit"
        return None


class PortfolioManager:
    """Coordinates sizing, risk management, and order enrichment."""

    def __init__(
        self,
        position_sizer: PositionSizer,
        risk_manager: Optional[RiskManager] = None,
        correlation_settings: Optional[CorrelationSettings] = None,
    ):
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager or RiskManager()
        self.correlation_settings = correlation_settings or CorrelationSettings()
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def prepare_order(
        self,
        symbol: str,
        side: str,
        equity: float,
        price: float,
        atr: Optional[float] = None,
        existing_position: Optional[float] = None,
        volatility: Optional[float] = None,
        target_weight: Optional[float] = None,
        explicit_quantity: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        if explicit_quantity is not None:
            desired_qty = explicit_quantity
        elif target_weight is not None:
            desired_qty = (target_weight * equity) / price if price > 0 else 0.0
        else:
            desired_qty = self.position_sizer.size_position(
                equity,
                price,
                atr=atr,
                volatility=volatility,
            )
        if side.lower() == "sell" and existing_position:
            desired_qty = existing_position
        elif side.lower() == "sell":
            desired_qty = 0.0

        desired_qty = self._apply_correlation_penalty(symbol, desired_qty)

        levels = self.risk_manager.compute_levels(
            entry_price=price,
            atr=atr,
            equity=equity,
            quantity=desired_qty,
        )
        return {
            "quantity": desired_qty,
            "stop_loss": levels.get("stop_loss"),
            "take_profit": levels.get("take_profit"),
        }

    def update_correlation_matrix(self, matrix: pd.DataFrame) -> None:
        if matrix.empty:
            self.correlation_matrix = None
        else:
            self.correlation_matrix = matrix

    def _apply_correlation_penalty(self, symbol: str, quantity: float) -> float:
        if quantity <= 0:
            return quantity
        if self.correlation_matrix is None:
            return quantity
        if symbol not in self.correlation_matrix.columns:
            return quantity
        correlations = self.correlation_matrix[symbol].drop(symbol, errors="ignore").abs()
        if correlations.empty:
            return quantity
        avg_corr = float(correlations.mean())
        settings = self.correlation_settings
        if avg_corr <= settings.max_average_correlation:
            return quantity
        scale = (settings.max_average_correlation / avg_corr) ** settings.penalty_exponent
        scale = max(settings.min_scaling, min(1.0, scale))
        return quantity * scale


class CorrelationAnalyzer:
    """Provides basic correlation statistics across instruments."""

    @staticmethod
    def correlation_matrix(price_history: Dict[str, pd.Series]) -> pd.DataFrame:
        aligned = pd.DataFrame(price_history).dropna()
        returns = aligned.pct_change().dropna()
        return returns.corr()

    @staticmethod
    def average_correlation(corr_matrix: pd.DataFrame) -> float:
        if corr_matrix.empty:
            return 0.0
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        values = upper.stack().values
        if len(values) == 0:
            return 0.0
        return float(np.nanmean(values))
