"""
Abstract broker interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Order:
    symbol: str
    side: str  # buy or sell
    qty: float
    order_type: str = "market"
    time_in_force: str = "gtc"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False
    tag: Optional[str] = None


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    cost_basis: Optional[float] = None
    unrealized_pl: Optional[float] = None


@dataclass
class RiskParameters:
    """Risk limits applied before submitting orders."""

    max_position_value: Optional[float] = None
    max_position_percent: Optional[float] = None
    max_single_order_value: Optional[float] = None
    max_leverage: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_symbol_exposure: Dict[str, float] = field(default_factory=dict)
    sector_limits: Dict[str, float] = field(default_factory=dict)
    symbol_sectors: Dict[str, str] = field(default_factory=dict)


class RiskControl:
    """Validates orders against predefined risk limits."""

    def __init__(self, params: Optional[RiskParameters] = None):
        self.params = params or RiskParameters()
        self.daily_realized_pnl: float = 0.0

    def register_realized_pnl(self, pnl: float) -> None:
        self.daily_realized_pnl += pnl

    def validate(self, order: Order, account: Dict, positions: Dict[str, Position]) -> None:
        buying_power = float(account.get("buying_power", float("inf")))
        equity = float(account.get("equity", account.get("portfolio_value", 0.0)))
        symbol = order.symbol.upper()

        reference_price = order.limit_price or order.stop_price
        if reference_price is None:
            position = positions.get(symbol)
            if position and position.qty:
                reference_price = abs(position.market_value / position.qty)
            else:
                reference_price = float(account.get("last_price", 0.0)) or 0.0
        if not reference_price or reference_price <= 0:
            reference_price = 1.0

        order_value = abs(order.qty) * reference_price

        if order_value > buying_power:
            raise ValueError(f"Order value {order_value:.2f} exceeds available buying power {buying_power:.2f}")

        if self.params.max_single_order_value and order_value > self.params.max_single_order_value:
            raise ValueError(
                f"Order value {order_value:.2f} exceeds max single order value {self.params.max_single_order_value:.2f}"
            )

        position = positions.get(symbol)
        current_value = abs(position.market_value) if position else 0.0
        projected_value = current_value + order_value

        if self.params.max_position_value and projected_value > self.params.max_position_value:
            raise ValueError(
                f"Projected position value {projected_value:.2f} exceeds limit {self.params.max_position_value:.2f}"
            )

        if self.params.max_position_percent and equity:
            projected_pct = projected_value / equity
            if projected_pct > self.params.max_position_percent:
                raise ValueError(
                    f"Projected position percent {projected_pct:.2%} exceeds limit "
                    f"{self.params.max_position_percent:.2%}"
                )

        if self.params.max_leverage and equity:
            total_exposure = sum(abs(pos.market_value) for pos in positions.values()) + order_value
            leverage = total_exposure / equity
            if leverage > self.params.max_leverage:
                raise ValueError(f"Leverage {leverage:.2f} exceeds limit {self.params.max_leverage:.2f}")

        if self.params.max_daily_loss is not None and (-self.daily_realized_pnl) > self.params.max_daily_loss:
            raise ValueError("Daily loss limit breached; cannot place additional orders.")

        symbol_limit = self.params.max_symbol_exposure.get(symbol)
        if symbol_limit and projected_value > symbol_limit:
            raise ValueError(
                f"Projected exposure {projected_value:.2f} for {symbol} exceeds symbol limit {symbol_limit:.2f}"
            )

        sector_limits = self.params.sector_limits
        symbol_sectors = self.params.symbol_sectors
        if sector_limits and symbol_sectors:
            sector = symbol_sectors.get(symbol)
            if sector and sector in sector_limits:
                current_sector_exposure = sum(
                    abs(pos.market_value)
                    for sym, pos in positions.items()
                    if symbol_sectors.get(sym) == sector
                )
                if order.side.lower() == "sell":
                    projected_sector = max(current_sector_exposure - order_value, 0.0)
                else:
                    projected_sector = current_sector_exposure + order_value
                if projected_sector > sector_limits[sector]:
                    raise ValueError(
                        f"Projected sector exposure {projected_sector:.2f} for sector {sector} exceeds limit "
                        f"{sector_limits[sector]:.2f}"
                    )


class Broker(ABC):
    """Defines broker interface for execution and account information."""

    @abstractmethod
    def submit_order(self, order: Order) -> Dict:
        """Submit a new order and return the broker response."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""

    @abstractmethod
    def list_open_orders(self) -> Dict[str, Dict]:
        """Return currently open orders keyed by order id."""

    @abstractmethod
    def list_positions(self) -> Dict[str, Position]:
        """Return current open positions keyed by symbol."""

    @abstractmethod
    def get_account(self) -> Dict:
        """Return account information such as cash and buying power."""

    @abstractmethod
    def sync(self) -> None:
        """Refresh internal caches if needed."""

    @abstractmethod
    def register_trade_pnl(self, pnl: float) -> None:
        """Track realized pnl (used for risk limits)."""
