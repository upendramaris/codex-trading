"""
Alpaca paper broker implementation using alpaca-py.
"""

from __future__ import annotations

from typing import Dict, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)

from config import settings
from utils.logger import get_logger
from .base import Broker, Order, Position, RiskControl, RiskParameters


class AlpacaPaperBroker(Broker):
    """Broker to interact with Alpaca paper trading account."""

    def __init__(self, risk_control: Optional[RiskControl] = None, paper: bool = True) -> None:
        self._client = TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.api_secret,
            paper=paper,
        )
        self._paper = paper
        self._risk_control = risk_control or RiskControl(
            params=RiskParameters(
                max_position_percent=0.25,
                max_leverage=2.0,
                max_single_order_value=250_000,
            )
        )
        self._positions: Dict[str, Position] = {}
        self._account: Dict = {}
        self._logger = get_logger(__name__)
        self.sync()

    def submit_order(self, order: Order) -> Dict:
        account = self.get_account()
        positions = self.list_positions()
        self._risk_control.validate(order, account, positions)

        request = self._build_order_request(order)
        response = self._client.submit_order(order_data=request)
        self._logger.info(
            "Submitted %s order for %s qty=%.4f at %s (client_order_id=%s)",
            order.order_type.upper(),
            order.symbol,
            order.qty,
            getattr(order, "limit_price", None),
            order.client_order_id,
        )
        self.sync()
        return response.dict()

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)
        self._logger.info("Cancelled order %s", order_id)
        self.sync()

    def list_open_orders(self) -> Dict[str, Dict]:
        orders = self._client.get_orders()
        return {order.id: order.dict() for order in orders}

    def list_positions(self) -> Dict[str, Position]:
        positions = self._client.get_all_positions()
        self._positions = {
            pos.symbol: Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                market_value=float(pos.market_value),
                cost_basis=float(pos.avg_entry_price or 0.0),
                unrealized_pl=float(pos.unrealized_pl or 0.0),
            )
            for pos in positions
        }
        return self._positions

    def get_account(self) -> Dict:
        self._account = self._client.get_account().dict()
        return self._account

    def sync(self) -> None:
        positions = self._client.get_all_positions()
        self._positions = {
            pos.symbol: Position(
                symbol=pos.symbol,
                qty=float(pos.qty),
                market_value=float(pos.market_value),
                cost_basis=float(pos.avg_entry_price or 0.0),
                unrealized_pl=float(pos.unrealized_pl or 0.0),
            )
            for pos in positions
        }
        self._account = self._client.get_account().dict()

    def register_trade_pnl(self, pnl: float) -> None:
        self._risk_control.register_realized_pnl(pnl)

    def _build_order_request(self, order: Order):
        time_in_force = (
            TimeInForce.GTC if order.time_in_force.lower() == "gtc" else TimeInForce.DAY
        )
        side = OrderSide.BUY if order.side.lower() == "buy" else OrderSide.SELL
        common_kwargs = {
            "symbol": order.symbol,
            "qty": order.qty,
            "side": side,
            "time_in_force": time_in_force,
            "client_order_id": order.client_order_id,
            "extended_hours": order.extended_hours,
        }

        order_type = order.order_type.lower()
        if order_type == "market":
            return MarketOrderRequest(**common_kwargs)
        if order_type == "limit":
            if order.limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            return LimitOrderRequest(limit_price=order.limit_price, **common_kwargs)
        if order_type in {"stop", "stop_loss"}:
            if order.stop_price is None:
                raise ValueError("stop_price is required for stop orders")
            return StopOrderRequest(stop_price=order.stop_price, **common_kwargs)
        if order_type in {"stop_limit", "stop_limit_order"}:
            if order.stop_price is None or order.limit_price is None:
                raise ValueError("stop_limit orders require stop_price and limit_price")
            return StopLimitOrderRequest(
                stop_price=order.stop_price,
                limit_price=order.limit_price,
                **common_kwargs,
            )
        if order_type in {"trailing_stop", "trail"}:
            if order.trail_price is None and order.trail_percent is None:
                raise ValueError("Trailing stop requires trail_price or trail_percent")
            return TrailingStopOrderRequest(
                trail_price=order.trail_price,
                trail_percent=order.trail_percent,
                **common_kwargs,
            )
        raise ValueError(f"Unsupported order type: {order.order_type}")
