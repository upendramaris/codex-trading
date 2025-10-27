"""
Real-time paper trading loop integrating data, models, broker, and risk controls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from brokers import Broker, Order
from data import MarketDataFetcher
from ml import FeatureEngineer, PredictionEngine
from strategies.base import Signal
from trading.portfolio import PortfolioManager
from trading.risk_controls import RiskController
from utils.logger import get_logger


@dataclass
class TradeLogEntry:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    reason: str
    order_id: Optional[str] = None


class PaperTradingLoop:
    """Manages live signal evaluation and execution against Alpaca paper brokerage."""

    def __init__(
        self,
        broker: Broker,
        data_fetcher: MarketDataFetcher,
        portfolio_manager: PortfolioManager,
        feature_engineer: FeatureEngineer,
        prediction_engine: PredictionEngine,
        symbols: List[str],
        primary_timeframe: str = "1m",
        strategy_handler: Optional[Callable[[str, pd.DataFrame, pd.DataFrame], Optional[Signal]]] = None,
        event_dispatcher: Optional[Callable[[str, Dict], None]] = None,
        risk_controller: Optional[RiskController] = None,
    ):
        self.broker = broker
        self.data_fetcher = data_fetcher
        self.portfolio_manager = portfolio_manager
        self.feature_engineer = feature_engineer
        self.prediction_engine = prediction_engine
        self.symbols = [symbol.upper() for symbol in symbols]
        self.primary_timeframe = primary_timeframe
        self.history: Dict[str, pd.DataFrame] = {symbol: pd.DataFrame() for symbol in self.symbols}
        self.risk_levels: Dict[str, Dict[str, float]] = {}
        self.trade_log: List[TradeLogEntry] = []
        self.feature_columns: Optional[List[str]] = None
        self.logger = get_logger(self.__class__.__name__)
        self.strategy_handler = strategy_handler
        self.event_dispatcher = event_dispatcher
        self.risk_controller = risk_controller
        self._last_equity_timestamp: Optional[pd.Timestamp] = None
        self._last_equity_value: Optional[float] = None
        self._current_halt_reason: Optional[str] = None

    def start(self, run_for_seconds: Optional[float] = None) -> None:
        """Start the live trading loop. This method blocks until completion."""

        def on_bar(symbol: str, bar: pd.Series) -> None:
            try:
                self._on_new_bar(symbol, bar)
            except Exception as exc:  # pragma: no cover - runtime guard
                self.logger.exception("Error handling bar for %s: %s", symbol, exc)

        self.logger.info(
            "Starting paper trading loop for symbols %s on %s timeframe",
            ", ".join(self.symbols),
            self.primary_timeframe,
        )
        self._dispatch_event(
            "loop_started",
            {"symbols": self.symbols, "timeframe": self.primary_timeframe},
        )
        self.data_fetcher.stream_live_data(self.symbols, on_bar, run_for_seconds)

    async def start_async(self, run_for_seconds: Optional[float] = None) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.start, run_for_seconds)

    def _on_new_bar(self, symbol: str, bar: pd.Series) -> None:
        timestamp = pd.Timestamp(bar.get("timestamp") or bar.name).tz_localize(None)
        bar_df = pd.DataFrame([bar.drop(labels=["timestamp"], errors="ignore")], index=[timestamp])
        history = self.history.setdefault(symbol, pd.DataFrame())
        history = pd.concat([history, bar_df])
        history = history[~history.index.duplicated(keep="last")].sort_index()
        self.history[symbol] = history.tail(1000)  # keep reasonable window

        self._dispatch_event("bar", {"symbol": symbol, "bar": bar_df.iloc[-1].to_dict()})
        if self.risk_controller:
            equity = self._get_equity(timestamp)
            allowed, reason = self.risk_controller.evaluate(symbol, timestamp, float(bar_df.iloc[-1]["close"]), equity)
            if allowed:
                if self._current_halt_reason:
                    self._dispatch_event("trading_resume", {"timestamp": timestamp, "previous_reason": self._current_halt_reason})
                    self._current_halt_reason = None
            else:
                if reason != self._current_halt_reason:
                    self._dispatch_event("trading_halt", {"timestamp": timestamp, "reason": reason})
                    self.logger.warning("Trading halted due to %s", reason)
                    if self.risk_controller.flatten_positions:
                        self._flatten_all_positions(reason)
                self._current_halt_reason = reason
                return
        self._evaluate_symbol(symbol)
        self._monitor_risk(symbol, bar_df.iloc[-1])

    def _evaluate_symbol(self, symbol: str) -> None:
        history = self.history[symbol]
        if len(history) < 100:  # ensure sufficient lookback for indicators
            return

        feature_output = self.feature_engineer.engineer(history, symbol)
        if not self.feature_columns:
            self.feature_columns = feature_output.feature_columns
        latest_features = feature_output.features.tail(1)

        signal: Optional[Signal] = None
        if self.strategy_handler:
            signal = self.strategy_handler(symbol, history, latest_features)
        else:
            signals_df = self.prediction_engine.generate_signals(latest_features[self.feature_columns])
            latest_signal = signals_df.iloc[-1]
            signal_value = int(latest_signal.signal)
            confidence = float(latest_signal.confidence)
            if signal_value == 0 or confidence <= 0:
                return
            side = "buy" if signal_value > 0 else "sell"
            signal = Signal(
                symbol=symbol,
                side=side,
                confidence=confidence,
            )

        if signal is None:
            return

        side = signal.side.lower()
        confidence = float(signal.confidence or 0.0)
        if side not in {"buy", "sell"}:
            return
        if confidence <= 0:
            confidence = 1.0

        self._dispatch_event(
            "signal",
            {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "timestamp": history.index[-1] if not history.empty else pd.Timestamp.utcnow(),
            },
        )

        account = self.broker.get_account()
        equity = float(account.get("equity", account.get("portfolio_value", 0.0)))
        current_positions = self.broker.list_positions()
        position = current_positions.get(symbol)
        current_qty = float(position.qty) if position else 0.0
        price = float(history.iloc[-1]["close"])
        atr = float(latest_features.get("atr", pd.Series(dtype=float)).iloc[-1]) if "atr" in latest_features.columns else None
        volatility = None
        for col in ["volatility_20", "volatility_50", "volatility_10"]:
            if col in latest_features.columns:
                volatility = float(latest_features[col].iloc[-1])
                break

        sizing = self.portfolio_manager.prepare_order(
            symbol=symbol,
            side=side,
            equity=equity,
            price=price,
            atr=atr,
            existing_position=current_qty if current_qty > 0 else None,
            volatility=volatility,
            target_weight=signal.target_weight,
            explicit_quantity=signal.quantity,
        )
        desired_qty = sizing["quantity"]
        if desired_qty <= 0:
            return

        if side == "buy":
            order_qty = max(desired_qty - current_qty, 0.0)
        else:
            order_qty = current_qty if current_qty > 0 else desired_qty

        if order_qty <= 0:
            return

        order = Order(
            symbol=symbol,
            side=side,
            qty=order_qty,
            order_type="market",
            time_in_force="gtc",
            tag=f"model_signal_conf_{confidence:.2f}",
        )
        response = self.broker.submit_order(order)
        order_id = response.get("id")
        self.trade_log.append(
            TradeLogEntry(
                timestamp=history.index[-1],
                symbol=symbol,
                side=side,
                quantity=order_qty,
                price=price,
                reason="model_signal",
                order_id=order_id,
            )
        )
        self._dispatch_event(
            "order_submitted",
            {
                "symbol": symbol,
                "side": side,
                "quantity": order_qty,
                "price": price,
                "confidence": confidence,
                "order_id": order_id,
                "signal": signal,
                "timestamp": history.index[-1] if not history.empty else pd.Timestamp.utcnow(),
            },
        )
        levels = {
            k: v for k, v in sizing.items() if k in {"stop_loss", "take_profit"} and v
        }
        if levels:
            self.risk_levels[symbol] = levels
        self.logger.info(
            "Executed %s order for %s qty=%.4f price=%.2f confidence=%.2f order_id=%s",
            side,
            symbol,
            order_qty,
            price,
            confidence,
            order_id,
        )

    def _monitor_risk(self, symbol: str, latest_bar: pd.Series) -> None:
        if symbol not in self.risk_levels:
            return
        price = float(latest_bar["close"])
        levels = self.risk_levels[symbol]
        exit_reason = self.portfolio_manager.risk_manager.should_exit(price, levels)
        if not exit_reason:
            return

        positions = self.broker.list_positions()
        position = positions.get(symbol)
        if not position or position.qty == 0:
            self.risk_levels.pop(symbol, None)
            return

        side = "sell" if position.qty > 0 else "buy"
        order = Order(symbol=symbol, side=side, qty=abs(position.qty), order_type="market")
        response = self.broker.submit_order(order)
        self.trade_log.append(
            TradeLogEntry(
                timestamp=pd.Timestamp.utcnow(),
                symbol=symbol,
                side=side,
                quantity=abs(position.qty),
                price=price,
                reason=exit_reason,
                order_id=response.get("id"),
            )
        )
        self.risk_levels.pop(symbol, None)
        self.logger.info(
            "Risk exit triggered for %s via %s at price %.2f",
            symbol,
            exit_reason,
            price,
        )
        self._dispatch_event(
            "risk_exit",
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "reason": exit_reason,
                "order_id": response.get("id"),
                "timestamp": pd.Timestamp.utcnow(),
            },
        )

    def _dispatch_event(self, event_type: str, payload: Dict) -> None:
        if self.event_dispatcher:
            try:
                self.event_dispatcher(event_type, payload)
            except Exception as exc:  # pragma: no cover - safety
                self.logger.exception("Error dispatching event %s: %s", event_type, exc)

    # Helpers ------------------------------------------------------------------
    def _get_equity(self, timestamp: pd.Timestamp) -> float:
        if self._last_equity_timestamp != timestamp:
            account = self.broker.get_account()
            self._last_equity_value = float(account.get("equity", account.get("portfolio_value", 0.0)))
            self._last_equity_timestamp = timestamp
        return float(self._last_equity_value or 0.0)

    def _flatten_all_positions(self, reason: str) -> None:
        positions = self.broker.list_positions()
        for pos in positions.values():
            qty = float(pos.qty)
            if abs(qty) < 1e-8:
                continue
            side = "sell" if qty > 0 else "buy"
            order = Order(symbol=pos.symbol, side=side, qty=abs(qty), order_type="market")
            try:
                response = self.broker.submit_order(order)
                self.trade_log.append(
                    TradeLogEntry(
                        timestamp=pd.Timestamp.utcnow(),
                        symbol=pos.symbol,
                        side=side,
                        quantity=abs(qty),
                        price=float(pos.market_value / qty) if qty != 0 else 0.0,
                        reason=f"flatten_{reason}",
                        order_id=response.get("id"),
                    )
                )
                self.logger.warning("Flattened %s position due to %s", pos.symbol, reason)
            except Exception as exc:  # pragma: no cover - safety
                self.logger.exception("Failed to flatten %s during %s: %s", pos.symbol, reason, exc)
        if self.risk_controller:
            self.risk_controller.flatten_positions = False
