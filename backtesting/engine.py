"""
Advanced backtesting engine with support for slippage, commissions, multi-timeframe
analysis, portfolio constraints, and walk-forward validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.base import DataProvider
from strategies.base import Signal, Strategy

from .performance import calculate_performance_metrics


@dataclass
class Position:
    quantity: float = 0.0
    cost_basis: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def exposure(self, price: float) -> float:
        return abs(self.quantity * price)


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def market_value(self, price_map: Dict[str, float]) -> float:
        return sum(
            self.positions.get(symbol, Position()).market_value(price)
            for symbol, price in price_map.items()
        )

    def total_value(self, price_map: Dict[str, float]) -> float:
        return self.cash + self.market_value(price_map)


@dataclass
class BacktestTrade:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    execution_price: float
    value: float
    commission: float
    slippage: float
    pnl: float
    portfolio_value: float


@dataclass
class BacktestResult:
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    returns: pd.Series
    portfolio_history: pd.DataFrame
    metrics: Dict[str, float]
    positions: Dict[str, Position]


class SlippageModel:
    """Applies a simple percentage (bps) and fixed slippage to executions."""

    def __init__(self, bps: float = 1.0, fixed: float = 0.0):
        self.bps = bps
        self.fixed = fixed

    def apply(self, price: float, side: str) -> float:
        direction = 1 if side.lower() == "buy" else -1
        adjusted = price * (1 + direction * self.bps / 10_000)
        adjusted += direction * self.fixed
        return adjusted


class CommissionModel:
    """Supports percentage- and per-share-based commissions."""

    def __init__(self, per_share: float = 0.0, percentage: float = 0.0, minimum: float = 0.0):
        self.per_share = per_share
        self.percentage = percentage
        self.minimum = minimum

    def calculate(self, price: float, quantity: float) -> float:
        qty = abs(quantity)
        commission = price * qty * self.percentage + qty * self.per_share
        return max(commission, self.minimum)


@dataclass
class PortfolioConstraints:
    max_leverage: float = 1.0
    max_position_weight: float = 0.3
    cash_reserve_pct: float = 0.02


class AdvancedBacktestEngine:
    """
    Advanced backtesting engine with portfolio awareness and execution modelling.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        initial_capital: float = 100_000.0,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        constraints: Optional[PortfolioConstraints] = None,
    ):
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        self.constraints = constraints or PortfolioConstraints()

    def run(
        self,
        strategy: Strategy,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        primary_timeframe: str = "1d",
        extra_timeframes: Optional[Sequence[str]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> BacktestResult:
        """
        Execute a backtest for the supplied strategy and symbol universe.
        """
        if not symbols:
            raise ValueError("At least one symbol must be supplied for backtesting.")

        symbols = [symbol.upper() for symbol in symbols]
        extra_timeframes = list(extra_timeframes or [])
        market_data = market_data or self._fetch_market_data(symbols, start, end, primary_timeframe, extra_timeframes)

        primary_aligned, timeline = self._prepare_primary_data(market_data, symbols)
        state = PortfolioState(cash=self.initial_capital, positions={symbol: Position() for symbol in symbols})
        trades: List[BacktestTrade] = []
        portfolio_history: List[Dict[str, float]] = []

        for ts in timeline:
            price_map: Dict[str, float] = {}
            valid_row = True
            for symbol in symbols:
                row = primary_aligned[symbol].loc[ts]
                price = float(row.get("close", np.nan))
                if np.isnan(price):
                    valid_row = False
                    break
                price_map[symbol] = price
            if not valid_row:
                continue

            portfolio_value = state.total_value(price_map)
            context_base = {
                "timestamp": ts,
                "portfolio_value": portfolio_value,
                "cash": state.cash,
                "positions": state.positions,
                "prices": price_map,
            }

            signals: List[Signal] = []
            if hasattr(strategy, "generate_portfolio_signals"):
                context = self._build_portfolio_context(ts, market_data, symbols)
                context.update(context_base)
                raw_signals = strategy.generate_portfolio_signals(context)  # type: ignore[attr-defined]
                signals.extend(self._ensure_signal_list(raw_signals))
            else:
                for symbol in symbols:
                    primary_history = primary_aligned[symbol].loc[:ts].dropna()
                    extra_history = {
                        timeframe: df.loc[:ts].dropna()
                        for timeframe, df in market_data[symbol]["extra"].items()
                    }
                    context = {**context_base, "symbol": symbol, "extra_timeframes": extra_history}
                    try:
                        raw_signal = strategy.generate_signals(primary_history, context=context)
                    except TypeError:
                        raw_signal = strategy.generate_signals(primary_history)
                    signals.extend(self._ensure_signal_list(raw_signal))

            for signal in signals:
                if signal is None or signal.symbol is None:
                    continue
                if signal.side.lower() == "hold":
                    continue
                symbol = signal.symbol.upper()
                if symbol not in symbols or symbol not in price_map:
                    continue
                trade = self._execute_signal(
                    signal=signal,
                    timestamp=ts,
                    price_map=price_map,
                    state=state,
                    portfolio_value=portfolio_value,
                )
                if trade:
                    trades.append(trade)

            # Mark-to-market portfolio after processing signals
            portfolio_value = state.total_value(price_map)
            gross_exposure = sum(state.positions[s].exposure(price_map[s]) for s in symbols)
            portfolio_history.append(
                {
                    "timestamp": ts,
                    "equity": portfolio_value,
                    "cash": state.cash,
                    "market_value": portfolio_value - state.cash,
                    "gross_exposure": gross_exposure,
                }
            )

        if not portfolio_history:
            raise RuntimeError("Backtest produced no portfolio history. Check input data or strategy.")

        history_df = pd.DataFrame(portfolio_history).set_index("timestamp")
        equity_curve = history_df["equity"]
        returns = equity_curve.pct_change().dropna()
        metrics = calculate_performance_metrics(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            portfolio_history=history_df,
            metrics=metrics,
            positions=state.positions,
        )

    def _execute_signal(
        self,
        signal: Signal,
        timestamp: pd.Timestamp,
        price_map: Dict[str, float],
        state: PortfolioState,
        portfolio_value: float,
    ) -> Optional[BacktestTrade]:
        symbol = signal.symbol.upper()
        position = state.positions.setdefault(symbol, Position())
        price = price_map[symbol]

        desired_qty = self._determine_target_quantity(
            signal=signal,
            position=position,
            price=price,
            portfolio_value=portfolio_value,
            price_map=price_map,
        )
        order_qty = desired_qty - position.quantity

        if abs(order_qty) < 1e-8:
            return None

        side = "buy" if order_qty > 0 else "sell"
        execution_price = self.slippage_model.apply(price, side)

        # Enforce cash reserve for buys
        if order_qty > 0:
            available_cash = max(
                0.0,
                state.cash - self.constraints.cash_reserve_pct * portfolio_value,
            )
            if available_cash <= 0:
                return None
            max_qty_cash = available_cash / execution_price
            if order_qty > max_qty_cash:
                order_qty = max_qty_cash
                if order_qty <= 0:
                    return None

        # Enforce leverage limit
        gross_excluding_symbol = sum(
            state.positions[s].exposure(price_map[s]) for s in state.positions if s != symbol
        )
        max_allowable = max(self.constraints.max_leverage * portfolio_value - gross_excluding_symbol, 0.0)
        max_qty_leverage = max_allowable / price if price > 0 else 0.0
        if abs(desired_qty) > max_qty_leverage and max_qty_leverage > 0:
            desired_qty = np.sign(desired_qty) * max_qty_leverage
            order_qty = desired_qty - position.quantity
        if abs(order_qty) < 1e-8:
            return None

        # Recalculate execution price and commission after adjustments
        side = "buy" if order_qty > 0 else "sell"
        execution_price = self.slippage_model.apply(price, side)
        commission = self.commission_model.calculate(execution_price, order_qty)

        if order_qty > 0:
            total_cost = execution_price * order_qty + commission
            if total_cost > state.cash:
                if execution_price <= 0:
                    return None
                affordable_qty = max((state.cash - commission) / execution_price, 0.0)
                order_qty = min(order_qty, affordable_qty)
                if order_qty <= 0:
                    return None
                commission = self.commission_model.calculate(execution_price, order_qty)
                total_cost = execution_price * order_qty + commission
            state.cash -= total_cost
            new_quantity = position.quantity + order_qty
            position.cost_basis = (
                (position.cost_basis * position.quantity + total_cost) / new_quantity
                if new_quantity > 0
                else 0.0
            )
            position.quantity = new_quantity
            realized_pnl = 0.0
        else:
            qty_to_sell = abs(order_qty)
            if qty_to_sell > position.quantity + 1e-8:
                qty_to_sell = position.quantity
                order_qty = -qty_to_sell
            if qty_to_sell <= 0:
                return None
            proceeds = execution_price * qty_to_sell - commission
            realized_pnl = (execution_price - position.cost_basis) * qty_to_sell - commission
            state.cash += proceeds
            position.quantity -= qty_to_sell
            position.realized_pnl += realized_pnl
            if position.quantity <= 1e-8:
                position.quantity = 0.0
                position.cost_basis = 0.0

        portfolio_value_after = state.total_value(price_map)
        trade_value = execution_price * abs(order_qty)
        return BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=float(order_qty),
            price=float(price),
            execution_price=float(execution_price),
            value=float(trade_value),
            commission=float(commission),
            slippage=float(execution_price - price),
            pnl=float(realized_pnl),
            portfolio_value=float(portfolio_value_after),
        )

    def _determine_target_quantity(
        self,
        signal: Signal,
        position: Position,
        price: float,
        portfolio_value: float,
        price_map: Dict[str, float],
    ) -> float:
        desired_qty = position.quantity
        extra = signal.extra or {}

        if signal.quantity is not None:
            desired_qty = signal.quantity
        elif "target_shares" in extra:
            desired_qty = float(extra["target_shares"])
        elif signal.target_weight is not None:
            desired_qty = (signal.target_weight * portfolio_value) / price if price > 0 else 0.0
        elif "target_weight" in extra:
            desired_qty = (float(extra["target_weight"]) * portfolio_value) / price if price > 0 else 0.0
        elif "notional" in extra:
            desired_qty = float(extra["notional"]) / price if price > 0 else 0.0
        else:
            if signal.side.lower() == "buy":
                desired_qty = position.quantity + (portfolio_value * (1.0 - self.constraints.cash_reserve_pct)) / price
            elif signal.side.lower() == "sell":
                desired_qty = 0.0

        # Apply per-position weight constraint
        max_position_value = self.constraints.max_position_weight * portfolio_value
        if max_position_value > 0 and price > 0:
            max_qty = max_position_value / price
            if abs(desired_qty) > max_qty:
                desired_qty = np.sign(desired_qty) * max_qty
        return desired_qty

    def _ensure_signal_list(self, raw_signal: Optional[Iterable[Signal] | Signal]) -> List[Signal]:
        if raw_signal is None:
            return []
        if isinstance(raw_signal, Signal):
            return [raw_signal]
        return list(raw_signal)

    def _fetch_market_data(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        primary_timeframe: str,
        extra_timeframes: Sequence[str],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol in symbols:
            primary = self.data_provider.fetch_price_history(symbol, start=start, end=end, interval=primary_timeframe)
            primary = primary.sort_index()
            extras: Dict[str, pd.DataFrame] = {}
            for timeframe in extra_timeframes:
                extra = self.data_provider.fetch_price_history(symbol, start=start, end=end, interval=timeframe)
                extras[timeframe] = extra.sort_index()
            data[symbol] = {"primary": primary, "extra": extras}
        return data

    def _prepare_primary_data(
        self,
        market_data: Dict[str, Dict[str, pd.DataFrame]],
        symbols: Sequence[str],
    ) -> Tuple[Dict[str, pd.DataFrame], List[pd.Timestamp]]:
        timelines: List[pd.Index] = []
        primary_aligned: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            primary = market_data[symbol]["primary"].copy()
            primary = primary[~primary.index.duplicated(keep="last")]
            timelines.append(primary.index)
            market_data[symbol]["primary"] = primary
        union_index = sorted(set().union(*timelines))
        for symbol in symbols:
            primary = market_data[symbol]["primary"]
            aligned = primary.reindex(union_index).ffill()
            primary_aligned[symbol] = aligned
        close_frames = []
        for symbol in symbols:
            if "close" in primary_aligned[symbol].columns:
                series = primary_aligned[symbol]["close"].copy()
                series.name = symbol
                close_frames.append(series)
        if close_frames:
            close_df = pd.concat(close_frames, axis=1)
            valid_indices = close_df.dropna().index.to_list()
        else:
            valid_indices = list(union_index)
        return primary_aligned, valid_indices

    def _build_portfolio_context(
        self,
        timestamp: pd.Timestamp,
        market_data: Dict[str, Dict[str, pd.DataFrame]],
        symbols: Sequence[str],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        context: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol in symbols:
            primary_history = market_data[symbol]["primary"].loc[:timestamp]
            extras = {tf: df.loc[:timestamp] for tf, df in market_data[symbol]["extra"].items()}
            context[symbol] = {"primary": primary_history, "extra": extras}
        return context


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardBacktester:
    """Coordinates walk-forward analysis using the advanced backtest engine."""

    def __init__(self, engine: AdvancedBacktestEngine):
        self.engine = engine

    def run(
        self,
        strategy_factory,
        windows: Sequence[WalkForwardWindow],
        symbols: Sequence[str],
        primary_timeframe: str = "1d",
        extra_timeframes: Optional[Sequence[str]] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> List[BacktestResult]:
        results: List[BacktestResult] = []
        for window in windows:
            training_data = self.engine._fetch_market_data(
                symbols=symbols,
                start=window.train_start,
                end=window.train_end,
                primary_timeframe=primary_timeframe,
                extra_timeframes=extra_timeframes or [],
            )
            strategy = strategy_factory(training_data, window)
            result = self.engine.run(
                strategy=strategy,
                symbols=symbols,
                start=window.test_start,
                end=window.test_end,
                primary_timeframe=primary_timeframe,
                extra_timeframes=extra_timeframes or [],
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            results.append(result)
        return results
