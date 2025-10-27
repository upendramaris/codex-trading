"""
Main trading bot orchestrating data pipeline, strategies, risk management, and brokerage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import pandas_market_calendars as mcal

from brokers import AlpacaPaperBroker, RiskControl, RiskParameters
from data import FeatureEngineer, FeatureEngineeringOutput, MarketDataFetcher
from ml import PredictionEngine
from ml.models import ModelWrapper
from strategies import (
    MLSignalStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    PortfolioOptimizationStrategy,
    Signal,
    Strategy,
)
from trading.loop import PaperTradingLoop
from trading.portfolio import (
    CorrelationAnalyzer,
    PortfolioManager,
    PositionSizer,
    RiskManager,
    VolatilityAdjustedSizer,
)
from trading.optimization import PortfolioOptimizer
from trading.risk_controls import CircuitBreakerSettings, DrawdownSettings, RiskController
from utils.logger import get_logger


StrategyHandler = Callable[[str, pd.DataFrame, pd.DataFrame], Optional[Signal]]
EventHandler = Callable[[str, Dict], None]


@dataclass
class TradingBotConfig:
    symbols: List[str]
    strategy_type: str = "momentum"  # momentum | mean_reversion | ml | portfolio
    strategy_params: Dict[str, float] = field(default_factory=dict)
    risk_parameters: RiskParameters = field(default_factory=RiskParameters)
    run_seconds: Optional[int] = None
    historical_lookback_days: int = 180
    calendar_name: str = "NYSE"
    position_sizer: Optional[PositionSizer] = None
    ml_models: Optional[List[ModelWrapper]] = None
    prediction_return_threshold: float = 0.0005
    paper_trading: bool = True
    drawdown_settings: Optional[DrawdownSettings] = None
    circuit_breaker_settings: Optional[CircuitBreakerSettings] = None
    optimizer_settings: Optional[Dict[str, Any]] = None


class TradingBot:
    """High-level coordinator for live/paper trading."""

    def __init__(
        self,
        config: TradingBotConfig,
        feature_engineer: Optional[FeatureEngineer] = None,
        data_fetcher: Optional[MarketDataFetcher] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        broker: Optional[AlpacaPaperBroker] = None,
    ):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        risk_control = RiskControl(config.risk_parameters)
        self.broker = broker or AlpacaPaperBroker(risk_control=risk_control, paper=config.paper_trading)
        self.data_fetcher = data_fetcher or MarketDataFetcher()
        self.feature_engineer = feature_engineer or FeatureEngineer(forecast_horizon=1)
        self.prediction_engine = prediction_engine
        self.position_sizer = config.position_sizer or VolatilityAdjustedSizer()
        self.portfolio_manager = portfolio_manager or PortfolioManager(
            position_sizer=self.position_sizer,
            risk_manager=RiskManager(),
        )
        self.portfolio_optimizer: Optional[PortfolioOptimizer] = None
        self._optimizer_transaction_costs: Optional[Dict[str, float]] = None
        self._optimizer_views: Optional[Dict[str, float]] = None
        self._optimizer_view_confidence: Optional[Dict[str, float]] = None
        if config.optimizer_settings:
            optimizer_kwargs = dict(config.optimizer_settings)
            sector_limits = optimizer_kwargs.pop("sector_limits", None)
            self._optimizer_transaction_costs = optimizer_kwargs.pop("transaction_costs", None)
            self._optimizer_views = optimizer_kwargs.pop("market_views", None)
            self._optimizer_view_confidence = optimizer_kwargs.pop("view_confidence", None)
            allowed_keys = {"ewma_window", "ewma_decay", "hrp_enabled", "bl_tau", "risk_aversion", "annealing_config"}
            filtered_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in allowed_keys}
            self.portfolio_optimizer = PortfolioOptimizer(
                sector_limits=sector_limits or config.risk_parameters.max_symbol_exposure,
                **filtered_kwargs,
            )
        self.risk_controller = RiskController(
            drawdown_settings=config.drawdown_settings,
            circuit_settings=config.circuit_breaker_settings,
        )

        self.market_calendar = mcal.get_calendar(config.calendar_name)
        self.event_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.trading_loop: Optional[PaperTradingLoop] = None
        self.historical_features: Dict[str, pd.DataFrame] = {}
        self.feature_columns: List[str] = []
        self.strategy_handler: Optional[StrategyHandler] = None
        self.portfolio_strategy: Optional[PortfolioOptimizationStrategy] = None
        self._cached_portfolio_weights: Dict[str, float] = {}
        self._optimizer_timestamp: Optional[pd.Timestamp] = None

    def on(self, event_type: str, handler: EventHandler) -> None:
        self.event_handlers[event_type].append(handler)

    def _emit(self, event_type: str, payload: Dict) -> None:
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_type, payload)
            except Exception as exc:  # pragma: no cover - logging guard
                self.logger.exception("Event handler error for %s: %s", event_type, exc)

    def prepare(self) -> None:
        self.logger.info("Preparing trading bot for symbols %s", ", ".join(self.config.symbols))
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=self.config.historical_lookback_days)
        historical = self.data_fetcher.fetch_historical(
            self.config.symbols,
            start=start,
            end=end,
            timeframe="1h",
        )

        engineered_outputs = {}
        for symbol, df in historical.items():
            features = self.feature_engineer.engineer(df, symbol)
            engineered_outputs[symbol] = features
            self.historical_features[symbol] = features.features

        sample_dataset = next(iter(engineered_outputs.values()))
        self.feature_columns = sample_dataset.feature_columns

        if self.prediction_engine is None:
            self.prediction_engine = PredictionEngine(
                feature_columns=self.feature_columns,
                return_threshold=self.config.prediction_return_threshold,
            )
        else:
            self.prediction_engine.feature_columns = self.feature_columns
        if self.config.ml_models:
            for model in self.config.ml_models:
                self.prediction_engine.register_model(model.name, model)

        self.strategy_handler = self._build_strategy_handler(engineered_outputs)

        self.trading_loop = PaperTradingLoop(
            broker=self.broker,
            data_fetcher=self.data_fetcher,
            portfolio_manager=self.portfolio_manager,
            feature_engineer=self.feature_engineer,
            prediction_engine=self.prediction_engine,
            symbols=self.config.symbols,
            primary_timeframe="1m",
            strategy_handler=self.strategy_handler,
            event_dispatcher=self._emit,
            risk_controller=self.risk_controller,
        )

        for symbol, output in engineered_outputs.items():
            # Seed loop history with OHLCV data
            ohlcv_columns = [col for col in ["open", "high", "low", "close", "volume"] if col in output.features.columns]
            if ohlcv_columns:
                self.trading_loop.history[symbol] = output.features[ohlcv_columns].tail(1000)

        closes = {symbol: output.features["close"] for symbol, output in engineered_outputs.items()}
        corr_matrix = CorrelationAnalyzer.correlation_matrix(closes)
        self.portfolio_manager.update_correlation_matrix(corr_matrix)
        self._emit("correlation", {"matrix": corr_matrix, "average": CorrelationAnalyzer.average_correlation(corr_matrix)})

    def run(self) -> None:
        if self.trading_loop is None:
            self.prepare()

        now = pd.Timestamp.utcnow().tz_localize("UTC")
        if not self._is_market_open(now):
            next_open = self._next_market_open(now)
            self.logger.info("Market is currently closed. Next open: %s", next_open)
            self._emit("market_closed", {"timestamp": now, "next_open": next_open})
            return

        self._emit("market_open", {"timestamp": now})
        self.trading_loop.start(run_for_seconds=self.config.run_seconds)
        self._emit("bot_shutdown", {"timestamp": pd.Timestamp.utcnow()})

    def _build_strategy_handler(self, engineered_outputs: Dict[str, FeatureEngineeringOutput]) -> StrategyHandler:
        strategy_type = self.config.strategy_type.lower()
        self.logger.info("Configuring strategy handler: %s", strategy_type)

        strategies: Dict[str, Strategy] = {}
        if strategy_type == "mean_reversion":
            for symbol in self.config.symbols:
                strategies[symbol] = MeanReversionStrategy(symbol=symbol, params=self.config.strategy_params)
        elif strategy_type == "momentum":
            for symbol in self.config.symbols:
                strategies[symbol] = MomentumStrategy(symbol=symbol, params=self.config.strategy_params)
        elif strategy_type == "ml":
            for symbol in self.config.symbols:
                strategies[symbol] = MLSignalStrategy(
                    symbol=symbol,
                    prediction_engine=self.prediction_engine,
                    feature_columns=self.feature_columns,
                    confidence_threshold=float(self.config.strategy_params.get("confidence_threshold", 0.0)),
                )
        elif strategy_type == "portfolio":
            lookback = int(self.config.strategy_params.get("lookback", 60))
            min_weight = float(self.config.strategy_params.get("min_weight", 0.0))
            self.portfolio_strategy = PortfolioOptimizationStrategy(symbols=self.config.symbols, lookback=lookback, min_weight=min_weight)
        else:
            raise ValueError(f"Unsupported strategy type: {self.config.strategy_type}")

        def handler(symbol: str, history: pd.DataFrame, latest_features: pd.DataFrame) -> Optional[Signal]:
            if strategy_type in {"mean_reversion", "momentum", "ml"}:
                strategy = strategies[symbol]
                return strategy.generate_signals(history)
            if strategy_type == "portfolio" and self.portfolio_strategy:
                if not self.trading_loop:
                    return None
                current_time = history.index[-1]
                if self._optimizer_timestamp != current_time:
                    weights = self._run_portfolio_optimizer()
                    self._cached_portfolio_weights = weights
                    self._optimizer_timestamp = current_time
                target_weight = self._cached_portfolio_weights.get(symbol, 0.0)
                positions = self.broker.list_positions()
                position = positions.get(symbol)
                account = self.broker.get_account()
                portfolio_value = float(account.get("portfolio_value", account.get("equity", 0.0)))
                current_weight = float(position.market_value / portfolio_value) if position else 0.0
                side = "buy" if target_weight >= current_weight else "sell"
                confidence = float(abs(target_weight - current_weight))
                return Signal(symbol=symbol, side=side, confidence=confidence, target_weight=target_weight)
            return None

        return handler

    def _run_portfolio_optimizer(self) -> Dict[str, float]:
        if not self.portfolio_optimizer or not self.trading_loop:
            # fallback to risk parity weights
            if self.portfolio_strategy:
                context = {
                    sym: {
                        "primary": self.trading_loop.history.get(sym, pd.DataFrame()),
                        "extra": {},
                    }
                    for sym in self.config.symbols
                }
                signals = self.portfolio_strategy.generate_portfolio_signals(context)
                return {signal.symbol: signal.target_weight or 0.0 for signal in signals}
            return {sym: 1.0 / len(self.config.symbols) for sym in self.config.symbols}

        price_history = pd.DataFrame(
            {
                sym: self.trading_loop.history.get(sym, pd.DataFrame()).get("close", pd.Series(dtype=float))
                for sym in self.config.symbols
            }
        ).dropna()
        if price_history.empty:
            raise ValueError("Insufficient price history for portfolio optimiser.")

        account = self.broker.get_account()
        portfolio_value = float(account.get("portfolio_value", account.get("equity", 0.0)))
        positions = self.broker.list_positions()
        current_weights = pd.Series(
            {
                sym: float(pos.market_value) / portfolio_value
                for sym, pos in positions.items()
            },
            name="current_weight",
        )
        transaction_costs = pd.Series(self._optimizer_transaction_costs or {}, name="transaction_cost")
        sector_map = pd.Series(self.config.risk_parameters.symbol_sectors or {})
        market_views = None
        view_confidence = None
        if self._optimizer_views:
            market_views = pd.Series(self._optimizer_views)
        if self._optimizer_view_confidence:
            view_confidence = pd.Series(self._optimizer_view_confidence)

        try:
            weights = self.portfolio_optimizer.optimise(
                price_history=price_history,
                current_weights=current_weights,
                transaction_costs=transaction_costs,
                sector_map=sector_map,
                market_views=market_views,
                view_confidence=view_confidence,
            )
            return weights.to_dict()
        except ValueError as exc:
            self.logger.warning("Portfolio optimiser fallback due to: %s", exc)
            equal_weight = 1.0 / len(self.config.symbols)
            return {sym: equal_weight for sym in self.config.symbols}

    def _is_market_open(self, timestamp: pd.Timestamp) -> bool:
        timestamp = timestamp.tz_convert("America/New_York")
        session = timestamp.normalize()
        schedule = self.market_calendar.schedule(start=session, end=session)
        if schedule.empty:
            return False
        open_time = schedule.iloc[0]["market_open"].tz_convert("America/New_York")
        close_time = schedule.iloc[0]["market_close"].tz_convert("America/New_York")
        return open_time <= timestamp <= close_time

    def _next_market_open(self, timestamp: pd.Timestamp) -> Optional[pd.Timestamp]:
        timestamp = timestamp.tz_convert("America/New_York")
        schedule = self.market_calendar.schedule(
            start=timestamp.normalize(),
            end=timestamp.normalize() + pd.Timedelta(days=10),
        )
        future = schedule[schedule["market_open"] > timestamp]
        if future.empty:
            return None
        return future.iloc[0]["market_open"]
