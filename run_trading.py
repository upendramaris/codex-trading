from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from backtesting import AdvancedBacktestEngine, CommissionModel, SlippageModel
from brokers import RiskParameters
from data import MarketDataFetcher, YFinanceDataProvider
from ml import FeatureEngineer, PredictionEngine
from ml.prediction import load_artifacts_from_directory
from strategies import MeanReversionStrategy, MomentumStrategy, MovingAverageCrossStrategy, MLSignalStrategy
from trading import (
    AlertService,
    PerformanceTracker,
    TradeJournal,
    TradeLogEntry,
    TradingBot,
    TradingBotConfig,
    TradingDashboard,
    VolatilityAdjustedSizer,
    CircuitBreakerSettings,
    DrawdownSettings,
)
from trading.portfolio import FixedFractionalSizer, KellyCriterionSizer, PositionSizer
from utils.logger import configure_logging, get_logger


def load_config(path: Path) -> Dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def build_risk_parameters(cfg: Dict) -> RiskParameters:
    return RiskParameters(
        max_position_value=cfg.get("max_position_value"),
        max_position_percent=cfg.get("max_position_percent"),
        max_single_order_value=cfg.get("max_single_order_value"),
        max_leverage=cfg.get("max_leverage"),
        max_daily_loss=cfg.get("max_daily_loss"),
        max_symbol_exposure=cfg.get("max_symbol_exposure", {}),
        sector_limits=cfg.get("sector_limits", {}),
        symbol_sectors=cfg.get("symbol_sectors", {}),
    )


def build_position_sizer(cfg: Optional[Dict]) -> PositionSizer:
    cfg = cfg or {}
    sizer_type = (cfg.get("type") or "volatility").lower()
    params = cfg.get("params", {})

    if sizer_type == "fixed_fractional":
        return FixedFractionalSizer(**params)
    if sizer_type == "kelly":
        return KellyCriterionSizer(**params)
    if sizer_type == "volatility":
        return VolatilityAdjustedSizer(**params)
    raise ValueError(f"Unsupported position sizer type: {sizer_type}")


def build_data_fetcher(cfg: Dict) -> MarketDataFetcher:
    return MarketDataFetcher(
        enable_alpaca=cfg.get("enable_alpaca", True),
        enable_yfinance=cfg.get("enable_yfinance", True),
    )


def configure_alert_service(cfg: Dict) -> AlertService:
    alerts_cfg = cfg.get("alerts", {})
    if not alerts_cfg.get("enabled", True):
        # set level high so that notify effectively no-ops except critical
        return AlertService(min_level="critical")
    return AlertService(min_level=alerts_cfg.get("level", "warning"))


def create_trade_entry(payload: Dict, default_reason: str) -> TradeLogEntry:
    timestamp = payload.get("timestamp", pd.Timestamp.utcnow())
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)
    return TradeLogEntry(
        timestamp=timestamp,
        symbol=payload["symbol"],
        side=payload["side"],
        quantity=float(payload.get("quantity", 0.0)),
        price=float(payload.get("price", 0.0)),
        reason=payload.get("reason", default_reason),
        order_id=payload.get("order_id"),
    )


def run_live_or_paper(config: Dict, mode: str, logger: logging.Logger) -> None:
    monitoring_cfg = config.get("monitoring", {})
    alert_service = configure_alert_service(monitoring_cfg)

    risk_cfg = config.get("risk", {})
    risk_controls_cfg = config.get("risk_controls", {})
    drawdown_cfg = risk_controls_cfg.get("drawdown", {})
    circuit_cfg = risk_controls_cfg.get("circuit_breaker", {})

    drawdown_settings = DrawdownSettings(
        max_total_drawdown=drawdown_cfg.get("max_total_drawdown", 0.20),
        max_intraday_drawdown=drawdown_cfg.get("max_intraday_drawdown", 0.07),
        flatten_on_breach=drawdown_cfg.get("flatten_on_breach", True),
    )
    circuit_settings = CircuitBreakerSettings(
        price_drop_threshold=circuit_cfg.get("price_drop_threshold", 0.08),
        cool_off_minutes=circuit_cfg.get("cool_off_minutes", 30),
        flatten_on_trigger=circuit_cfg.get("flatten_on_trigger", True),
    )
    trading_cfg = config.get("trading", {})
    strategy_cfg = trading_cfg.get("strategy", {})
    position_sizer = build_position_sizer(config.get("position_sizer"))
    optimizer_cfg = config.get("optimizer", {})
    optimizer_settings = None
    if optimizer_cfg.get("enabled"):
        optimizer_settings = {k: v for k, v in optimizer_cfg.items() if k != "enabled"}
    bot_config = TradingBotConfig(
        symbols=trading_cfg.get("symbols", []),
        strategy_type=strategy_cfg.get("type", "momentum"),
        strategy_params=strategy_cfg.get("params", {}),
        risk_parameters=build_risk_parameters(risk_cfg),
        run_seconds=trading_cfg.get("run_seconds"),
        historical_lookback_days=trading_cfg.get("historical_lookback_days", 180),
        calendar_name=config.get("calendar", "NYSE"),
        position_sizer=position_sizer,
        prediction_return_threshold=strategy_cfg.get("prediction_return_threshold", 0.0005),
        paper_trading=(mode != "live"),
        drawdown_settings=drawdown_settings,
        circuit_breaker_settings=circuit_settings,
        optimizer_settings=optimizer_settings,
    )

    data_cfg = config.get("data", {})
    data_fetcher = build_data_fetcher(data_cfg)

    bot = TradingBot(config=bot_config, data_fetcher=data_fetcher)
    dashboard = TradingDashboard(bot.broker)
    tracker = PerformanceTracker()
    journal_path = Path(monitoring_cfg.get("journal", "journals/trade_journal.csv"))
    trade_journal = TradeJournal(journal_path)

    def record_equity(snapshot_time: Optional[pd.Timestamp] = None) -> None:
        account = bot.broker.get_account()
        equity = float(account.get("equity", account.get("portfolio_value", 0.0)))
        tracker.record_equity(equity, snapshot_time)
        dashboard.record_equity()

    def handle_event(event_type: str, payload: Dict) -> None:
        if event_type == "order_submitted":
            trade = create_trade_entry(payload, default_reason="model_signal")
            record_equity(payload.get("timestamp"))
            trade_journal.record_trade(trade, confidence=payload.get("confidence"))
        elif event_type == "risk_exit":
            trade = create_trade_entry(payload, default_reason=payload.get("reason", "risk_exit"))
            record_equity(payload.get("timestamp"))
            trade_journal.record_trade(trade, confidence=None, extra={"event": "risk_exit"})
            alert_service.notify("warning", "Risk exit executed", symbol=trade.symbol, reason=trade.reason)
        elif event_type == "signal":
            dashboard.update_performance_metrics({"last_signal_symbol": payload.get("symbol")})
        elif event_type == "market_closed":
            alert_service.notify("info", "Market closed", next_open=payload.get("next_open"))
        elif event_type == "correlation":
            avg_corr = payload.get("average")
            dashboard.update_performance_metrics({"average_correlation": avg_corr})
        elif event_type == "trading_halt":
            alert_service.notify("warning", "Trading halted", reason=payload.get("reason"), timestamp=payload.get("timestamp"))
        elif event_type == "trading_resume":
            alert_service.notify("info", "Trading resumed", timestamp=payload.get("timestamp"), previous_reason=payload.get("previous_reason"))

    bot.on("order_submitted", handle_event)
    bot.on("risk_exit", handle_event)
    bot.on("signal", handle_event)
    bot.on("market_closed", handle_event)
    bot.on("correlation", handle_event)

    try:
        bot.prepare()
        record_equity()
        bot.run()
    except Exception as exc:
        alert_service.notify("critical", f"Fatal trading error: {exc}")
        logger.exception("Fatal trading error: %s", exc)
        raise
    finally:
        record_equity()
        metrics = tracker.summary()
        if metrics:
            dashboard.update_performance_metrics(metrics)
            logger.info("Performance summary: %s", metrics)
        dashboard.create_text_report(bot.trading_loop.trade_log if bot.trading_loop else [])
        fig_path = monitoring_cfg.get("dashboard_image", "monitoring/dashboard.png")
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        figure = dashboard.create_figure(bot.trading_loop.trade_log if bot.trading_loop else [])
        figure.savefig(fig_path)
        logger.info("Dashboard figure saved to %s", fig_path)


def run_backtest_mode(config: Dict, logger: logging.Logger) -> None:
    monitoring_cfg = config.get("monitoring", {})
    alert_service = configure_alert_service(monitoring_cfg)

    trading_cfg = config.get("trading", {})
    strategy_cfg = trading_cfg.get("strategy", {})

    configured_symbols = trading_cfg.get("symbols", [])
    if not configured_symbols:
        raise ValueError("No symbols specified for backtest.")
    symbols: List[str] = [str(symbol).upper() for symbol in configured_symbols]
    primary_symbol = symbols[0]
    if len(symbols) > 1:
        logger.warning("Backtest currently supports a single symbol; proceeding with %s.", primary_symbol)
        symbols = [primary_symbol]
    else:
        symbols = [primary_symbol]

    backtest_cfg = config.get("backtest", {})
    start = pd.Timestamp(backtest_cfg.get("start"))
    end = pd.Timestamp(backtest_cfg.get("end"))
    initial_capital = backtest_cfg.get("initial_capital", 100_000)

    data_cfg = config.get("data", {})
    primary_timeframe = data_cfg.get("primary_timeframe", "1d")
    extra_timeframes = data_cfg.get("extra_timeframes", [])
    if data_cfg.get("enable_yfinance", True):
        data_provider = YFinanceDataProvider()
    else:
        raise ValueError("Backtest requires yfinance data; set enable_yfinance: true in config.")

    slippage_cfg = backtest_cfg.get("slippage", {})
    commission_cfg = backtest_cfg.get("commission", {})

    slippage_model = SlippageModel(
        bps=float(slippage_cfg.get("bps", 0.0)),
        fixed=float(slippage_cfg.get("fixed", 0.0)),
    )
    commission_model = CommissionModel(
        per_share=float(commission_cfg.get("per_share", 0.0)),
        percentage=float(commission_cfg.get("percentage", 0.0)),
        minimum=float(commission_cfg.get("minimum", 0.0)),
    )

    engine = AdvancedBacktestEngine(
        data_provider=data_provider,
        initial_capital=initial_capital,
        slippage_model=slippage_model,
        commission_model=commission_model,
    )

    strategy_type = strategy_cfg.get("type", "momentum").lower()
    params = strategy_cfg.get("params", {})
    market_data_override: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
    if strategy_type == "momentum":
        strategy = MomentumStrategy(symbol=primary_symbol, params=params)
    elif strategy_type == "mean_reversion":
        strategy = MeanReversionStrategy(symbol=primary_symbol, params=params)
    elif strategy_type in {"moving_average", "moving_average_cross"}:
        strategy = MovingAverageCrossStrategy(symbol=primary_symbol, params=params)
    elif strategy_type == "ml":
        feature_engineer = FeatureEngineer(
            forecast_horizon=int(params.get("forecast_horizon", 1)),
            classification_threshold=float(params.get("classification_threshold", 0.002)),
            regime_lookback=int(params.get("regime_lookback", 60)),
        )
        feature_columns: Optional[List[str]] = None
        market_data_override = {}
        for symbol in symbols:
            history = data_provider.fetch_price_history(
                symbol=symbol,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                interval=primary_timeframe,
            )
            if history.empty:
                raise ValueError(f"No price history returned for {symbol} using yfinance.")
            engineered = feature_engineer.engineer(history, symbol=symbol)
            if engineered.features.empty:
                raise ValueError(f"Feature engineering produced no usable rows for {symbol}.")
            feature_columns = (
                engineered.feature_columns
                if feature_columns is None
                else [col for col in feature_columns if col in engineered.feature_columns]
            )
            if not feature_columns:
                raise ValueError("No common feature columns available for ML backtest across symbols.")
            market_data_override[symbol] = {"primary": engineered.features, "extra": {}}
            logger.info("Prepared %d engineered rows for %s.", len(engineered.features), symbol)
        if feature_columns is None:
            raise ValueError("Unable to determine feature columns for ML backtest.")
        return_threshold = float(params.get("return_threshold", 0.001))
        prediction_engine = PredictionEngine(feature_columns=feature_columns, return_threshold=return_threshold)
        artifacts_dir = Path(params.get("artifacts_dir") or config.get("artifacts_dir", "artifacts"))
        loaded_models = load_artifacts_from_directory(prediction_engine, artifacts_dir)
        if loaded_models == 0:
            raise ValueError(
                f"No ML model artifacts found in {artifacts_dir}. Train models before running an ML backtest."
            )
        aligned_columns: Optional[List[str]] = None
        for wrapper in prediction_engine.models.values():
            native_model = getattr(wrapper, "model", None)
            feature_names = getattr(native_model, "feature_names_in_", None)
            if feature_names is None:
                continue
            ordered = [str(name) for name in feature_names]
            missing = [col for col in ordered if col not in feature_columns]
            if missing:
                raise ValueError(
                    f"Model '{wrapper.name}' expects feature columns absent in engineered data: {missing[:5]}"
                )
            aligned_columns = [col for col in ordered if col in feature_columns]
            break
        if aligned_columns:
            feature_columns = aligned_columns
            prediction_engine.feature_columns = aligned_columns
        confidence_threshold = float(params.get("confidence_threshold", 0.0))
        strategy = MLSignalStrategy(
            symbol=primary_symbol,
            prediction_engine=prediction_engine,
            feature_columns=feature_columns,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise ValueError(f"Strategy type '{strategy_type}' is not supported in backtest mode.")

    logger.info("Running backtest for %s from %s to %s", primary_symbol, start.date(), end.date())
    result = engine.run(
        strategy=strategy,
        symbols=symbols,
        start=start,
        end=end,
        primary_timeframe=primary_timeframe,
        extra_timeframes=extra_timeframes,
        market_data=market_data_override,
    )

    logger.info("Backtest completed. Metrics: %s", result.metrics)

    tracker = PerformanceTracker()
    for timestamp, equity in result.equity_curve.items():
        tracker.record_equity(float(equity), timestamp)

    journal_path = Path(monitoring_cfg.get("journal", "journals/backtest_trades.csv"))
    journal = TradeJournal(journal_path)
    for trade in result.trades:
        entry = TradeLogEntry(
            timestamp=trade.timestamp,
            symbol=trade.symbol,
            side=trade.side,
            quantity=trade.quantity,
            price=trade.execution_price,
            reason="backtest",
            order_id=None,
        )
        journal.record_trade(entry, confidence=None, extra={"pnl": trade.pnl})

    alert_service.notify("info", "Backtest finished", metrics=result.metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading execution manager.")
    parser.add_argument("--config", type=str, default="configs/trading.sample.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--mode", type=str, choices=["paper", "live", "backtest"], help="Override mode specified in config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    mode = args.mode or config.get("mode", "paper")
    logging_cfg = config.get("logging", {})
    log_level = getattr(logging, logging_cfg.get("level", "INFO").upper(), logging.INFO)
    configure_logging(logging_cfg.get("file"), level=log_level)
    logger = get_logger("execution")

    logger.info("Starting trading execution in %s mode using config %s", mode, config_path)

    try:
        if mode in {"paper", "live"}:
            run_live_or_paper(config, mode, logger)
        elif mode == "backtest":
            run_backtest_mode(config, logger)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user. Shutting down gracefully.")
    except Exception as exc:
        logger.exception("Unrecoverable error: %s", exc)
        sys.exit(1)
    else:
        logger.info("Execution completed successfully.")


if __name__ == "__main__":
    main()
