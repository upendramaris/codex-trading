"""
Utility tools exposed to the agent runtime.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backtesting import AdvancedBacktestEngine
from brokers import AlpacaPaperBroker, Order, RiskControl, RiskParameters
from data import MarketDataFetcher, YFinanceDataProvider
from ml import FeatureEngineer
from strategies.base import Signal, Strategy


def _default_start(days: int = 120) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


def _serialise_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    serialised = frame.copy()
    serialised.index = serialised.index.astype("datetime64[ns]")
    serialised = serialised.reset_index().rename(columns={"index": "timestamp"})
    serialised["timestamp"] = serialised["timestamp"].dt.tz_localize("UTC").dt.isoformat()
    return serialised.to_dict(orient="records")


def market_data(
    symbol: str,
    timeframe: str = "1d",
    features: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "auto",
) -> Dict[str, Any]:
    """
    Fetch market data for the requested symbol and optionally include engineered features.
    """
    fetcher = MarketDataFetcher(enable_alpaca=True, enable_yfinance=True)
    start_dt = pd.Timestamp(start).to_pydatetime() if start else _default_start().replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = pd.Timestamp(end).to_pydatetime() if end else datetime.utcnow()
    history = fetcher.fetch_historical([symbol], start=start_dt, end=end_dt, timeframe=timeframe, source=source)[symbol]
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "bars": _serialise_frame(history),
    }
    if features:
        engineer = FeatureEngineer()
        engineered = engineer.engineer(history, symbol)
        payload["feature_columns"] = engineered.feature_columns
        payload["features"] = _serialise_frame(engineered.features)
    return payload


def backtest(
    strategy_code: str,
    universe: Optional[Iterable[str]] = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    initial_capital: float = 100_000.0,
    primary_timeframe: str = "1d",
    extra_timeframes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Execute an ad-hoc backtest using user supplied strategy code.

    The strategy code must define a `build_strategy(symbol: str) -> Strategy` factory.
    """
    symbols = list(universe or ["SPY"])
    if not symbols:
        raise ValueError("Universe must contain at least one symbol.")
    symbol = symbols[0]
    compiled: Dict[str, Any] = {"Strategy": Strategy, "Signal": Signal, "np": np, "pd": pd}
    exec(strategy_code, compiled)  # nosec - controlled environment for power users
    if "build_strategy" not in compiled:
        raise ValueError("strategy_code must define a build_strategy(symbol: str) -> Strategy function.")
    strategy: Strategy = compiled["build_strategy"](symbol)
    provider = YFinanceDataProvider()
    engine = AdvancedBacktestEngine(data_provider=provider, initial_capital=initial_capital)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.utcnow()
    result = engine.run(
        strategy=strategy,
        symbols=[symbol],
        start=start_ts,
        end=end_ts,
        primary_timeframe=primary_timeframe,
        extra_timeframes=list(extra_timeframes or []),
    )
    trades_payload = [
        {
            "timestamp": trade.timestamp.isoformat(),
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": trade.price,
            "pnl": trade.pnl,
        }
        for trade in result.trades
    ]
    equity_payload = [
        {"timestamp": ts.isoformat(), "equity": float(value)}
        for ts, value in result.equity_curve.items()
    ]
    return {
        "symbol": symbol,
        "metrics": result.metrics,
        "equity_curve": equity_payload,
        "trades": trades_payload,
    }


def exec_order(
    symbol: str,
    side: str,
    quantity: float,
    risk_params: Optional[Dict[str, Any]] = None,
    order_type: str = "market",
    time_in_force: str = "day",
) -> Dict[str, Any]:
    """
    Submit an order through the Alpaca paper broker with optional risk constraints.
    """
    params = RiskParameters(**(risk_params or {}))
    broker = AlpacaPaperBroker(risk_control=RiskControl(params), paper=True)
    order = Order(symbol=symbol.upper(), side=side.lower(), qty=float(quantity), order_type=order_type, time_in_force=time_in_force)
    confirmation = broker.submit_order(order)
    return confirmation


def persist_artifact(
    artifact_type: str,
    content: Any,
    metadata: Optional[Dict[str, Any]] = None,
    directory: Path | str = Path("artifacts") / "agent",
) -> Dict[str, Any]:
    """
    Persist artefacts (prompts, generated code, metrics) for auditing or reuse.
    """
    base_dir = Path(directory) / artifact_type
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ext = "json" if isinstance(content, (dict, list)) else "txt"
    file_path = base_dir / f"{timestamp}.{ext}"
    if ext == "json":
        serialisable = {"content": content, "metadata": metadata or {}}
        file_path.write_text(json.dumps(serialisable, indent=2, default=str))
    else:
        lines: List[str] = []
        if metadata:
            lines.append(json.dumps(metadata, indent=2, default=str))
            lines.append("\n---\n")
        lines.append(str(content))
        file_path.write_text("".join(lines))
    return {"path": str(file_path), "artifact_type": artifact_type, "metadata": metadata or {}}
