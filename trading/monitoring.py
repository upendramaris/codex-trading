"""
Monitoring utilities covering performance tracking, alerts, and trade journaling.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from backtesting.performance import calculate_performance_metrics
from trading.loop import TradeLogEntry
from utils.logger import get_logger


class PerformanceTracker:
    """Tracks equity and generates performance summaries in real time."""

    def __init__(self) -> None:
        self.equity_history = pd.Series(dtype=float)
        self.logger = get_logger(self.__class__.__name__)

    def record_equity(self, equity: float, timestamp: Optional[pd.Timestamp] = None) -> None:
        ts = (timestamp or pd.Timestamp.utcnow()).tz_localize(None)
        self.equity_history.loc[ts] = float(equity)
        self.logger.debug("Equity snapshot recorded at %s: %.2f", ts, equity)

    def summary(self) -> Dict[str, float]:
        if len(self.equity_history) < 2:
            return {}
        equity_curve = self.equity_history.sort_index()
        returns = equity_curve.pct_change().dropna()
        try:
            metrics = calculate_performance_metrics(equity_curve, returns, trades=[])
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception("Failed to compute performance metrics: %s", exc)
            metrics = {}
        return metrics


class AlertService:
    """Lightweight alerting layer that logs important events and forwards to callbacks."""

    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self, min_level: str = "warning", callbacks: Optional[List[Callable[[Dict], None]]] = None) -> None:
        self.min_level = self.LEVEL_MAP.get(min_level.lower(), logging.WARNING)
        self.callbacks = callbacks or []
        self.logger = get_logger(self.__class__.__name__)

    def notify(self, level: str, message: str, **context: Dict) -> None:
        severity = self.LEVEL_MAP.get(level.lower(), logging.INFO)
        record = {
            "level": level.lower(),
            "message": message,
            "context": context,
            "timestamp": pd.Timestamp.utcnow(),
        }
        if severity >= self.min_level:
            self.logger.log(severity, "%s | context=%s", message, context)
            for callback in self.callbacks:
                try:
                    callback(record)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.exception("Alert callback error: %s", exc)


class TradeJournal:
    """Persists trade events to disk for later analysis."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = [
            "timestamp",
            "symbol",
            "side",
            "quantity",
            "price",
            "reason",
            "order_id",
            "confidence",
            "extra",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.headers)
                writer.writeheader()
        self.logger = get_logger(self.__class__.__name__)

    def record_trade(self, trade: TradeLogEntry, confidence: Optional[float] = None, extra: Optional[Dict] = None) -> None:
        row = {
            "timestamp": trade.timestamp.tz_localize(None) if trade.timestamp.tzinfo else trade.timestamp,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": float(trade.quantity),
            "price": float(trade.price),
            "reason": trade.reason,
            "order_id": trade.order_id or "",
            "confidence": confidence if confidence is not None else "",
            "extra": extra or {},
        }
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.headers)
            writer.writerow(row)
        self.logger.debug("Trade journal entry appended: %s", row)
