"""
Advanced risk controls including drawdown guards and circuit breakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional, Tuple

import pandas as pd

from utils.logger import get_logger


@dataclass
class DrawdownSettings:
    max_total_drawdown: float = 0.20
    max_intraday_drawdown: float = 0.07
    flatten_on_breach: bool = True


@dataclass
class CircuitBreakerSettings:
    price_drop_threshold: float = 0.08  # 8% drop from session reference price
    cool_off_minutes: int = 30
    flatten_on_trigger: bool = True


class RiskController:
    """Monitors equity and price behaviour to halt trading under extreme conditions."""

    def __init__(
        self,
        drawdown_settings: Optional[DrawdownSettings] = None,
        circuit_settings: Optional[CircuitBreakerSettings] = None,
    ):
        self.drawdown_settings = drawdown_settings or DrawdownSettings()
        self.circuit_settings = circuit_settings or CircuitBreakerSettings()
        self.logger = get_logger(self.__class__.__name__)

        self.total_peak: Optional[float] = None
        self.session_peak: Optional[float] = None
        self.session_start: Optional[pd.Timestamp] = None
        self.session_reference_price: Dict[str, float] = {}

        self.trading_halted_until: Optional[pd.Timestamp] = None
        self.halt_reason: Optional[str] = None
        self.flatten_positions: bool = False

    def evaluate(self, symbol: str, timestamp: pd.Timestamp, price: float, equity: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        timestamp = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
        self._reset_session_if_needed(timestamp)
        self._resume_if_ready(timestamp)

        if equity is not None:
            self._update_equity(timestamp, equity)

        self._update_price(symbol, timestamp, price)

        if self.halt_reason is not None:
            return False, self.halt_reason
        return True, None

    # Equity handling -----------------------------------------------------------------
    def _update_equity(self, timestamp: pd.Timestamp, equity: float) -> None:
        if equity <= 0:
            return

        if self.total_peak is None or equity > self.total_peak:
            self.total_peak = equity
        if self.session_peak is None or equity > self.session_peak:
            self.session_peak = equity

        total_drawdown = self._compute_drawdown(equity, self.total_peak)
        if total_drawdown >= self.drawdown_settings.max_total_drawdown:
            self._halt("max_drawdown", timestamp, permanent=True)
            return

        intraday_drawdown = self._compute_drawdown(equity, self.session_peak)
        if intraday_drawdown >= self.drawdown_settings.max_intraday_drawdown:
            self._halt("intraday_drawdown", timestamp)

    # Price handling ------------------------------------------------------------------
    def _update_price(self, symbol: str, timestamp: pd.Timestamp, price: float) -> None:
        if price <= 0:
            return
        session_key = symbol
        ref_price = self.session_reference_price.get(session_key)
        if ref_price is None:
            self.session_reference_price[session_key] = price
            return
        drop = (ref_price - price) / ref_price
        if drop >= self.circuit_settings.price_drop_threshold:
            self._halt("circuit_breaker", timestamp)

    # Session management --------------------------------------------------------------
    def _reset_session_if_needed(self, timestamp: pd.Timestamp) -> None:
        if self.session_start is None or timestamp.normalize() != self.session_start.normalize():
            self.session_start = timestamp.normalize()
            self.session_peak = None
            self.session_reference_price = {}
            self.trading_halted_until = None if self.halt_reason != "max_drawdown" else self.trading_halted_until
            if self.halt_reason not in (None, "max_drawdown"):
                self.logger.info("Resetting intraday risk controls for new session.")
                self.halt_reason = None
                self.flatten_positions = False

    def _resume_if_ready(self, timestamp: pd.Timestamp) -> None:
        if self.halt_reason and self.trading_halted_until:
            if timestamp >= self.trading_halted_until:
                self.logger.info("Risk controller releasing halt for reason %s", self.halt_reason)
                self.halt_reason = None
                self.trading_halted_until = None
                self.flatten_positions = False

    # Helpers ------------------------------------------------------------------------
    def _halt(self, reason: str, timestamp: pd.Timestamp, permanent: bool = False) -> None:
        if self.halt_reason == reason:
            return
        self.halt_reason = reason
        if permanent:
            self.trading_halted_until = None
        else:
            self.trading_halted_until = timestamp + timedelta(minutes=self.circuit_settings.cool_off_minutes)
        self.flatten_positions = (
            (reason == "max_drawdown" and self.drawdown_settings.flatten_on_breach)
            or (reason == "intraday_drawdown" and self.drawdown_settings.flatten_on_breach)
            or (reason == "circuit_breaker" and self.circuit_settings.flatten_on_trigger)
        )
        self.logger.warning("Trading halt triggered due to %s. Halt until %s", reason, self.trading_halted_until)

    @staticmethod
    def _compute_drawdown(current: float, peak: Optional[float]) -> float:
        if not peak or peak <= 0:
            return 0.0
        drawdown = (peak - current) / peak
        return max(0.0, float(drawdown))
