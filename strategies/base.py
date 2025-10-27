"""
Strategy abstractions used across trading engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Signal:
    """Simple signal container."""

    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float = 1.0
    quantity: Optional[float] = None
    target_weight: Optional[float] = None
    extra: Optional[Dict[str, float]] = None


class Strategy(ABC):
    """Defines the interface for all strategies."""

    def __init__(self, symbol: str, params: Optional[Dict[str, float]] = None):
        self.symbol = symbol
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> Signal:
        """Return the next trading signal based on provided market data."""

    def update_params(self, **kwargs: float) -> None:
        """Update strategy parameters."""
        self.params.update(kwargs)
