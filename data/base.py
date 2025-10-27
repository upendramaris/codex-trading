"""
Abstract data provider definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

import pandas as pd


class SupportsDataFrame(Protocol):
    """Simple protocol describing objects convertible to a pandas DataFrame."""

    def to_pandas(self) -> pd.DataFrame:  # pragma: no cover - structural type only
        ...


class DataProvider(ABC):
    """Base interface for all market data providers."""

    @abstractmethod
    def fetch_price_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return historical OHLCV bars for the requested symbol.

        Implementations should normalise column names to ['open', 'high', 'low', 'close', 'volume'].
        """

    @abstractmethod
    def fetch_latest_quote(self, symbol: str) -> pd.Series:
        """Return the latest available quote for a symbol."""
