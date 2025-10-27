"""
Data package providing abstractions and concrete implementations for market data feeds.
"""

from .base import DataProvider
from .market_data import MarketDataFetcher
from .processor import DataProcessor
from .yfinance_provider import YFinanceDataProvider

__all__ = ["DataProvider", "YFinanceDataProvider", "MarketDataFetcher", "DataProcessor"]
