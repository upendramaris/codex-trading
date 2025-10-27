"""
Yahoo Finance data provider leveraging yfinance.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFinanceDataProvider(DataProvider):
    """Fetches market data using yfinance."""

    def fetch_price_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, threads=True)
        if data.empty:
            raise ValueError(f"No data returned for {symbol}")
        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        return data

    def fetch_latest_quote(self, symbol: str) -> pd.Series:
        ticker = yf.Ticker(symbol)
        info = ticker.history(period="1d")
        if info.empty:
            raise ValueError(f"No quote available for {symbol}")
        return info.iloc[-1].rename(
            {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
