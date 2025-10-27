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
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(-1):
                data = data.xs(symbol, axis=1, level=-1)
            else:
                data = data.droplevel(0, axis=1)
        data.columns = [str(col).lower() for col in data.columns]
        data = data.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj close": "adj_close",
                "adjclose": "adj_close",
                "adj_close": "adj_close",
                "volume": "volume",
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
