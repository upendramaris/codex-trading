"""
Comprehensive market data fetcher combining yfinance and Alpaca APIs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Dict, Optional, Sequence

import pandas as pd
import yfinance as yf

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models import Bar
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import settings


BarCallback = Callable[[str, pd.Series], None]


class MarketDataFetcher:
    """
    Fetches historical and real-time market data from multiple sources.
    """

    _TIMEFRAME_MAP = {
        "1min": {"alpaca": TimeFrame(1, TimeFrameUnit.Minute), "yfinance": "1m"},
        "5min": {"alpaca": TimeFrame(5, TimeFrameUnit.Minute), "yfinance": "5m"},
        "1h": {"alpaca": TimeFrame(1, TimeFrameUnit.Hour), "yfinance": "60m"},
        "1d": {"alpaca": TimeFrame.Day, "yfinance": "1d"},
    }
    _TIMEFRAME_ALIAS = {"1m": "1min", "5m": "5min", "60m": "1h"}

    def __init__(self, enable_alpaca: bool = True, enable_yfinance: bool = True):
        self.enable_alpaca = enable_alpaca
        self.enable_yfinance = enable_yfinance
        self.__alpaca_client: Optional[StockHistoricalDataClient] = None

    def fetch_historical(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
        source: str = "auto",
        adjustment: str = "raw",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for the requested symbols and timeframe.
        """
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        symbols = [symbol.upper() for symbol in symbols]
        timeframe_key = self._normalise_timeframe(timeframe)

        if source == "alpaca" and not self.enable_alpaca:
            raise ValueError("Alpaca requested but disabled in MarketDataFetcher")
        if source == "yfinance" and not self.enable_yfinance:
            raise ValueError("yfinance requested but disabled in MarketDataFetcher")

        if source == "auto":
            # Use Alpaca for intraday data when available, else yfinance.
            if timeframe_key in {"1min", "5min", "1h"} and self.enable_alpaca:
                source = "alpaca"
            elif self.enable_yfinance:
                source = "yfinance"
            elif self.enable_alpaca:
                source = "alpaca"
            else:
                raise RuntimeError("No data sources enabled")

        if source == "alpaca":
            return self._fetch_historical_alpaca(symbols, start, end, timeframe_key, adjustment)
        return self._fetch_historical_yfinance(symbols, start, end, timeframe_key)

    async def stream_live_data_async(
        self,
        symbols: Sequence[str],
        on_bar: BarCallback,
        run_for_seconds: Optional[float] = None,
    ) -> None:
        """
        Stream live bar data via websockets using Alpaca.
        """
        if not self.enable_alpaca:
            raise RuntimeError("Real-time streaming requires Alpaca support")
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        stream = StockDataStream(api_key=settings.alpaca.api_key, secret_key=settings.alpaca.api_secret)

        async def _bar_handler(bar: Bar) -> None:
            timestamp = pd.Timestamp(bar.timestamp)
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert(None)
            series = pd.Series(
                {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "trade_count": float(getattr(bar, "trade_count", 0)),
                    "vwap": float(getattr(bar, "vwap", float("nan"))),
                    "timestamp": timestamp,
                }
            )
            on_bar(bar.symbol, series)

        for symbol in symbols:
            stream.subscribe_bars(_bar_handler, symbol.upper())

        if run_for_seconds is not None:
            async def _stop_after_delay() -> None:
                await asyncio.sleep(run_for_seconds)
                await stream.stop()

            asyncio.create_task(_stop_after_delay())

        await stream.run()

    def stream_live_data(
        self,
        symbols: Sequence[str],
        on_bar: BarCallback,
        run_for_seconds: Optional[float] = None,
    ) -> Optional[asyncio.Task]:
        """
        Convenience wrapper that runs the websocket stream.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.stream_live_data_async(symbols, on_bar, run_for_seconds))
            return None

        return loop.create_task(self.stream_live_data_async(symbols, on_bar, run_for_seconds))

    def _fetch_historical_alpaca(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe_key: str,
        adjustment: str,
    ) -> Dict[str, pd.DataFrame]:
        request = StockBarsRequest(
            symbol_or_symbols=list(dict.fromkeys(symbols)),
            timeframe=self._TIMEFRAME_MAP[timeframe_key]["alpaca"],
            start=start,
            end=end,
            adjustment=adjustment,
            feed=DataFeed.IEX,
        )
        dataset = self._alpaca_client.get_stock_bars(request)
        if dataset.df.empty:
            raise ValueError("No data returned by Alpaca")

        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol not in dataset.df.index.get_level_values("symbol"):
                continue
            frame = dataset.df.xs(symbol, level="symbol").copy()
            frame.index = pd.to_datetime(frame.index).tz_convert(None)
            frame = frame.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "vwap": "vwap",
                    "trade_count": "trade_count",
                }
            )
            results[symbol] = frame.sort_index()
        return results

    def _fetch_historical_yfinance(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        timeframe_key: str,
    ) -> Dict[str, pd.DataFrame]:
        interval = self._TIMEFRAME_MAP[timeframe_key]["yfinance"]
        data = yf.download(
            tickers=" ".join(symbols),
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        if data.empty:
            raise ValueError("No data returned by yfinance")

        results: Dict[str, pd.DataFrame] = {}

        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol not in data.columns.levels[0]:
                    continue
                subset = data[symbol].copy()
                results[symbol] = self._normalise_yfinance_dataframe(subset)
        else:
            results[symbols[0]] = self._normalise_yfinance_dataframe(data)
        return results

    @staticmethod
    def _normalise_yfinance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        return frame.sort_index()

    def _normalise_timeframe(self, timeframe: str) -> str:
        key = timeframe.lower()
        key = self._TIMEFRAME_ALIAS.get(key, key)
        if key not in self._TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return key

    @property
    def _alpaca_client(self) -> StockHistoricalDataClient:
        if getattr(self, "__alpaca_client", None) is None:
            self.__alpaca_client = StockHistoricalDataClient(settings.alpaca.api_key, settings.alpaca.api_secret)
        return self.__alpaca_client
