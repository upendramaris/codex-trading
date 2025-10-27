"""
Data processing utilities for technical indicators and feature engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DataProcessorConfig:
    sma_windows: Sequence[int] = (20, 50)
    ema_windows: Sequence[int] = (12, 26)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    volatility_windows: Sequence[int] = (10, 20, 50)
    volume_profile_bins: int = 20


class DataProcessor:
    """Processes price data into technical indicators and model-ready features."""

    required_columns = {"open", "high", "low", "close", "volume"}

    def __init__(self, config: Optional[DataProcessorConfig] = None):
        self.config = config or DataProcessorConfig()

    def validate(self, df: pd.DataFrame) -> None:
        missing = self.required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, enforce sorting, and forward/back fill missing values."""
        self.validate(df)
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        numeric_cols = frame.select_dtypes(include="number").columns
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
        frame[numeric_cols] = frame[numeric_cols].ffill().bfill()
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        return frame

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append SMA, EMA, RSI, MACD, Bollinger Bands, and ATR indicators."""
        frame = df.copy()
        close = frame["close"]

        for window in self.config.sma_windows:
            frame[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()

        for window in self.config.ema_windows:
            frame[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.config.rsi_period, min_periods=self.config.rsi_period).mean()
        avg_loss = loss.rolling(window=self.config.rsi_period, min_periods=self.config.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        frame["rsi"] = 100 - (100 / (1 + rs))

        ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
        frame["macd"] = ema_fast - ema_slow
        frame["macd_signal"] = frame["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
        frame["macd_hist"] = frame["macd"] - frame["macd_signal"]

        rolling_mean = close.rolling(window=self.config.bollinger_window, min_periods=self.config.bollinger_window).mean()
        rolling_std = close.rolling(window=self.config.bollinger_window, min_periods=self.config.bollinger_window).std()
        frame["bb_mid"] = rolling_mean
        frame["bb_upper"] = rolling_mean + self.config.bollinger_std * rolling_std
        frame["bb_lower"] = rolling_mean - self.config.bollinger_std * rolling_std
        frame["bb_width"] = frame["bb_upper"] - frame["bb_lower"]

        high_low = frame["high"] - frame["low"]
        high_prev_close = (frame["high"] - close.shift()).abs()
        low_prev_close = (frame["low"] - close.shift()).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        frame["atr"] = true_range.rolling(window=self.config.atr_period, min_periods=self.config.atr_period).mean()

        return frame

    def add_price_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add returns, log returns, volatility estimates, and volume z-scores."""
        frame = df.copy()
        frame["returns"] = frame["close"].pct_change()
        frame["log_returns"] = np.log(frame["close"] / frame["close"].shift(1))
        frame["rolling_mean_volume"] = frame["volume"].rolling(window=20, min_periods=20).mean()
        std_volume = frame["volume"].rolling(window=20, min_periods=20).std()
        frame["rolling_std_volume"] = std_volume
        frame["volume_zscore"] = (frame["volume"] - frame["rolling_mean_volume"]) / std_volume.replace(0, np.nan)
        for window in self.config.volatility_windows:
            frame[f"volatility_{window}"] = frame["returns"].rolling(window=window, min_periods=window).std() * np.sqrt(window)
        return frame

    def build_feature_matrix(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Produce a feature matrix combining technical indicators and derived features.
        """
        clean = self.clean_data(df)
        enriched = self.add_technical_indicators(clean)
        enriched = self.add_price_transforms(enriched)

        if len(self.config.sma_windows) >= 2:
            fast, slow = self.config.sma_windows[0], self.config.sma_windows[-1]
            enriched[f"sma_ratio_{fast}_{slow}"] = enriched[f"sma_{fast}"] / enriched[f"sma_{slow}"] - 1

        if len(self.config.ema_windows) >= 2:
            fast, slow = self.config.ema_windows[0], self.config.ema_windows[-1]
            enriched[f"ema_ratio_{fast}_{slow}"] = enriched[f"ema_{fast}"] / enriched[f"ema_{slow}"] - 1

        bb_width = (enriched["bb_upper"] - enriched["bb_lower"]).replace(0, np.nan)
        enriched["price_position_bb"] = (enriched["close"] - enriched["bb_lower"]) / bb_width
        enriched["atr_pct"] = enriched["atr"] / enriched["close"]

        return enriched.dropna() if dropna else enriched

    def build_volume_profile(self, df: pd.DataFrame, bins: Optional[int] = None) -> pd.DataFrame:
        """Aggregate traded volume by price bins for volume profile analysis."""
        frame = self.clean_data(df)
        bins = bins or self.config.volume_profile_bins
        price_bins = pd.cut(frame["close"], bins=bins, include_lowest=True)
        profile = frame.groupby(price_bins)["volume"].sum().to_frame(name="volume")
        profile["price_mid"] = profile.index.map(lambda interval: (interval.left + interval.right) / 2)
        return profile
