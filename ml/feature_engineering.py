"""
Feature engineering utilities for machine learning trading models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from data.processor import DataProcessor, DataProcessorConfig


@dataclass
class FeatureEngineeringOutput:
    symbol: str
    features: pd.DataFrame
    feature_columns: List[str]
    regression_target: str
    classification_target: str


class FeatureEngineer:
    """Creates enriched feature sets for machine learning models."""

    def __init__(
        self,
        processor: Optional[DataProcessor] = None,
        forecast_horizon: int = 1,
        classification_threshold: float = 0.002,
        regime_lookback: int = 60,
    ):
        if processor is None:
            processor = DataProcessor(
                config=DataProcessorConfig(
                    sma_windows=(10, 20, 50, 100, 200),
                    ema_windows=(12, 26, 50),
                    volatility_windows=(10, 20, 50, 100),
                )
            )
        self.processor = processor
        self.forecast_horizon = forecast_horizon
        self.classification_threshold = classification_threshold
        self.regime_lookback = regime_lookback

    def engineer(self, df: pd.DataFrame, symbol: str) -> FeatureEngineeringOutput:
        """Return engineered feature set for a single symbol."""
        processed = self.processor.build_feature_matrix(df, dropna=False)
        enriched = self._add_price_action_features(processed)
        enriched = self._add_market_regime_features(enriched)
        enriched = self._add_additional_volatility_features(enriched)
        enriched = self._add_targets(enriched)

        enriched["symbol"] = symbol
        enriched = enriched.dropna()

        regression_target = "future_return"
        classification_target = "signal_class"
        exclude_cols = {
            regression_target,
            classification_target,
            "symbol",
            "future_close",
        }
        feature_columns = [col for col in enriched.columns if col not in exclude_cols]

        return FeatureEngineeringOutput(
            symbol=symbol,
            features=enriched,
            feature_columns=feature_columns,
            regression_target=regression_target,
            classification_target=classification_target,
        )

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["body"] = frame["close"] - frame["open"]
        frame["range"] = frame["high"] - frame["low"]
        frame["upper_shadow"] = frame["high"] - frame[["close", "open"]].max(axis=1)
        frame["lower_shadow"] = frame[["close", "open"]].min(axis=1) - frame["low"]
        range_safe = frame["range"].replace(0, np.nan)
        frame["body_ratio"] = frame["body"] / range_safe
        frame["upper_shadow_ratio"] = frame["upper_shadow"] / range_safe
        frame["lower_shadow_ratio"] = frame["lower_shadow"] / range_safe
        frame["gap_up"] = (frame["open"] > frame["close"].shift(1)).astype(int)
        frame["gap_down"] = (frame["open"] < frame["close"].shift(1)).astype(int)
        frame["high_low_ratio"] = frame["high"] / frame["low"] - 1
        frame["close_over_open"] = frame["close"] / frame["open"] - 1
        frame["momentum_3"] = frame["close"] / frame["close"].shift(3) - 1
        frame["momentum_5"] = frame["close"] / frame["close"].shift(5) - 1
        frame["momentum_10"] = frame["close"] / frame["close"].shift(10) - 1

        prev_body = frame["body"].shift(1)
        prev_open = frame["open"].shift(1)
        prev_close = frame["close"].shift(1)

        frame["bullish_engulfing"] = (
            (frame["body"] > 0)
            & (prev_body < 0)
            & (frame["close"] > prev_open)
            & (frame["open"] < prev_close)
        ).astype(int)
        frame["bearish_engulfing"] = (
            (frame["body"] < 0)
            & (prev_body > 0)
            & (frame["close"] < prev_open)
            & (frame["open"] > prev_close)
        ).astype(int)

        small_body = frame["body"].abs() <= frame["range"] * 0.1
        frame["doji"] = small_body.astype(int)
        frame["hammer"] = (
            (frame["body"] > 0)
            & (frame["lower_shadow_ratio"] > 0.6)
            & (frame["upper_shadow_ratio"] < 0.2)
        ).astype(int)
        frame["inverted_hammer"] = (
            (frame["body"] > 0)
            & (frame["upper_shadow_ratio"] > 0.6)
            & (frame["lower_shadow_ratio"] < 0.2)
        ).astype(int)
        frame["shooting_star"] = (
            (frame["body"] < 0)
            & (frame["upper_shadow_ratio"] > 0.6)
            & (frame["lower_shadow_ratio"] < 0.2)
        ).astype(int)
        frame["inside_bar"] = (
            (frame["high"] <= frame["high"].shift(1))
            & (frame["low"] >= frame["low"].shift(1))
        ).astype(int)
        frame["outside_bar"] = (
            (frame["high"] >= frame["high"].shift(1))
            & (frame["low"] <= frame["low"].shift(1))
        ).astype(int)

        return frame

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        if "sma_200" not in frame.columns:
            frame["sma_200"] = frame["close"].rolling(window=200, min_periods=200).mean()
        if "sma_100" not in frame.columns:
            frame["sma_100"] = frame["close"].rolling(window=100, min_periods=100).mean()

        sma200_safe = frame["sma_200"].replace(0, np.nan)
        frame["trend_regime"] = np.where(frame["close"] > frame["sma_200"], 1, -1)
        frame["trend_strength"] = (frame["sma_50"] - frame["sma_200"]) / sma200_safe
        frame["momentum_regime"] = frame["returns"].rolling(window=self.regime_lookback, min_periods=self.regime_lookback).mean()
        frame["volatility_regime"] = frame["volatility_20"] / frame["volatility_50"].replace(0, np.nan)

        high_vol = frame["volatility_regime"] > 1.2
        low_vol = frame["volatility_regime"] < 0.8
        positive_momentum = frame["momentum_regime"] > 0
        negative_momentum = frame["momentum_regime"] < 0

        frame["market_regime"] = np.select(
            [
                positive_momentum & ~high_vol,
                negative_momentum & high_vol,
                positive_momentum & high_vol,
                negative_momentum & ~high_vol,
            ],
            [2, -2, 1, -1],
            default=0,
        )

        frame["rolling_skew"] = frame["returns"].rolling(window=self.regime_lookback, min_periods=self.regime_lookback).apply(
            lambda x: pd.Series(x).skew(), raw=False
        )
        frame["rolling_kurtosis"] = frame["returns"].rolling(
            window=self.regime_lookback, min_periods=self.regime_lookback
        ).apply(lambda x: pd.Series(x).kurt(), raw=False)

        return frame

    def _add_additional_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["intraday_range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
        frame["close_to_high_pct"] = (frame["high"] - frame["close"]) / frame["close"]
        frame["close_to_low_pct"] = (frame["close"] - frame["low"]) / frame["close"]
        frame["volatility_change"] = frame["volatility_20"].pct_change()
        frame["atr_change"] = frame["atr"].pct_change()
        frame["volume_roc_5"] = frame["volume"].pct_change(5)
        frame["volume_roc_10"] = frame["volume"].pct_change(10)
        frame["returns_abs"] = frame["returns"].abs()
        frame["returns_squared"] = frame["returns"] ** 2
        frame["rolling_var_20"] = frame["returns"].rolling(window=20, min_periods=20).var()
        frame["rolling_var_50"] = frame["returns"].rolling(window=50, min_periods=50).var()
        frame["drawdown"] = frame["close"] / frame["close"].cummax() - 1
        frame["rolling_max_drawdown"] = frame["drawdown"].rolling(window=100, min_periods=100).min()
        frame["volume_price_trend"] = ((frame["close"] - frame["close"].shift(1)) / frame["close"].shift(1)) * frame["volume"]
        frame["price_volume_corr"] = (
            frame["returns"]
            .rolling(window=20, min_periods=20)
            .corr(frame["volume"].pct_change())
        )
        return frame

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        future_close = frame["close"].shift(-self.forecast_horizon)
        frame["future_close"] = future_close
        frame["future_return"] = future_close / frame["close"] - 1

        threshold = self.classification_threshold
        frame["signal_class"] = np.select(
            [
                frame["future_return"] > threshold,
                frame["future_return"] < -threshold,
            ],
            [1, -1],
            default=0,
        )
        return frame
