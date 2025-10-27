"""
Example paper trading workflow leveraging Alpaca paper API.

This script sets up the broker, portfolio manager, feature pipeline, and real-time
trading loop. A simple rule-based model is registered with the prediction engine
so the example can run without pre-trained artifacts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd

from brokers import AlpacaPaperBroker
from data import MarketDataFetcher
from ml import FeatureEngineer
from ml.models import ModelArtifact, ModelWrapper
from ml.prediction import PredictionEngine
from trading import (
    CorrelationAnalyzer,
    FixedFractionalSizer,
    PaperTradingLoop,
    PortfolioManager,
    RiskManager,
    TradingDashboard,
)
from utils.logger import get_logger


class RuleBasedMomentumModel(ModelWrapper):
    """
    Lightweight rule-based model that issues buy/sell signals based on moving average ratios.
    """

    def __init__(self, name: str = "rule_based_momentum", threshold: float = 0.0):
        super().__init__(name)
        self.threshold = threshold
        self._fitted = True  # model is stateless

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:  # pragma: no cover - no-op
        self._fitted = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        fast = X.filter(like="sma_ratio_").iloc[:, 0] if any(col.startswith("sma_ratio_") for col in X.columns) else X["returns"]
        return np.where(fast > self.threshold, 1, -1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        return None

    def save(self, directory) -> ModelArtifact:  # pragma: no cover - not used in example
        raise NotImplementedError("RuleBasedMomentumModel is not persisted.")

    @classmethod
    def load(cls, artifact_path):  # pragma: no cover - not used in example
        raise NotImplementedError("RuleBasedMomentumModel does not load from disk.")


def bootstrap_feature_columns(symbols, fetcher, feature_engineer):
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    data = fetcher.fetch_historical(symbols, start=start, end=end, timeframe="1h")
    datasets = {}
    for symbol, df in data.items():
        engineered = feature_engineer.engineer(df, symbol)
        datasets[symbol] = engineered
    return datasets


def main() -> None:
    logger = get_logger("paper_trading_example")
    symbols = ["SPY", "AAPL"]
    broker = AlpacaPaperBroker()
    fetcher = MarketDataFetcher()
    feature_engineer = FeatureEngineer(forecast_horizon=1)

    # Bootstrap feature columns using recent historical data.
    engineered_datasets = bootstrap_feature_columns(symbols, fetcher, feature_engineer)
    sample_dataset = next(iter(engineered_datasets.values()))
    feature_columns = sample_dataset.feature_columns

    prediction_engine = PredictionEngine(feature_columns=feature_columns, return_threshold=0.0005)
    prediction_engine.register_model("rule_based_momentum", RuleBasedMomentumModel())

    position_sizer = FixedFractionalSizer(fraction=0.05)
    risk_manager = RiskManager()
    portfolio_manager = PortfolioManager(position_sizer, risk_manager)

    # Seed the trading loop with historical data so indicators have sufficient context.
    trading_loop = PaperTradingLoop(
        broker=broker,
        data_fetcher=fetcher,
        portfolio_manager=portfolio_manager,
        feature_engineer=feature_engineer,
        prediction_engine=prediction_engine,
        symbols=symbols,
        primary_timeframe="1m",
    )
    for symbol, dataset in engineered_datasets.items():
        trading_loop.history[symbol] = dataset.features.tail(500)[["open", "high", "low", "close", "volume"]]

    # Correlation analysis for informational purposes.
    closes = {symbol: ds.features["close"] for symbol, ds in engineered_datasets.items()}
    corr_matrix = CorrelationAnalyzer.correlation_matrix(closes)
    logger.info("Average correlation: %.3f", CorrelationAnalyzer.average_correlation(corr_matrix))

    dashboard = TradingDashboard(broker)

    # Run the live loop for a short demo window (e.g., 120 seconds).
    trading_loop.start(run_for_seconds=120)

    dashboard.record_equity()
    dashboard.update_performance_metrics({"avg_correlation": CorrelationAnalyzer.average_correlation(corr_matrix)})
    dashboard.create_text_report(trading_loop.trade_log)
    fig = dashboard.create_figure(trading_loop.trade_log)
    fig.savefig("paper_trading_dashboard.png")
    logger.info("Dashboard saved to paper_trading_dashboard.png")


if __name__ == "__main__":
    main()
