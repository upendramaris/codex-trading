"""
Example workflow for training multiple ML models and generating trading signals.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from data import MarketDataFetcher
from ml import (
    FeatureEngineer,
    PredictionEngine,
    RandomForestSignalModel,
    TrainingPipeline,
    XGBoostReturnModel,
    LSTMReturnModel,
)


def main() -> None:
    symbols = ["SPY"]
    fetcher = MarketDataFetcher()
    end = datetime.utcnow()
    start = end - timedelta(days=730)

    historical = fetcher.fetch_historical(symbols, start=start, end=end, timeframe="1d")
    feature_engineer = FeatureEngineer(forecast_horizon=1)

    datasets = [feature_engineer.engineer(frame, symbol) for symbol, frame in historical.items()]
    dataset = datasets[0]
    features_df = dataset.features

    pipeline = TrainingPipeline(
        data=features_df,
        feature_columns=dataset.feature_columns,
        regression_target=dataset.regression_target,
        classification_target=dataset.classification_target,
    )

    rf_grid = {"n_estimators": [200, 400], "max_depth": [4, 6], "min_samples_leaf": [4, 6]}
    best_rf_params, rf_score = pipeline.hyperparameter_search(
        model_builder=RandomForestSignalModel,
        target=dataset.classification_target,
        param_grid=rf_grid,
        n_splits=4,
        greater_is_better=True,
    )
    print(f"Best RF params: {best_rf_params}, score: {rf_score:.4f}")

    rf_model = RandomForestSignalModel(**best_rf_params)
    rf_artifact = pipeline.train_and_save(rf_model, dataset.classification_target, Path("artifacts/rf"))

    xgb_grid = {"max_depth": [3, 4], "learning_rate": [0.05, 0.1], "subsample": [0.7, 0.9]}
    best_xgb_params, xgb_score = pipeline.hyperparameter_search(
        model_builder=XGBoostReturnModel,
        target=dataset.regression_target,
        param_grid=xgb_grid,
        n_splits=4,
        metric_fn=None,
        greater_is_better=False,
    )
    print(f"Best XGBoost params: {best_xgb_params}, score: {xgb_score:.5f}")

    xgb_model = XGBoostReturnModel(**best_xgb_params)
    xgb_artifact = pipeline.train_and_save(xgb_model, dataset.regression_target, Path("artifacts/xgb"))

    lstm_model = LSTMReturnModel(sequence_length=32, epochs=10, hidden_size=64)
    lstm_artifact = pipeline.train_and_save(lstm_model, dataset.regression_target, Path("artifacts/lstm"))

    engine = PredictionEngine(feature_columns=dataset.feature_columns, return_threshold=0.0005)
    engine.load_and_register(rf_artifact)
    engine.load_and_register(xgb_artifact)
    engine.load_and_register(lstm_artifact)

    recent_window = features_df.tail(256)
    signals = engine.generate_signals(recent_window[dataset.feature_columns])
    print(signals.tail())


if __name__ == "__main__":
    main()
