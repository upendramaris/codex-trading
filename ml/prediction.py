"""
Prediction engine that converts model outputs into actionable trading signals.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .models import (
    LSTMReturnModel,
    ModelArtifact,
    ModelWrapper,
    RandomForestSignalModel,
    XGBoostReturnModel,
)


class PredictionEngine:
    """Aggregates multiple model outputs into unified buy/sell/hold signals."""

    def __init__(self, feature_columns: list[str], return_threshold: float = 0.001):
        self.feature_columns = feature_columns
        self.return_threshold = return_threshold
        self.models: Dict[str, ModelWrapper] = {}

    def register_model(self, name: str, model: ModelWrapper) -> None:
        if not model.fitted:
            raise ValueError(f"Model '{name}' must be trained before registration.")
        self.models[name] = model

    def load_and_register(self, artifact: ModelArtifact, model_type: Optional[str] = None) -> None:
        kind = model_type or artifact.model_type
        if kind == "random_forest":
            model = RandomForestSignalModel.load(artifact.path)
        elif kind == "xgboost":
            model = XGBoostReturnModel.load(artifact.path)
        elif kind == "lstm":
            model = LSTMReturnModel.load(artifact.path)
        else:
            raise ValueError(f"Unsupported model type: {kind}")
        self.register_model(model.name, model)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("No models registered. Use register_model or load_and_register first.")
        if not set(self.feature_columns).issubset(data.columns):
            raise ValueError("Input data missing required feature columns.")

        frame = data.copy()
        frame = frame.sort_index()
        frame = frame.dropna(subset=self.feature_columns)
        if frame.empty:
            raise ValueError("No rows available after cleaning for inference.")

        votes: Dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            predictions = model.predict(frame[self.feature_columns])
            if isinstance(model, RandomForestSignalModel):
                prob = model.predict_proba(frame[self.feature_columns])
                confidence = np.max(prob, axis=1) if prob is not None else np.ones_like(predictions, dtype=float)
                votes[name] = predictions * confidence
            else:
                votes[name] = self._convert_return_to_signal(predictions)

        combined_vote = np.sum(list(votes.values()), axis=0)
        signal = np.sign(combined_vote).astype(int)
        confidence = np.clip(np.abs(combined_vote) / max(len(self.models), 1), 0, 1)

        output = frame.copy()
        output["signal"] = signal
        output["confidence"] = confidence
        for model_name, model_votes in votes.items():
            output[f"{model_name}_vote"] = model_votes

        return output[["signal", "confidence"] + [col for col in output.columns if col.endswith("_vote")]]

    def _convert_return_to_signal(self, returns: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(returns, dtype=int)
        signal[returns > self.return_threshold] = 1
        signal[returns < -self.return_threshold] = -1
        return signal
