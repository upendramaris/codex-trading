"""
Prediction engine that converts model outputs into actionable trading signals.
"""

from __future__ import annotations

import json
from pathlib import Path
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

        model_votes: Dict[str, np.ndarray] = {}
        model_confidences: Dict[str, np.ndarray] = {}
        effective_length = len(frame)
        for name, model in self.models.items():
            subset = frame[self.feature_columns]
            predictions = model.predict(subset)
            if predictions is None or len(predictions) == 0:
                continue
            effective_length = min(effective_length, len(predictions))
            if isinstance(model, RandomForestSignalModel):
                prob = model.predict_proba(subset)
                confidence = np.max(prob, axis=1) if prob is not None else np.ones(len(predictions), dtype=float)
                model_votes[name] = predictions
                model_confidences[name] = confidence
            else:
                model_votes[name] = self._convert_return_to_signal(predictions)

        if not model_votes:
            raise RuntimeError("No models produced usable predictions for inference.")
        if effective_length <= 0:
            raise RuntimeError("Models did not return enough predictions for inference.")

        if effective_length != len(frame):
            frame = frame.tail(effective_length)

        votes: Dict[str, np.ndarray] = {}
        for name, raw_votes in model_votes.items():
            trimmed = raw_votes[-effective_length:]
            if name in model_confidences:
                confidence = model_confidences[name][-effective_length:]
                votes[name] = trimmed * confidence
            else:
                votes[name] = trimmed

        combined_vote = np.sum(list(votes.values()), axis=0)
        signal = np.sign(combined_vote).astype(int)
        contributors = max(len(votes), 1)
        confidence = np.clip(np.abs(combined_vote) / contributors, 0, 1)

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


def load_artifacts_from_directory(engine: PredictionEngine, directory: Path) -> int:
    """Load model artifacts from a directory into the given prediction engine."""
    if not directory.exists():
        return 0

    loaded = 0
    for meta_file in directory.rglob("*.json"):
        try:
            metadata = json.loads(meta_file.read_text())
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        model_type = metadata.get("model_type")
        if not name or not model_type:
            continue

        if model_type == "xgboost":
            if not meta_file.name.endswith("_meta.json"):
                continue
            model_path = meta_file.with_name(meta_file.name.replace("_meta", ""))
        elif model_type == "random_forest":
            model_path = meta_file.with_suffix(".joblib")
        elif model_type == "lstm":
            model_path = meta_file.with_suffix(".pt")
        else:
            continue

        if not model_path.exists():
            continue

        artifact = ModelArtifact(
            name=name,
            model_type=model_type,
            path=model_path,
            params=metadata.get("params", {}),
            metrics=metadata.get("metrics", {}),
        )
        try:
            engine.load_and_register(artifact)
            loaded += 1
        except Exception:
            continue
    return loaded
