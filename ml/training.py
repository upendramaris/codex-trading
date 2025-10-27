"""
Training utilities for machine learning trading models.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .models import ModelArtifact, ModelWrapper


MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class WalkForwardResult:
    split: int
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    validation_range: Tuple[pd.Timestamp, pd.Timestamp]
    metrics: Dict[str, float]


class TrainingPipeline:
    """
    Coordinates feature preparation, model selection, walk-forward validation, and persistence.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        regression_target: str,
        classification_target: str,
    ):
        self.data = data.sort_index()
        self.feature_columns = feature_columns
        self.regression_target = regression_target
        self.classification_target = classification_target

    def split_datasets(
        self, test_size: float = 0.15, validation_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based split into train, validation, and test sets."""
        if test_size + validation_size >= 1.0:
            raise ValueError("Sum of test_size and validation_size must be less than 1.")
        n = len(self.data)
        if n < 3:
            raise ValueError("Insufficient data for splitting.")

        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - validation_size))

        train = self.data.iloc[:val_start].copy()
        validation = self.data.iloc[val_start:test_start].copy()
        test = self.data.iloc[test_start:].copy()
        return train, validation, test

    def walk_forward_validation(
        self,
        model_builder: Callable[..., ModelWrapper],
        target: str,
        params: Dict[str, Any],
        n_splits: int = 5,
        metric_fn: Optional[MetricFn] = None,
    ) -> List[WalkForwardResult]:
        clean = self.data.dropna(subset=self.feature_columns + [target])
        if len(clean) <= n_splits:
            raise ValueError("Not enough observations for the requested number of walk-forward splits.")

        X = clean[self.feature_columns].values
        y = clean[target].values
        index = clean.index

        splitter = TimeSeriesSplit(n_splits=n_splits)
        results: List[WalkForwardResult] = []
        scores: List[float] = []

        for split_num, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
            model = model_builder(**params)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Filter NaNs specific to each split
            train_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            val_mask = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val)
            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_val[val_mask], y_val[val_mask]

            if len(X_train) == 0 or len(X_val) == 0:
                continue

            model.fit(pd.DataFrame(X_train, columns=self.feature_columns), pd.Series(y_train))
            preds = model.predict(pd.DataFrame(X_val, columns=self.feature_columns))

            if metric_fn is None:
                metric = self._default_metric(target, y_val, preds)
            else:
                metric = metric_fn(y_val, preds)
            scores.append(metric)

            results.append(
                WalkForwardResult(
                    split=split_num,
                    train_range=(index[train_idx[0]], index[train_idx[-1]]),
                    validation_range=(index[val_idx[0]], index[val_idx[-1]]),
                    metrics={"score": float(metric)},
                )
            )

        if not results:
            raise RuntimeError("Walk-forward validation produced no results. Check data sufficiency.")
        return results

    def hyperparameter_search(
        self,
        model_builder: Callable[..., ModelWrapper],
        target: str,
        param_grid: Dict[str, Iterable[Any]],
        n_splits: int = 4,
        metric_fn: Optional[MetricFn] = None,
        greater_is_better: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """Grid search using walk-forward splits."""
        best_score: Optional[float] = None
        best_params: Dict[str, Any] = {}

        for values in itertools.product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            results = self.walk_forward_validation(
                model_builder=model_builder,
                target=target,
                params=params,
                n_splits=n_splits,
                metric_fn=metric_fn,
            )
            avg_score = np.mean([r.metrics["score"] for r in results])
            if best_score is None or (greater_is_better and avg_score > best_score) or (
                not greater_is_better and avg_score < best_score
            ):
                best_score = avg_score
                best_params = params

        if best_score is None:
            raise RuntimeError("Failed to evaluate any hyperparameter combinations.")
        return best_params, float(best_score)

    def train_and_save(
        self,
        model: ModelWrapper,
        target: str,
        output_dir: Path,
    ) -> ModelArtifact:
        """Fit the model on the full dataset (excluding NaNs) and persist to disk."""
        clean = self.data.dropna(subset=self.feature_columns + [target])
        X = clean[self.feature_columns]
        y = clean[target]
        metrics = model.fit(X, y)
        artifact = model.save(output_dir)
        artifact.metrics.update(metrics)
        return artifact

    @staticmethod
    def _default_metric(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Choose accuracy for classification targets and RMSE for regression."""
        if "class" in target or "signal" in target:
            return float(accuracy_score(y_true, y_pred))
        return float(mean_squared_error(y_true, y_pred, squared=False))
