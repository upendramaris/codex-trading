"""
Ensemble machine learning models combining deep learning and tree-based estimators.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from torch import nn
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from .models import ModelWrapper, SequenceDataset


def _create_sequences(features: np.ndarray, targets: np.ndarray, sequence_length: int) -> SequenceDataset:
    if len(features) <= sequence_length:
        raise ValueError("Not enough observations to construct sequences.")
    return SequenceDataset(features=features, targets=targets, sequence_length=sequence_length)


class AttentionLayer(nn.Module):
    """Simple attention layer for sequence outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # outputs: [batch, seq_len, hidden]
        scores = torch.tanh(self.attn(outputs))  # [batch, seq_len, hidden]
        weights = torch.softmax(self.context_vector(scores).squeeze(-1), dim=1)  # [batch, seq_len]
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)  # [batch, hidden]
        return context, weights


class AttentionLSTMNet(nn.Module):
    """LSTM network with attention for regression."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(x)
        context, weights = self.attention(outputs)
        prediction = self.fc(context)
        return prediction.squeeze(-1), weights


@dataclass
class LSTMHyperParams:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 15
    batch_size: int = 64


class AttentionLSTMModel(ModelWrapper):
    """PyTorch Attention LSTM wrapper."""

    def __init__(self, name: str = "attention_lstm", sequence_length: int = 32, hyperparams: Optional[LSTMHyperParams] = None):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.hyperparams = hyperparams or LSTMHyperParams()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[AttentionLSTMNet] = None
        self.input_size: Optional[int] = None
        self.last_attention: Optional[np.ndarray] = None

    def _initialise_model(self, input_size: int) -> None:
        self.model = AttentionLSTMNet(
            input_size=input_size,
            hidden_size=self.hyperparams.hidden_size,
            num_layers=self.hyperparams.num_layers,
            dropout=self.hyperparams.dropout,
        ).to(self.device)
        self.input_size = input_size

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        dataset = _create_sequences(X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32), self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams.batch_size, shuffle=True, drop_last=False)

        self._initialise_model(X.shape[1])
        assert self.model is not None

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams.learning_rate)

        self.model.train()
        losses: List[float] = []
        for _ in range(self.hyperparams.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).squeeze(-1)

                optimizer.zero_grad()
                outputs, _ = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(len(dataloader), 1))

        self._fitted = True
        return {"train_loss": float(np.mean(losses))}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None or self.input_size is None:
            raise RuntimeError("Model not fitted.")
        dataset = _create_sequences(X.to_numpy(dtype=np.float32), np.zeros(len(X), dtype=np.float32), self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams.batch_size, shuffle=False, drop_last=False)

        self.model.eval()
        preds: List[float] = []
        attentions: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                outputs, weights = self.model(batch_x)
                preds.extend(outputs.cpu().numpy().tolist())
                attentions.append(weights.cpu().numpy())
        if attentions:
            self.last_attention = np.concatenate(attentions, axis=0)
        return np.asarray(preds, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def save(self, directory: Path):
        raise NotImplementedError("Saving not implemented for AttentionLSTMModel.")

    @classmethod
    def load(cls, artifact_path: Path):
        raise NotImplementedError("Loading not implemented for AttentionLSTMModel.")


class EnsembleRegressor(ModelWrapper):
    """
    Ensemble regressor combining Attention LSTM, Random Forest, and XGBoost.
    """

    def __init__(
        self,
        name: str = "ensemble_regressor",
        sequence_length: int = 32,
        lstm_param_grid: Optional[List[Dict[str, float]]] = None,
        rf_param_grid: Optional[List[Dict[str, float]]] = None,
        xgb_param_grid: Optional[List[Dict[str, float]]] = None,
        validation_splits: int = 4,
    ):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.validation_splits = validation_splits
        self.lstm_param_grid = lstm_param_grid or [
            {"hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 1e-3},
            {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "learning_rate": 5e-4},
        ]
        self.rf_param_grid = rf_param_grid or [
            {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 4},
            {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3},
        ]
        self.xgb_param_grid = xgb_param_grid or [
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
            {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9},
        ]

        self.lstm_model: Optional[AttentionLSTMModel] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.xgb_model: Optional[XGBRegressor] = None
        self.weights: Optional[np.ndarray] = None
        self.sequence_offset: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        metrics_lstm = self._tune_lstm(X, y)
        metrics_rf = self._tune_random_forest(X, y)
        metrics_xgb = self._tune_xgboost(X, y)

        self.weights = self._calculate_weights(
            [metrics_lstm["rmse"], metrics_rf["rmse"], metrics_xgb["rmse"]]
        )
        self._fitted = True

        combined_metrics = {
            "lstm_rmse": metrics_lstm["rmse"],
            "rf_rmse": metrics_rf["rmse"],
            "xgb_rmse": metrics_xgb["rmse"],
            "lstm_mae": metrics_lstm["mae"],
            "rf_mae": metrics_rf["mae"],
            "xgb_mae": metrics_xgb["mae"],
        }
        return combined_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.weights is None:
            raise RuntimeError("EnsembleRegressor must be fitted before inference.")

        preds = []
        lengths = []

        if self.lstm_model:
            lstm_preds = self.lstm_model.predict(X)
            preds.append(lstm_preds)
            lengths.append(len(lstm_preds))
        if self.rf_model:
            rf_preds = self.rf_model.predict(X)
            preds.append(rf_preds[self.sequence_offset :])
            lengths.append(len(rf_preds) - (self.sequence_offset or 0))
        if self.xgb_model:
            xgb_preds = self.xgb_model.predict(X)
            preds.append(xgb_preds[self.sequence_offset :])
            lengths.append(len(xgb_preds) - (self.sequence_offset or 0))

        min_len = min(lengths)
        stacked = np.vstack([pred[-min_len:] for pred in preds])
        ensemble_pred = np.average(stacked, axis=0, weights=self.weights)
        return ensemble_pred

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def save(self, directory: Path):
        raise NotImplementedError("Saving not implemented for EnsembleRegressor.")

    @classmethod
    def load(cls, artifact_path: Path):
        raise NotImplementedError("Loading not implemented for EnsembleRegressor.")

    def _tune_lstm(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        tscv = TimeSeriesSplit(n_splits=self.validation_splits)
        best_params = None
        best_rmse = np.inf
        best_model: Optional[AttentionLSTMModel] = None

        for params in self.lstm_param_grid:
            rmse_scores = []
            mae_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = AttentionLSTMModel(
                    sequence_length=self.sequence_length,
                    hyperparams=LSTMHyperParams(**params),
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                target = y_val.iloc[self.sequence_length - 1 :].to_numpy()
                rmse_scores.append(float(np.sqrt(mean_squared_error(target, preds))))
                mae_scores.append(mean_absolute_error(target, preds))

            mean_rmse = float(np.mean(rmse_scores))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params = params
                best_model = AttentionLSTMModel(
                    sequence_length=self.sequence_length,
                    hyperparams=LSTMHyperParams(**params),
                )

        assert best_model is not None
        best_model.fit(X, y)
        self.lstm_model = best_model
        self.sequence_offset = self.sequence_length - 1
        preds_full = best_model.predict(X)
        target_full = y.iloc[self.sequence_offset :].to_numpy()
        return {
            "rmse": float(np.sqrt(mean_squared_error(target_full, preds_full))),
            "mae": mean_absolute_error(target_full, preds_full),
        }

    def _tune_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        tscv = TimeSeriesSplit(n_splits=self.validation_splits)
        best_rmse = np.inf
        best_model: Optional[RandomForestRegressor] = None

        for params in self.rf_param_grid:
            rmse_scores = []
            mae_scores = []
            for train_idx, val_idx in tscv.split(X):
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                rmse_scores.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], preds))))
                mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))

            mean_rmse = float(np.mean(rmse_scores))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        assert best_model is not None
        best_model.fit(X, y)
        self.rf_model = best_model
        preds_full = best_model.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, preds_full))),
            "mae": mean_absolute_error(y, preds_full),
        }

    def _tune_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        tscv = TimeSeriesSplit(n_splits=self.validation_splits)
        best_rmse = np.inf
        best_model: Optional[XGBRegressor] = None

        for params in self.xgb_param_grid:
            rmse_scores = []
            mae_scores = []
            for train_idx, val_idx in tscv.split(X):
                model = XGBRegressor(
                    **params,
                    random_state=42,
                    objective="reg:squarederror",
                    tree_method="hist",
                )
                model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
                preds = model.predict(X.iloc[val_idx])
                rmse_scores.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], preds))))
                mae_scores.append(mean_absolute_error(y.iloc[val_idx], preds))

            mean_rmse = float(np.mean(rmse_scores))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = XGBRegressor(
                    **params,
                    random_state=42,
                    objective="reg:squarederror",
                    tree_method="hist",
                )

        assert best_model is not None
        best_model.fit(X, y, verbose=False)
        self.xgb_model = best_model
        preds_full = best_model.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, preds_full))),
            "mae": mean_absolute_error(y, preds_full),
        }

    @staticmethod
    def _calculate_weights(rmses: Iterable[float]) -> np.ndarray:
        rmses = np.array(list(rmses), dtype=float)
        inv = np.where(rmses > 0, 1.0 / rmses, 0.0)
        total = inv.sum()
        if total == 0:
            return np.ones_like(inv) / len(inv)
        return inv / total

    def validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """Walk-forward validation using the fitted ensemble weights."""
        if not self.fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before validation.")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []
        mae_scores = []

        for train_idx, val_idx in tscv.split(X):
            self.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = self.predict(X.iloc[val_idx])
            target = y.iloc[val_idx][self.sequence_offset :]
            rmse_scores.append(float(np.sqrt(mean_squared_error(target, preds))))
            mae_scores.append(mean_absolute_error(target, preds))

        return {
            "rmse": float(np.mean(rmse_scores)),
            "mae": float(np.mean(mae_scores)),
        }
