"""
Model architectures and wrappers for the trading ML stack.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor


@dataclass
class ModelArtifact:
    """Metadata persisted alongside trained models."""

    name: str
    model_type: str
    path: Path
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


class ModelWrapper(ABC):
    """Common interface for training, inference, and persistence."""

    def __init__(self, name: str):
        self.name = name
        self._fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        ...

    @abstractmethod
    def save(self, directory: Path) -> ModelArtifact:
        ...

    @classmethod
    @abstractmethod
    def load(cls, artifact_path: Path) -> "ModelWrapper":
        ...

    @property
    def fitted(self) -> bool:
        return self._fitted


class RandomForestSignalModel(ModelWrapper):
    """Random forest classifier that outputs buy/sell/hold probabilities."""

    def __init__(self, name: str = "random_forest_signal", **params: Any):
        super().__init__(name)
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
        self.params = default_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False, stratify=None
        )
        self.model.fit(X_train, y_train)
        val_preds = self.model.predict(X_val)
        accuracy = (val_preds == y_val).mean()
        self._fitted = True
        return {"accuracy": float(accuracy)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        self._check_fitted()
        return self.model.predict_proba(X)

    def save(self, directory: Path) -> ModelArtifact:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.name}.joblib"
        meta_path = directory / f"{self.name}.json"
        joblib.dump(self.model, model_path)
        metadata = {"name": self.name, "model_type": "random_forest", "params": self.params}
        meta_path.write_text(json.dumps(metadata, indent=2))
        return ModelArtifact(name=self.name, model_type="random_forest", path=model_path, params=self.params)

    @classmethod
    def load(cls, artifact_path: Path) -> "RandomForestSignalModel":
        metadata_path = artifact_path.with_suffix(".json")
        params: Dict[str, Any] = {}
        if metadata_path.exists():
            params = json.loads(metadata_path.read_text()).get("params", {})
        instance = cls(**params)
        instance.model = joblib.load(artifact_path)
        instance._fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RandomForestSignalModel must be trained before inference.")


class XGBoostReturnModel(ModelWrapper):
    """XGBoost regressor targeting forward returns."""

    def __init__(self, name: str = "xgboost_return", **params: Any):
        super().__init__(name)
        default_params = {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        default_params.update(params)
        self.model = XGBRegressor(**default_params)
        self.params = default_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = self.model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        self._fitted = True
        return {"rmse": float(rmse)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def save(self, directory: Path) -> ModelArtifact:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.name}.json"
        meta_path = directory / f"{self.name}_meta.json"
        self.model.save_model(model_path)
        metadata = {"name": self.name, "model_type": "xgboost", "params": self.params}
        meta_path.write_text(json.dumps(metadata, indent=2))
        return ModelArtifact(name=self.name, model_type="xgboost", path=model_path, params=self.params)

    @classmethod
    def load(cls, artifact_path: Path) -> "XGBoostReturnModel":
        metadata_path = artifact_path.with_name(artifact_path.stem + "_meta.json")
        params: Dict[str, Any] = {}
        if metadata_path.exists():
            params = json.loads(metadata_path.read_text()).get("params", {})
        instance = cls(**params)
        instance.model.load_model(artifact_path)
        instance._fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("XGBoostReturnModel must be trained before inference.")


class SequenceDataset(Dataset):
    """torch Dataset for sliding window sequence modelling."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, sequence_length: int):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.indices = list(self._build_indices(len(features), sequence_length))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = self.indices[idx]
        x = torch.tensor(self.features[start:end], dtype=torch.float32)
        y = torch.tensor(self.targets[end - 1], dtype=torch.float32)
        return x, y.unsqueeze(0)

    @staticmethod
    def _build_indices(length: int, sequence_length: int) -> Iterable[Tuple[int, int]]:
        if length < sequence_length:
            return []
        return ((i, i + sequence_length) for i in range(length - sequence_length + 1))


class LSTMRegressorNet(nn.Module):
    """Simple stacked LSTM for return prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])


class LSTMReturnModel(ModelWrapper):
    """PyTorch LSTM wrapper for sequence-based return forecasting."""

    def __init__(
        self,
        name: str = "lstm_return",
        sequence_length: int = 32,
        epochs: int = 15,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model: Optional[LSTMRegressorNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        features = X.to_numpy(dtype=np.float32)
        targets = y.to_numpy(dtype=np.float32)
        dataset = SequenceDataset(features, targets, self.sequence_length)
        if len(dataset) == 0:
            raise ValueError("Not enough observations to build sequences for LSTM training.")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        input_size = features.shape[1]
        self.input_size = input_size
        self.model = LSTMRegressorNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        epoch_losses: list[float] = []
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_losses.append(epoch_loss / max(len(dataloader), 1))

        self._fitted = True
        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        return {"train_loss": mean_loss}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        self.model.eval()
        features = X.to_numpy(dtype=np.float32)
        dataset = SequenceDataset(features, np.zeros(len(features), dtype=np.float32), self.sequence_length)
        if len(dataset) == 0:
            return np.array([])
        preds: list[float] = []
        with torch.no_grad():
            for batch_x, _ in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds.extend(outputs.cpu().numpy().flatten())
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def save(self, directory: Path) -> ModelArtifact:
        if self.model is None or self.input_size is None:
            raise RuntimeError("Cannot save an uninitialised LSTM model.")
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.name}.pt"
        metadata_path = directory / f"{self.name}.json"
        torch.save(self.model.state_dict(), model_path)
        metadata = {
            "name": self.name,
            "model_type": "lstm",
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "input_size": self.input_size,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return ModelArtifact(name=self.name, model_type="lstm", path=model_path, params=metadata)

    @classmethod
    def load(cls, artifact_path: Path) -> "LSTMReturnModel":
        metadata_path = artifact_path.with_suffix(".json")
        kwargs: Dict[str, Any] = {}
        if metadata_path.exists():
            kwargs = json.loads(metadata_path.read_text())
        instance = cls(
            name=kwargs.get("name", "lstm_return"),
            sequence_length=kwargs.get("sequence_length", 32),
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.2),
        )
        state_dict = torch.load(artifact_path, map_location=instance.device)
        dummy_input_size = kwargs.get("input_size")
        if dummy_input_size is None:
            raise RuntimeError("Metadata missing input_size for LSTM model reload.")
        instance.model = LSTMRegressorNet(
            input_size=dummy_input_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
        ).to(instance.device)
        instance.model.load_state_dict(state_dict)
        instance.input_size = dummy_input_size
        instance._fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self._fitted or self.model is None:
            raise RuntimeError("LSTMReturnModel must be trained before inference.")
