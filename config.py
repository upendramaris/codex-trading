"""
Central configuration management for the trading framework.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _load_env(dotenv_path: Optional[Path] = None) -> None:
    """Load environment variables from an optional .env file."""
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _env_required(key: str) -> str:
    try:
        value = os.environ[key]
    except KeyError as exc:  # pragma: no cover - guard clause
        raise ConfigError(f"Missing required environment variable: {key}") from exc
    if not value:
        raise ConfigError(f"Environment variable {key} cannot be empty")
    return value


def _env_json(key: str, default: Optional[Dict] = None) -> Dict:
    raw = os.environ.get(key)
    if not raw:
        return default or {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - guard clause
        raise ConfigError(f"Environment variable {key} must contain valid JSON") from exc


@dataclass
class AlpacaSettings:
    api_key: str = field(default_factory=lambda: _env_required("ALPACA_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env_required("ALPACA_API_SECRET"))
    base_url: str = field(default_factory=lambda: os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))


@dataclass
class DataSourceSettings:
    provider: str = field(default_factory=lambda: os.environ.get("DATA_PROVIDER", "yfinance"))
    params: Dict[str, str] = field(default_factory=lambda: _env_json("DATA_PROVIDER_PARAMS"))


@dataclass
class StrategySettings:
    default_strategy: str = field(default_factory=lambda: os.environ.get("DEFAULT_STRATEGY", "moving_average"))
    parameters: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: _env_json(
            "STRATEGY_PARAMS",
            {
                "moving_average": {"short_window": 10, "long_window": 30},
            },
        )
    )


@dataclass
class RiskSettings:
    max_position_size: float = field(default_factory=lambda: float(os.environ.get("RISK_MAX_POSITION_SIZE", 0.2)))
    max_drawdown: float = field(default_factory=lambda: float(os.environ.get("RISK_MAX_DRAWDOWN", 0.15)))
    stop_loss_pct: float = field(default_factory=lambda: float(os.environ.get("RISK_STOP_LOSS_PCT", 0.05)))
    take_profit_pct: float = field(default_factory=lambda: float(os.environ.get("RISK_TAKE_PROFIT_PCT", 0.1)))


@dataclass
class Settings:
    """Aggregates configuration for the trading framework."""

    alpaca: AlpacaSettings = field(default_factory=AlpacaSettings)
    data: DataSourceSettings = field(default_factory=DataSourceSettings)
    strategies: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)

    @classmethod
    def load(cls, dotenv_path: Optional[Path] = None) -> "Settings":
        _load_env(dotenv_path)
        return cls()

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "alpaca": self.alpaca.__dict__,
            "data": self.data.__dict__,
            "strategies": self.strategies.__dict__,
            "risk": self.risk.__dict__,
        }


# Module-level settings instance for convenience imports.
settings = Settings.load(dotenv_path=Path(".env") if Path(".env").exists() else None)
