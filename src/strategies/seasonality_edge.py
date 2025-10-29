import numpy as np
import pandas as pd


class SeasonalityEdge:
    """
    Seasonality and momentum confirmation strategy.

    Parameters
    ----------
    seasonal_weight : float
        Blend between weekday (cosine encoding) and turn-of-month features.
    momentum_floor : float
        Minimum 5-bar return required to confirm long.
    momentum_cap : float
        Maximum 5-bar return allowed to confirm short (negative threshold).
    rsi_band : tuple
        RSI bounds to exit or suppress trades.
    stop_atr_mult : float
        Not used in predict; provided for external position sizing.
    """

    def __init__(self, params=None):
        params = params or {}
        self.seasonal_weight = params.get("seasonal_weight", 0.6)
        self.momentum_floor = params.get("momentum_floor", 0.001)
        self.momentum_cap = params.get("momentum_cap", -0.001)
        self.rsi_band = params.get("rsi_band", (55, 45))
        self.stop_atr_mult = params.get("stop_atr_mult", 1.5)

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def _seasonal_score(self, df: pd.DataFrame) -> pd.Series:
        weekday_angle = np.arctan2(df["weekday_sin"], df["weekday_cos"])
        weekday_component = np.cos(weekday_angle)
        turn_component = df["turn_of_month"].astype(float)
        base = self.seasonal_weight * weekday_component + (1 - self.seasonal_weight) * turn_component
        return base.rolling(20, min_periods=5).mean().fillna(base)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        needed = {"weekday_sin", "weekday_cos", "turn_of_month", "ret_5", "rsi_14"}
        missing = needed.difference(df.columns)
        if missing:
            raise KeyError(f"Missing columns for SeasonalityEdge: {sorted(missing)}")

        score = self._seasonal_score(df)
        momentum = df["ret_5"].fillna(0.0)

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask((score > 0) & (momentum >= self.momentum_floor), 1.0)
        signal = signal.mask((score < 0) & (momentum <= self.momentum_cap), -1.0)

        upper, lower = self.rsi_band
        exit_mask = (df["rsi_14"] >= upper) | (df["rsi_14"] <= lower)
        signal = signal.where(~exit_mask, 0.0)
        return signal.fillna(0.0).clip(-1.0, 1.0)

    def position(self, signal: float, risk_ctx: dict) -> float:
        equity = risk_ctx.get("equity", 1.0)
        fraction = risk_ctx.get("seasonal_fraction", 0.03)
        return np.clip(signal, -1, 1) * equity * fraction
