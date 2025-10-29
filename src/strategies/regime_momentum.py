import numpy as np
import pandas as pd


class RegimeMomentum:
    """
    Regime-aware momentum strategy.

    Parameters
    ----------
    trend_threshold : float
        Minimum EMA spread required to mark bullish/bearish momentum.
    vol_lookback : int
        Lookback window for realized-volatility terciles.
    ml_gate : float
        Minimum ML score magnitude required to activate signals.
    hedge_bias : float
        Residual bias applied when regime is high volatility (negative to hedge).
    """

    def __init__(self, params=None):
        params = params or {}
        self.trend_threshold = params.get("trend_threshold", 0.0015)
        self.vol_lookback = params.get("vol_lookback", 120)
        self.ml_gate = params.get("ml_gate", 0.2)
        self.hedge_bias = params.get("hedge_bias", -0.2)

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def _classify_regime(self, vol: pd.Series) -> pd.Series:
        window = vol.rolling(self.vol_lookback, min_periods=self.vol_lookback // 2)
        low = window.quantile(1 / 3)
        high = window.quantile(2 / 3)
        regime = pd.Series(0, index=vol.index, dtype=int)
        regime = regime.mask(vol <= low, 0)
        regime = regime.mask((vol > low) & (vol <= high), 1)
        regime = regime.mask(vol > high, 2)
        return regime.fillna(2).astype(int)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        required_cols = {"close", "ema_fast", "ema_slow", "realized_vol_20", "ml_score"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise KeyError(f"Missing columns for RegimeMomentum: {sorted(missing)}")

        regime = self._classify_regime(df["realized_vol_20"])
        ema_spread = df["ema_fast"] - df["ema_slow"]
        ml_score = df["ml_score"]

        signal = pd.Series(0.0, index=df.index)
        calm_normal = regime.isin([0, 1])
        bullish = (ema_spread > self.trend_threshold) & (ml_score >= self.ml_gate)
        bearish = (ema_spread < -self.trend_threshold) & (ml_score <= -self.ml_gate)

        signal = signal.mask(calm_normal & bullish, 1.0)
        signal = signal.mask(calm_normal & bearish, -1.0)
        signal = signal.mask(regime == 2, float(self.hedge_bias))
        return signal.fillna(0.0).clip(-1.0, 1.0)

    def position(self, signal: float, risk_ctx: dict) -> float:
        equity = risk_ctx.get("equity", 1.0)
        vol = max(risk_ctx.get("volatility", 0.02), 1e-6)
        target_vol = risk_ctx.get("target_vol", 0.12)
        exposure = np.clip(signal, -1, 1) * target_vol / vol
        return exposure * equity
