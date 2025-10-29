import numpy as np
import pandas as pd


class ATRBreakout:
    """
    ATR-scaled channel breakout strategy with ADX filter and time exit.

    Parameters
    ----------
    channel : int
        Lookback for Donchian channel.
    k : float
        ATR multiplier for breakout buffer.
    hold : int
        Maximum holding period after entry.
    adx_floor : float
        Minimum ADX required to permit new positions.
    """

    def __init__(self, params=None):
        params = params or {}
        self.channel = params.get("channel", 20)
        self.k = params.get("k", 0.75)
        self.hold = params.get("hold", 7)
        self.adx_floor = params.get("adx_floor", 25.0)

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def predict(self, df: pd.DataFrame) -> pd.Series:
        required = {"high", "low", "close", "atr_14", "adx_14"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Missing columns for ATRBreakout: {sorted(missing)}")

        upper = df["high"].rolling(self.channel, min_periods=1).max().shift(1)
        lower = df["low"].rolling(self.channel, min_periods=1).min().shift(1)
        atr = df["atr_14"]
        adx = df["adx_14"]

        long_cond = (df["close"] > (upper + self.k * atr)) & (adx >= self.adx_floor)
        short_cond = (df["close"] < (lower - self.k * atr)) & (adx >= self.adx_floor)

        raw = pd.Series(np.nan, index=df.index)
        raw = raw.mask(long_cond, 1.0)
        raw = raw.mask(short_cond, -1.0)

        signal = raw.ffill(limit=self.hold).fillna(0.0)
        return signal.clip(-1.0, 1.0)

    def position(self, signal: float, risk_ctx: dict) -> float:
        equity = risk_ctx.get("equity", 1.0)
        atr = max(risk_ctx.get("atr", risk_ctx.get("atr_14", 1.0)), 1e-6)
        risk_fraction = risk_ctx.get("risk_fraction", 0.01)
        qty = equity * risk_fraction / atr
        return np.clip(signal, -1, 1) * qty
