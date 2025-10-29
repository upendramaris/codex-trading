import numpy as np
import pandas as pd

from src.strategies.regime_momentum import RegimeMomentum
from src.strategies.atr_breakout import ATRBreakout
from src.strategies.seasonality_edge import SeasonalityEdge


def _make_index(n=100):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_regime_momentum_predict_shape_and_nan_handling():
    idx = _make_index()
    df = pd.DataFrame(
        {
            "close": np.linspace(300, 310, len(idx)),
            "ema_fast": np.linspace(300, 312, len(idx)),
            "ema_slow": np.linspace(299, 309, len(idx)),
            "realized_vol_20": np.linspace(0.01, 0.03, len(idx)),
            "ml_score": np.linspace(-0.5, 0.5, len(idx)),
        },
        index=idx,
    )
    df.iloc[5, 0] = np.nan
    strat = RegimeMomentum()
    signal = strat.predict(df.fillna(method="ffill"))
    assert signal.shape == (len(idx),)
    assert not signal.isna().any()


def test_atr_breakout_predict_shape_and_nan_handling():
    idx = _make_index()
    df = pd.DataFrame(
        {
            "high": np.linspace(310, 320, len(idx)),
            "low": np.linspace(300, 305, len(idx)),
            "close": np.linspace(305, 315, len(idx)),
            "atr_14": np.linspace(1.0, 1.5, len(idx)),
            "adx_14": np.linspace(10, 40, len(idx)),
        },
        index=idx,
    )
    df.iloc[10, 2] = np.nan
    strat = ATRBreakout()
    signal = strat.predict(df.fillna(method="ffill"))
    assert signal.shape == (len(idx),)
    assert not signal.isna().any()


def test_seasonality_edge_predict_shape_and_nan_handling():
    idx = _make_index()
    df = pd.DataFrame(
        {
            "weekday_sin": np.sin(np.arange(len(idx))),
            "weekday_cos": np.cos(np.arange(len(idx))),
            "turn_of_month": np.where(((np.arange(len(idx)) + 1) % 20) < 3, 1, 0),
            "ret_5": np.linspace(-0.002, 0.002, len(idx)),
            "rsi_14": np.linspace(40, 60, len(idx)),
        },
        index=idx,
    )
    df.iloc[15, 3] = np.nan
    strat = SeasonalityEdge()
    signal = strat.predict(df.fillna(method="ffill"))
    assert signal.shape == (len(idx),)
    assert not signal.isna().any()
