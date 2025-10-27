"""
Performance metrics for strategies and portfolios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    if equity_curve.empty:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years == 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio."""
    if returns.empty:
        return 0.0
    excess_return = returns - risk_free_rate / periods_per_year
    std = excess_return.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess_return.mean() / std)
