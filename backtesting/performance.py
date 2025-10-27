"""
Performance metrics for backtesting results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestTrade


def calculate_performance_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: List["BacktestTrade"],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    if equity_curve.empty:
        raise ValueError("Equity curve is empty; cannot compute performance metrics.")

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cumulative_return = float(total_return)
    annualized_return = _annualized_return(equity_curve, periods_per_year)
    sharpe = _sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = _sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_drawdown, calmar = _max_drawdown_and_calmar(equity_curve, annualized_return)
    win_rate, profit_factor, avg_trade, avg_win, avg_loss = _trade_statistics(trades)

    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std(ddof=0) * np.sqrt(periods_per_year) if not downside_returns.empty else 0.0
    volatility = returns.std(ddof=0) * np.sqrt(periods_per_year) if not returns.empty else 0.0
    risk_adjusted = annualized_return - risk_free_rate

    return {
        "total_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(volatility),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "downside_deviation": float(downside_deviation),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade_pnl": avg_trade,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "risk_adjusted_return": risk_adjusted,
    }


def _annualized_return(equity_curve: pd.Series, periods_per_year: int) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def _sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=0)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def _sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_std)


def _max_drawdown_and_calmar(equity_curve: pd.Series, annualized_return: float) -> tuple[float, float]:
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    calmar = 0.0
    if max_drawdown != 0:
        calmar = float(annualized_return / abs(max_drawdown))
    return abs(max_drawdown), calmar


def _trade_statistics(trades: List["BacktestTrade"]) -> tuple[float, float, float, float, float]:
    realized = [trade.pnl for trade in trades if trade.pnl is not None and not np.isnan(trade.pnl) and trade.pnl != 0.0]
    if not realized:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    wins = [pnl for pnl in realized if pnl > 0]
    losses = [abs(pnl) for pnl in realized if pnl < 0]
    win_rate = float(len(wins) / len(realized)) if realized else 0.0
    profit_factor = float(sum(wins) / sum(losses)) if losses else float("inf")
    avg_trade = float(np.mean(realized))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return win_rate, profit_factor, avg_trade, avg_win, -avg_loss
