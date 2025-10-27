"""
Visualization helpers for backtesting results.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import BacktestResult, BacktestTrade


def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label="Strategy", color="tab:blue")
    if benchmark is not None and not benchmark.empty:
        benchmark.loc[result.equity_curve.index].plot(ax=ax, label="Benchmark", color="tab:orange")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(result: BacktestResult) -> plt.Figure:
    equity = result.equity_curve
    drawdown = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    drawdown.plot(ax=ax, color="tab:red")
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_trade_distribution(trades: List[BacktestTrade]) -> plt.Figure:
    pnls = [trade.pnl for trade in trades if trade.pnl]
    if not pnls:
        pnls = [0.0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnls, bins=30, color="tab:green", alpha=0.7)
    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_risk_metrics(metrics: Dict[str, float]) -> plt.Figure:
    keys = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"]
    values = [metrics.get(key, 0.0) for key in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(keys, values, color="tab:purple", alpha=0.7)
    ax.set_title("Risk Metrics")
    for index, value in enumerate(values):
        ax.text(value, index, f"{value:.2f}", va="center", ha="left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
