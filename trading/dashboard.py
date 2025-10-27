"""
Simple monitoring dashboard for live paper trading.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from brokers import Broker
from trading.loop import TradeLogEntry
from utils.logger import get_logger


class TradingDashboard:
    """Generates textual and graphical summaries of the live trading state."""

    def __init__(self, broker: Broker):
        self.broker = broker
        self.performance_metrics: Dict[str, float] = {}
        self.equity_history = pd.Series(dtype=float)
        self.logger = get_logger(self.__class__.__name__)

    def record_equity(self) -> None:
        account = self.broker.get_account()
        timestamp = pd.Timestamp.utcnow()
        equity = float(account.get("equity", account.get("portfolio_value", 0.0)))
        self.equity_history.loc[timestamp] = equity

    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        self.performance_metrics.update(metrics)

    def create_text_report(self, trades: Iterable[TradeLogEntry]) -> str:
        account = self.broker.get_account()
        positions = self.broker.list_positions()
        lines = [
            "==== PAPER TRADING DASHBOARD ====",
            f"Equity: {float(account.get('equity', account.get('portfolio_value', 0.0))):,.2f}",
            f"Cash: {float(account.get('cash', 0.0)) :,.2f}",
            f"Buying Power: {float(account.get('buying_power', 0.0)) :,.2f}",
            "",
            "Open Positions:",
        ]
        if positions:
            for pos in positions.values():
                lines.append(
                    f" - {pos.symbol}: qty={pos.qty:.4f} value={pos.market_value:,.2f} "
                    f"unrealized={pos.unrealized_pl or 0.0:,.2f}"
                )
        else:
            lines.append(" - None")

        lines.append("")
        lines.append("Recent Trades:")
        recent = list(trades)[-5:]
        for trade in recent:
            lines.append(
                f" - {trade.timestamp:%Y-%m-%d %H:%M:%S} {trade.symbol} {trade.side.upper()} "
                f"qty={trade.quantity:.4f} price={trade.price:.2f} reason={trade.reason}"
            )

        if self.performance_metrics:
            lines.append("")
            lines.append("Performance Metrics:")
            for key, value in self.performance_metrics.items():
                lines.append(f" - {key}: {value:.4f}")

        report = "\n".join(lines)
        self.logger.info("\n%s", report)
        return report

    def create_figure(self, trades: Iterable[TradeLogEntry]) -> plt.Figure:
        positions = self.broker.list_positions()
        account = self.broker.get_account()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Positions bar plot
        ax_positions = axes[0, 0]
        if positions:
            symbols = list(positions.keys())
            values = [pos.market_value for pos in positions.values()]
            ax_positions.bar(symbols, values, color="tab:blue")
        ax_positions.set_title("Position Market Value")
        ax_positions.set_ylabel("USD")
        ax_positions.grid(True, axis="y", alpha=0.3)

        # Equity history
        ax_equity = axes[0, 1]
        if not self.equity_history.empty:
            self.equity_history.sort_index().plot(ax=ax_equity, color="tab:green")
        ax_equity.axhline(float(account.get("equity", 0.0)), color="tab:grey", linestyle="--", label="current")
        ax_equity.set_title("Equity History")
        ax_equity.legend()
        ax_equity.grid(True, alpha=0.3)

        # Trade scatter
        ax_trades = axes[1, 0]
        trade_list = list(trades)
        if trade_list:
            times = [trade.timestamp for trade in trade_list]
            prices = [trade.price for trade in trade_list]
            colors = ["tab:green" if trade.side.lower() == "buy" else "tab:red" for trade in trade_list]
            ax_trades.scatter(times, prices, c=colors)
        ax_trades.set_title("Trade Timeline")
        ax_trades.set_xlabel("Time")
        ax_trades.set_ylabel("Price")
        ax_trades.grid(True, alpha=0.3)

        # Performance table
        ax_metrics = axes[1, 1]
        ax_metrics.axis("off")
        if self.performance_metrics:
            metrics_df = pd.DataFrame.from_dict(self.performance_metrics, orient="index", columns=["Value"])
            table = ax_metrics.table(
                cellText=[[f"{v:.4f}"] for v in metrics_df["Value"]],
                rowLabels=metrics_df.index,
                colLabels=["Value"],
                loc="center",
            )
            table.scale(1, 1.5)
            ax_metrics.set_title("Performance Metrics")
        else:
            ax_metrics.text(0.5, 0.5, "No metrics available", ha="center", va="center")

        fig.tight_layout()
        return fig
