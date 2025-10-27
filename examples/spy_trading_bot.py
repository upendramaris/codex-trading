"""
End-to-end example showing how to launch the TradingBot for SPY momentum trading.
"""

from __future__ import annotations

from brokers import RiskParameters
from trading import TradingBot, TradingBotConfig, TradingDashboard, VolatilityAdjustedSizer
from utils.logger import get_logger


def main() -> None:
    logger = get_logger("spy_trading_bot")

    config = TradingBotConfig(
        symbols=["SPY"],
        strategy_type="momentum",
        risk_parameters=RiskParameters(
            max_position_percent=0.3,
            max_single_order_value=150_000,
            max_leverage=1.5,
            max_daily_loss=5_000,
        ),
        run_seconds=120,  # demo run window
        position_sizer=VolatilityAdjustedSizer(risk_fraction=0.02),
    )

    bot = TradingBot(config=config)
    dashboard = TradingDashboard(bot.broker)

    def handle_events(event_type: str, payload: dict) -> None:
        if event_type == "order_submitted":
            logger.info(
                "Order submitted: %s %s qty=%.4f price=%.2f",
                payload["symbol"],
                payload["side"],
                payload["quantity"],
                payload["price"],
            )
        elif event_type == "risk_exit":
            logger.info("Risk exit -> %s", payload)
        elif event_type == "market_closed":
            logger.info("Market closed. Next open: %s", payload.get("next_open"))

    bot.on("order_submitted", handle_events)
    bot.on("risk_exit", handle_events)
    bot.on("market_closed", handle_events)

    bot.prepare()
    bot.run()

    dashboard.record_equity()
    dashboard.update_performance_metrics({"sessions_completed": 1})
    dashboard.create_text_report(bot.trading_loop.trade_log if bot.trading_loop else [])
    fig = dashboard.create_figure(bot.trading_loop.trade_log if bot.trading_loop else [])
    fig.savefig("spy_trading_bot_dashboard.png")
    logger.info("Dashboard saved to spy_trading_bot_dashboard.png")


if __name__ == "__main__":
    main()
