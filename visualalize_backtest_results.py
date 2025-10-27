import ast
import pandas as pd
import matplotlib.pyplot as plt

cols = ["timestamp","symbol","side","quantity","price","reason","order_id","confidence","extra"]
trades = pd.read_csv("journals/backtest_trades.csv", header=None, names=cols, skiprows=1)
#trades = pd.read_csv(
#    "journals/backtest_trades.csv",
#    converters={"timestamp": pd.to_datetime,
#                "extra": lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}}
#)
trades["timestamp"] = pd.to_datetime(trades["timestamp"])
trades["extra"] = trades["extra"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {})
trades["pnl"] = trades["extra"].apply(lambda d: d.get("pnl", 0.0))
trades["cum_pnl"] = trades["pnl"].cumsum()

trades.plot(x="timestamp", y="cum_pnl", title="Backtest cumulative P&L")
plt.tight_layout()
plt.show()


cols = ["timestamp","symbol","side","quantity","price","reason","order_id","confidence","extra"]
trades = pd.read_csv("journals/backtest_trades.csv", header=None, names=cols, skiprows=1)

trades["timestamp"] = pd.to_datetime(trades["timestamp"])
trades["extra"] = trades["extra"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {})
trades["pnl"] = trades["extra"].apply(lambda d: d.get("pnl", 0.0))
trades["cum_pnl"] = trades["pnl"].cumsum()
