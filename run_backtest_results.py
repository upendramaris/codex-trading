from pathlib import Path
from run_trading import load_config, run_backtest_mode
import logging

logging.basicConfig(level=logging.INFO)
cfg = load_config(Path("configs/backtest.sample.yaml"))
run_backtest_mode(cfg, logging.getLogger("backtest"))
