# run_continuous.py
import time
import subprocess

CONFIG = "configs/trading.sample.yaml"
SLEEP_SECONDS = 60  # pause between runs

while True:
    subprocess.run(["python", "run_trading.py", "--config", CONFIG], check=False)
    time.sleep(SLEEP_SECONDS)
