"""
Example workflow for fetching and processing market data for multiple equities.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from data import DataProcessor, MarketDataFetcher


def main() -> None:
    fetcher = MarketDataFetcher()
    processor = DataProcessor()
    symbols = ["SPY", "AAPL", "MSFT"]
    end = datetime.utcnow()
    start = end - timedelta(days=365)

    historical = fetcher.fetch_historical(symbols, start=start, end=end, timeframe="1d")

    for symbol, df in historical.items():
        print(f"\n=== {symbol} ===")
        features = processor.build_feature_matrix(df)
        volume_profile = processor.build_volume_profile(df)
        print("Feature sample:")
        print(features.tail(3))
        print("\nVolume profile (first 5 bins):")
        print(volume_profile.head())


if __name__ == "__main__":
    main()
