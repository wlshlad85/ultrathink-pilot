#!/usr/bin/env python3
"""
Historical market data fetcher using yfinance.
Provides clean OHLCV data with technical indicators for backtesting.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and prepare historical market data for backtesting."""

    def __init__(self, symbol: str = "BTC-USD"):
        self.symbol = symbol
        self.data = None

    def fetch(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from yfinance.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {self.symbol} data from {start_date} to {end_date}")

        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {self.symbol} in date range")

        # Clean column names
        df.columns = [col.lower() for col in df.columns]

        self.data = df
        logger.info(f"Fetched {len(df)} data points")
        return df

    def add_technical_indicators(self) -> pd.DataFrame:
        """Add common technical indicators to the data."""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch() first.")

        df = self.data.copy()

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['close'], period=14)

        # ATR (Average True Range) - important for stop-loss
        df['atr_14'] = self._calculate_atr(df, period=14)

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price momentum
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)

        self.data = df
        logger.info(f"Added technical indicators")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def get_market_context(self, date_index: int) -> dict:
        """
        Get market context for a specific date to pass to agents.

        Args:
            date_index: Index position in the dataframe

        Returns:
            Dictionary with market data and indicators
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if date_index < 0 or date_index >= len(self.data):
            raise IndexError(f"Invalid date index: {date_index}")

        row = self.data.iloc[date_index]

        # Determine sentiment based on indicators
        sentiment = "neutral"
        if pd.notna(row.get('rsi_14')):
            if row['rsi_14'] > 70:
                sentiment = "overbought"
            elif row['rsi_14'] < 30:
                sentiment = "oversold"
            elif row.get('sma_20', 0) > row.get('sma_50', 0):
                sentiment = "bullish"
            elif row.get('sma_20', 0) < row.get('sma_50', 0):
                sentiment = "bearish"

        context = {
            "date": str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
            "price": float(row['close']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "volume": float(row['volume']),
            "indicators": {
                "rsi_14": float(row.get('rsi_14', 0)) if pd.notna(row.get('rsi_14')) else None,
                "atr_14": float(row.get('atr_14', 0)) if pd.notna(row.get('atr_14')) else None,
                "sma_20": float(row.get('sma_20', 0)) if pd.notna(row.get('sma_20')) else None,
                "sma_50": float(row.get('sma_50', 0)) if pd.notna(row.get('sma_50')) else None,
                "macd": float(row.get('macd', 0)) if pd.notna(row.get('macd')) else None,
                "macd_signal": float(row.get('macd_signal', 0)) if pd.notna(row.get('macd_signal')) else None,
                "bb_upper": float(row.get('bb_upper', 0)) if pd.notna(row.get('bb_upper')) else None,
                "bb_lower": float(row.get('bb_lower', 0)) if pd.notna(row.get('bb_lower')) else None,
                "volume_ratio": float(row.get('volume_ratio', 1)) if pd.notna(row.get('volume_ratio')) else None,
            },
            "returns": {
                "1d": float(row.get('returns_1d', 0)) if pd.notna(row.get('returns_1d')) else None,
                "5d": float(row.get('returns_5d', 0)) if pd.notna(row.get('returns_5d')) else None,
                "20d": float(row.get('returns_20d', 0)) if pd.notna(row.get('returns_20d')) else None,
            },
            "sentiment": sentiment
        }

        return context

    def save_to_csv(self, filepath: str):
        """Save data to CSV file."""
        if self.data is None:
            raise ValueError("No data to save")
        self.data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {filepath}")
        return self.data


if __name__ == "__main__":
    # Example usage
    fetcher = DataFetcher("BTC-USD")

    # Fetch 1 year of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    df = fetcher.fetch(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    # Add indicators
    df = fetcher.add_technical_indicators()

    # Show sample context
    context = fetcher.get_market_context(100)
    print("\nSample market context:")
    print(json.dumps(context, indent=2))

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
