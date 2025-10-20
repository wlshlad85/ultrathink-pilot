#!/usr/bin/env python3
"""
Multi-source data provider system.
Supports multiple data sources with fallback and caching.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging
import json
import hashlib

# Try multiple data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ccxt  # For crypto exchanges
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProvider:
    """Base class for data providers."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, symbol: str, start: str, end: str, interval: str) -> str:
        """Generate cache key for data."""
        key_str = f"{self.__class__.__name__}_{symbol}_{start}_{end}_{interval}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached data."""
        return self.cache_dir / f"{cache_key}.parquet"

    def load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded from cache: {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None

    def save_to_cache(self, df: pd.DataFrame, cache_key: str):
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_key)
            df.to_parquet(cache_path)
            logger.info(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data (to be implemented by subclasses)."""
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider."""

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not available")

        cache_key = self.get_cache_key(symbol, start, end, interval)

        # Try cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        logger.info(f"Fetching {symbol} from Yahoo Finance...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data from Yahoo Finance for {symbol}")

        # Standardize columns
        df.columns = [col.lower() for col in df.columns]

        # Save to cache
        self.save_to_cache(df, cache_key)

        return df


class BinanceProvider(DataProvider):
    """Binance exchange data provider (via CCXT)."""

    def __init__(self, cache_dir: str = "data/cache"):
        super().__init__(cache_dir)
        if CCXT_AVAILABLE:
            self.exchange = ccxt.binance()

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Binance."""
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt not available")

        cache_key = self.get_cache_key(symbol, start, end, interval)

        # Try cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        logger.info(f"Fetching {symbol} from Binance...")

        # Convert interval to Binance format
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
            "1d": "1d", "1w": "1w"
        }
        timeframe = interval_map.get(interval, "1d")

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)

        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=start_ts,
            limit=1000
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Filter by date range
        df = df[(df.index >= start) & (df.index <= end)]

        if df.empty:
            raise ValueError(f"No data from Binance for {symbol}")

        # Save to cache
        self.save_to_cache(df, cache_key)

        return df


class CryptoCompareProvider(DataProvider):
    """CryptoCompare data provider."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        super().__init__(cache_dir)
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2"

    def fetch(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data from CryptoCompare."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not available")

        cache_key = self.get_cache_key(symbol, start, end, interval)

        # Try cache first
        cached_data = self.load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        logger.info(f"Fetching {symbol} from CryptoCompare...")

        # Parse symbol (e.g., "BTC-USD" -> "BTC", "USD")
        if "-" in symbol:
            fsym, tsym = symbol.split("-")
        else:
            fsym, tsym = symbol, "USD"

        # Determine endpoint based on interval
        if interval in ["1m", "5m", "15m", "1h"]:
            endpoint = "histominute" if "m" in interval else "histohour"
            aggregate = int(interval[:-1])
        else:
            endpoint = "histoday"
            aggregate = 1

        # Calculate time range
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        to_ts = int(end_dt.timestamp())

        url = f"{self.base_url}/{endpoint}"
        params = {
            "fsym": fsym,
            "tsym": tsym,
            "toTs": to_ts,
            "limit": 2000,
            "aggregate": aggregate
        }

        if self.api_key:
            params["api_key"] = self.api_key

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("Response") != "Success":
            raise ValueError(f"CryptoCompare API error: {data.get('Message')}")

        # Convert to DataFrame
        prices = data["Data"]["Data"]
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volumefrom']]
        df.rename(columns={'volumefrom': 'volume'}, inplace=True)

        # Filter by date range
        df = df[(df.index >= start) & (df.index <= end)]

        if df.empty:
            raise ValueError(f"No data from CryptoCompare for {symbol}")

        # Save to cache
        self.save_to_cache(df, cache_key)

        return df


class MultiSourceDataFetcher:
    """
    Intelligent data fetcher with multiple sources and fallback.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        cryptocompare_api_key: Optional[str] = None,
        preferred_source: str = "yfinance"
    ):
        """
        Initialize multi-source fetcher.

        Args:
            cache_dir: Directory for caching data
            cryptocompare_api_key: API key for CryptoCompare
            preferred_source: Preferred data source (yfinance, binance, cryptocompare)
        """
        self.cache_dir = Path(cache_dir)
        self.preferred_source = preferred_source

        # Initialize providers
        self.providers: Dict[str, DataProvider] = {}

        if YFINANCE_AVAILABLE:
            self.providers["yfinance"] = YFinanceProvider(cache_dir)
            logger.info("YFinance provider available")

        if CCXT_AVAILABLE:
            self.providers["binance"] = BinanceProvider(cache_dir)
            logger.info("Binance provider available")

        if REQUESTS_AVAILABLE:
            self.providers["cryptocompare"] = CryptoCompareProvider(
                api_key=cryptocompare_api_key,
                cache_dir=cache_dir
            )
            logger.info("CryptoCompare provider available")

        if not self.providers:
            raise RuntimeError("No data providers available. Install yfinance, ccxt, or requests.")

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data with automatic fallback.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "BTC/USDT")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval
            source: Specific source to use (optional)

        Returns:
            DataFrame with OHLCV data
        """
        # Determine source order
        if source:
            source_order = [source]
        else:
            source_order = [self.preferred_source] + [
                s for s in self.providers.keys() if s != self.preferred_source
            ]

        # Try each source
        for src in source_order:
            if src not in self.providers:
                logger.warning(f"Provider {src} not available")
                continue

            try:
                logger.info(f"Trying {src} for {symbol}...")
                df = self.providers[src].fetch(symbol, start, end, interval)

                # Validate data
                if not df.empty and len(df) > 10:
                    logger.info(f"Successfully fetched from {src}: {len(df)} rows")
                    return df
                else:
                    logger.warning(f"{src} returned insufficient data")

            except Exception as e:
                logger.warning(f"{src} failed: {e}")
                continue

        raise ValueError(f"All data sources failed for {symbol}")

    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return list(self.providers.keys())


if __name__ == "__main__":
    # Test multi-source fetcher
    print("Testing MultiSourceDataFetcher...\n")

    fetcher = MultiSourceDataFetcher()
    print(f"Available sources: {fetcher.get_available_sources()}\n")

    # Test fetch
    try:
        df = fetcher.fetch(
            symbol="BTC-USD",
            start="2024-01-01",
            end="2024-06-01"
        )
        print(f"Fetched data: {len(df)} rows")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
        print(f"\nData types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error: {e}")
