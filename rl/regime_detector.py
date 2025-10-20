#!/usr/bin/env python3
"""
Market Regime Detection for model specialization analysis.
Classifies market conditions as Bull, Bear, or Neutral.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Literal, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.data_fetcher import DataFetcher


RegimeType = Literal["bull", "bear", "neutral"]


@dataclass
class MarketRegime:
    """Represents a classified market period."""
    regime_type: RegimeType
    start_date: str
    end_date: str
    price_change_pct: float
    avg_volatility: float
    confidence: float  # 0-1 score


class RegimeDetector:
    """
    Detects market regimes using technical indicators and price action.

    Classification Logic:
    - Bull: Strong uptrend with positive momentum
    - Bear: Strong downtrend with negative momentum
    - Neutral: Sideways/choppy market with no clear trend
    """

    def __init__(
        self,
        bull_threshold: float = 0.10,  # 10% gain over lookback
        bear_threshold: float = -0.10,  # 10% loss over lookback
        volatility_threshold: float = 0.03,  # 3% daily volatility
        lookback_days: int = 60
    ):
        """
        Initialize regime detector.

        Args:
            bull_threshold: Minimum return % to classify as bull
            bear_threshold: Maximum return % to classify as bear (negative)
            volatility_threshold: Volatility threshold for confidence scoring
            lookback_days: Window for regime detection
        """
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.volatility_threshold = volatility_threshold
        self.lookback_days = lookback_days

    def detect_regime(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> RegimeType:
        """
        Detect market regime at a specific point in time.

        Args:
            df: DataFrame with price and indicator data
            current_idx: Index position to evaluate

        Returns:
            Regime classification (bull/bear/neutral)
        """
        # Calculate lookback window
        start_idx = max(0, current_idx - self.lookback_days)
        window_data = df.iloc[start_idx:current_idx + 1]

        if len(window_data) < 5:  # Need minimum data
            return "neutral"

        # Price momentum
        start_price = window_data['close'].iloc[0]
        end_price = window_data['close'].iloc[-1]
        price_change = (end_price / start_price) - 1

        # Trend strength (SMA relationships)
        current_row = df.iloc[current_idx]
        sma_20 = current_row.get('sma_20', end_price)
        sma_50 = current_row.get('sma_50', end_price)

        trend_strength = 0
        if pd.notna(sma_20) and pd.notna(sma_50):
            if sma_20 > sma_50:
                trend_strength += 1  # Bullish alignment
            else:
                trend_strength -= 1  # Bearish alignment

        # RSI momentum
        rsi = current_row.get('rsi_14', 50)
        if pd.notna(rsi):
            if rsi > 60:
                trend_strength += 1
            elif rsi < 40:
                trend_strength -= 1

        # Classify regime
        if price_change >= self.bull_threshold and trend_strength > 0:
            return "bull"
        elif price_change <= self.bear_threshold or trend_strength < 0:
            return "bear"
        else:
            return "neutral"

    def classify_period(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> MarketRegime:
        """
        Classify an entire time period as a single regime.

        Args:
            symbol: Trading symbol
            start_date: Period start
            end_date: Period end

        Returns:
            MarketRegime object with classification
        """
        # Fetch data
        fetcher = DataFetcher(symbol)
        fetcher.fetch(start_date, end_date)
        fetcher.add_technical_indicators()
        df = fetcher.data

        if df.empty:
            raise ValueError(f"No data available for {symbol} {start_date} to {end_date}")

        # Calculate overall metrics
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        price_change_pct = ((end_price / start_price) - 1) * 100

        # Average volatility
        if 'returns_1d' in df.columns:
            avg_volatility = df['returns_1d'].std()
        else:
            returns = df['close'].pct_change()
            avg_volatility = returns.std()

        # Detect regime at multiple points
        sample_indices = np.linspace(
            self.lookback_days,
            len(df) - 1,
            min(10, len(df) - self.lookback_days),
            dtype=int
        )

        regime_votes = {"bull": 0, "bear": 0, "neutral": 0}
        for idx in sample_indices:
            regime = self.detect_regime(df, idx)
            regime_votes[regime] += 1

        # Determine overall regime
        regime_type = max(regime_votes, key=regime_votes.get)
        total_votes = sum(regime_votes.values())
        confidence = regime_votes[regime_type] / total_votes if total_votes > 0 else 0

        return MarketRegime(
            regime_type=regime_type,
            start_date=start_date,
            end_date=end_date,
            price_change_pct=price_change_pct,
            avg_volatility=avg_volatility,
            confidence=confidence
        )

    def get_historical_regimes(
        self,
        symbol: str = "BTC-USD"
    ) -> List[MarketRegime]:
        """
        Get predefined regime classifications for known historical periods.

        Returns:
            List of MarketRegime objects for testing
        """
        # Define test periods based on BTC history
        test_periods = [
            ("2022-01-01", "2022-12-31"),  # Bear market (BTC: 47k → 16k)
            ("2023-01-01", "2023-06-30"),  # Bull recovery (BTC: 16k → 30k)
            ("2024-01-01", "2024-06-30"),  # Bull continuation (BTC: 44k → 60k+)
        ]

        regimes = []
        for start_date, end_date in test_periods:
            regime = self.classify_period(symbol, start_date, end_date)
            regimes.append(regime)

        return regimes


def print_regime_analysis(regimes: List[MarketRegime]):
    """Print formatted regime classification results."""
    print("\n" + "="*70)
    print("MARKET REGIME ANALYSIS")
    print("="*70)

    for regime in regimes:
        print(f"\n{regime.start_date} to {regime.end_date}")
        print(f"  Regime:        {regime.regime_type.upper()}")
        print(f"  Price Change:  {regime.price_change_pct:+.2f}%")
        print(f"  Volatility:    {regime.avg_volatility*100:.2f}%")
        print(f"  Confidence:    {regime.confidence*100:.1f}%")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Test regime detection on known periods
    print("Testing Regime Detector...")

    detector = RegimeDetector()
    regimes = detector.get_historical_regimes(symbol="BTC-USD")

    print_regime_analysis(regimes)

    # Verify expected classifications
    expected = ["bear", "bull", "bull"]
    actual = [r.regime_type for r in regimes]

    print("\nVerification:")
    for i, (exp, act) in enumerate(zip(expected, actual)):
        status = "✓" if exp == act else "✗"
        print(f"  Period {i+1}: Expected {exp.upper()}, Got {act.upper()} {status}")
