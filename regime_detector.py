#!/usr/bin/env python3
"""
Multi-Feature Market Regime Detector

Detects market regimes using a combination of:
1. Rule-based classification (quantitative criteria)
2. Technical indicator analysis
3. Regime transition smoothing

Supports both historical labeling and real-time detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass


class RegimePrimary(Enum):
    """Primary market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class RegimeVolatility(Enum):
    """Volatility regime overlay."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class RegimeStrength(Enum):
    """Trend strength regime overlay."""
    STRONG = "STRONG"
    WEAK = "WEAK"
    CHOPPY = "CHOPPY"


@dataclass
class RegimeLabel:
    """Complete regime classification for a given time period."""
    primary: RegimePrimary
    volatility: RegimeVolatility
    strength: RegimeStrength
    confidence: float  # 0-1, how confident we are in this classification
    features: Optional[Dict[str, float]] = None  # Raw feature values


class RegimeDetector:
    """
    Comprehensive market regime detection system.

    Uses technical indicators and quantitative rules to classify
    market regimes for specialized agent selection.
    """

    def __init__(
        self,
        lookback_window: int = 30,
        volatility_window: int = 20,
        trend_threshold: float = 0.05,
        min_regime_duration: int = 5
    ):
        """
        Initialize regime detector.

        Args:
            lookback_window: Days for trend calculations (default: 30)
            volatility_window: Days for volatility calculations (default: 20)
            trend_threshold: Return threshold for bull/bear (+/- 5%)
            min_regime_duration: Minimum days before regime change (smoothing)
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.min_regime_duration = min_regime_duration

    def calculate_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all regime detection features.

        Args:
            market_data: DataFrame with OHLCV columns

        Returns:
            DataFrame with additional feature columns
        """
        df = market_data.copy()

        # === PRICE-BASED FEATURES ===

        # Returns
        df['returns_1d'] = df['close'].pct_change()
        df['returns_7d'] = df['close'].pct_change(7)
        df[f'returns_{self.lookback_window}d'] = df['close'].pct_change(self.lookback_window)

        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # SMA slopes (rate of change)
        df['sma_30_slope'] = df['close'].rolling(self.lookback_window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == self.lookback_window else np.nan
        )

        # Price position relative to MAs
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']

        # Moving average alignment (trend confirmation)
        df['ma_alignment'] = (
            (df['sma_10'] > df['sma_20']).astype(int) +
            (df['sma_20'] > df['sma_50']).astype(int) +
            (df['sma_50'] > df['sma_200']).astype(int)
        )  # 0-3 score, 3 = perfect bull alignment

        # === VOLATILITY FEATURES ===

        # Historical volatility (annualized)
        df['volatility'] = df['returns_1d'].rolling(self.volatility_window).std() * np.sqrt(365)

        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']  # ATR as % of price

        # Bollinger Band width
        bb_std = df['close'].rolling(20).std()
        bb_middle = df['close'].rolling(20).mean()
        df['bb_width'] = (4 * bb_std) / bb_middle  # Width as % of middle band

        # === MOMENTUM FEATURES ===

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Rate of Change
        df['roc'] = df['close'].pct_change(10) * 100

        # === TREND STRENGTH FEATURES ===

        # ADX (Average Directional Index)
        # Simplified calculation
        high_diff = df['high'].diff()
        low_diff = df['low'].diff().abs()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr_14 = true_range.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # === VOLUME FEATURES ===

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Volume-weighted return
        df['volume_weighted_return'] = df['returns_1d'] * df['volume_ratio']

        # === DERIVED FEATURES ===

        # Positive/negative days ratio
        df['positive_days_ratio'] = (
            df['returns_1d'].rolling(self.lookback_window).apply(
                lambda x: (x > 0).sum() / len(x)
            )
        )

        # Volatility rank (current vol vs historical)
        vol_percentile = df['volatility'].rolling(252).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else np.nan
        )
        df['volatility_rank'] = vol_percentile

        return df

    def classify_primary_regime(self, features: pd.Series) -> Tuple[RegimePrimary, float]:
        """
        Classify primary market regime (Bull/Bear/Sideways).

        Args:
            features: Series with calculated features for a single time point

        Returns:
            (regime, confidence): Primary regime and confidence score
        """
        # Extract key features
        return_30d = features.get(f'returns_{self.lookback_window}d', 0)
        sma_slope = features.get('sma_30_slope', 0)
        price_vs_sma_50 = features.get('price_vs_sma_50', 0)
        positive_days = features.get('positive_days_ratio', 0.5)
        ma_alignment = features.get('ma_alignment', 1)

        # Count signals for each regime
        bull_signals = 0
        bear_signals = 0
        sideways_signals = 0

        # Signal 1: 30-day return
        if return_30d > self.trend_threshold:
            bull_signals += 2  # Strong weight
        elif return_30d < -self.trend_threshold:
            bear_signals += 2
        else:
            sideways_signals += 2

        # Signal 2: SMA slope
        if not np.isnan(sma_slope):
            if sma_slope > 0:
                bull_signals += 1
            elif sma_slope < 0:
                bear_signals += 1

        # Signal 3: Price vs SMA
        if not np.isnan(price_vs_sma_50):
            if price_vs_sma_50 > 0.05:  # 5% above SMA
                bull_signals += 1
            elif price_vs_sma_50 < -0.05:  # 5% below SMA
                bear_signals += 1
            else:
                sideways_signals += 1

        # Signal 4: Positive days ratio
        if not np.isnan(positive_days):
            if positive_days > 0.6:
                bull_signals += 1
            elif positive_days < 0.4:
                bear_signals += 1
            else:
                sideways_signals += 1

        # Signal 5: MA alignment
        if ma_alignment >= 3:  # All MAs aligned bullish
            bull_signals += 1
        elif ma_alignment == 0:  # All MAs aligned bearish
            bear_signals += 1

        # Determine regime based on signals
        total_signals = bull_signals + bear_signals + sideways_signals
        max_signals = max(bull_signals, bear_signals, sideways_signals)

        if total_signals == 0:
            return RegimePrimary.UNKNOWN, 0.0

        confidence = max_signals / total_signals

        if bull_signals == max_signals:
            regime = RegimePrimary.BULL
        elif bear_signals == max_signals:
            regime = RegimePrimary.BEAR
        elif sideways_signals == max_signals:
            regime = RegimePrimary.SIDEWAYS
        else:
            regime = RegimePrimary.UNKNOWN

        return regime, confidence

    def classify_volatility_regime(self, features: pd.Series) -> Tuple[RegimeVolatility, float]:
        """
        Classify volatility regime.

        Args:
            features: Series with calculated features

        Returns:
            (regime, confidence): Volatility regime and confidence
        """
        volatility = features.get('volatility', 0.5)  # Annualized
        atr_pct = features.get('atr_pct', 0.03)
        bb_width = features.get('bb_width', 0.1)
        return_7d = features.get('returns_7d', 0)

        # Check for crash conditions first (highest priority)
        if abs(return_7d) > 0.30:  # 30% move in 7 days
            return RegimeVolatility.EXTREME, 1.0

        # Normal volatility classification
        if volatility < 0.30:  # <30% annualized
            regime = RegimeVolatility.LOW
            confidence = 1.0 - (volatility / 0.30)  # Higher confidence when very low
        elif volatility < 0.80:  # 30-80% annualized
            regime = RegimeVolatility.MEDIUM
            confidence = 0.7  # Medium confidence in medium vol
        else:  # >80% annualized
            regime = RegimeVolatility.HIGH
            confidence = min(1.0, volatility / 1.5)  # Higher confidence when very high

        # Adjust confidence based on ATR agreement
        expected_atr = {
            RegimeVolatility.LOW: 0.03,
            RegimeVolatility.MEDIUM: 0.06,
            RegimeVolatility.HIGH: 0.10
        }

        if not np.isnan(atr_pct):
            atr_agreement = 1.0 - min(1.0, abs(atr_pct - expected_atr[regime]) / 0.05)
            confidence = (confidence + atr_agreement) / 2

        return regime, confidence

    def classify_strength_regime(self, features: pd.Series) -> Tuple[RegimeStrength, float]:
        """
        Classify trend strength regime.

        Args:
            features: Series with calculated features

        Returns:
            (regime, confidence): Strength regime and confidence
        """
        adx = features.get('adx', 20)
        ma_alignment = features.get('ma_alignment', 1)
        roc = features.get('roc', 0)

        # ADX is primary indicator
        if adx > 25:
            regime = RegimeStrength.STRONG
            confidence = min(1.0, adx / 40)
        elif adx < 20:
            regime = RegimeStrength.CHOPPY
            confidence = 1.0 - (adx / 20)
        else:
            regime = RegimeStrength.WEAK
            confidence = 0.6  # Medium confidence

        # Adjust based on MA alignment
        if ma_alignment in [0, 3]:  # Perfect alignment
            if regime == RegimeStrength.STRONG:
                confidence = min(1.0, confidence + 0.2)
        elif ma_alignment == 1 or ma_alignment == 2:  # Mixed
            if regime == RegimeStrength.CHOPPY:
                confidence = min(1.0, confidence + 0.2)

        return regime, confidence

    def detect_regime(self, market_data: pd.DataFrame, index: int) -> RegimeLabel:
        """
        Detect regime at a specific time point.

        Args:
            market_data: DataFrame with features
            index: Index of the row to classify

        Returns:
            RegimeLabel with complete classification
        """
        if index < 0 or index >= len(market_data):
            return RegimeLabel(
                primary=RegimePrimary.UNKNOWN,
                volatility=RegimeVolatility.MEDIUM,
                strength=RegimeStrength.WEAK,
                confidence=0.0
            )

        features = market_data.iloc[index]

        # Classify all regime dimensions
        primary, primary_conf = self.classify_primary_regime(features)
        volatility, vol_conf = self.classify_volatility_regime(features)
        strength, strength_conf = self.classify_strength_regime(features)

        # Overall confidence is average of individual confidences
        overall_confidence = (primary_conf + vol_conf + strength_conf) / 3

        # Store raw features for analysis
        feature_dict = {
            'return_30d': features.get(f'returns_{self.lookback_window}d'),
            'volatility': features.get('volatility'),
            'adx': features.get('adx'),
            'rsi': features.get('rsi'),
            'price_vs_sma_50': features.get('price_vs_sma_50')
        }

        return RegimeLabel(
            primary=primary,
            volatility=volatility,
            strength=strength,
            confidence=overall_confidence,
            features=feature_dict
        )

    def label_historical_data(
        self,
        market_data: pd.DataFrame,
        smooth_transitions: bool = True
    ) -> pd.DataFrame:
        """
        Label entire historical dataset with regime classifications.

        Args:
            market_data: DataFrame with OHLCV data
            smooth_transitions: Apply minimum regime duration smoothing

        Returns:
            DataFrame with added regime columns
        """
        print("Calculating regime detection features...")
        df = self.calculate_features(market_data)

        print(f"Classifying regimes for {len(df)} data points...")
        regimes = []
        for i in range(len(df)):
            regime = self.detect_regime(df, i)
            regimes.append(regime)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(df)} data points...")

        # Add regime labels to dataframe
        df['regime_primary'] = [r.primary.value for r in regimes]
        df['regime_volatility'] = [r.volatility.value for r in regimes]
        df['regime_strength'] = [r.strength.value for r in regimes]
        df['regime_confidence'] = [r.confidence for r in regimes]

        # Optionally smooth transitions
        if smooth_transitions:
            print("Smoothing regime transitions...")
            df = self._smooth_regime_transitions(df)

        return df

    def _smooth_regime_transitions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply minimum regime duration to avoid excessive switching.

        Args:
            df: DataFrame with regime labels

        Returns:
            DataFrame with smoothed regime labels
        """
        smoothed = df.copy()

        # Smooth primary regime
        smoothed['regime_primary'] = self._smooth_column(
            df['regime_primary'],
            self.min_regime_duration
        )

        return smoothed

    def _smooth_column(self, series: pd.Series, min_duration: int) -> pd.Series:
        """
        Smooth a categorical series to have minimum run length.

        Args:
            series: Series to smooth
            min_duration: Minimum consecutive occurrences

        Returns:
            Smoothed series
        """
        smoothed = series.copy()
        current_regime = None
        regime_start = 0

        for i in range(len(series)):
            if series.iloc[i] != current_regime:
                # Regime changed
                regime_duration = i - regime_start

                # If previous regime was too short, extend it
                if regime_duration < min_duration and regime_start > 0:
                    smoothed.iloc[regime_start:i] = smoothed.iloc[regime_start - 1]

                current_regime = series.iloc[i]
                regime_start = i

        return smoothed


if __name__ == "__main__":
    """Test regime detector on sample data."""
    print("Regime Detector Test")
    print("=" * 60)

    # This is a placeholder - in practice, load real market data
    print("\nNote: This is a library module.")
    print("Use label_historical_data.py to label full datasets.")
    print("=" * 60)
