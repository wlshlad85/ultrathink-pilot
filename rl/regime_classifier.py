#!/usr/bin/env python3
"""
Market regime classifier for regime-specific agent training.

Classifies each data point into one of three regimes:
- BULL: Strong uptrend (price > SMA50 > SMA200)
- BEAR: Strong downtrend (price < SMA50 < SMA200)
- SIDEWAYS: Consolidation/choppy (neither bull nor bear)
"""

import pandas as pd
import numpy as np
from typing import Literal

RegimeType = Literal["BULL", "BEAR", "SIDEWAYS"]


class RegimeClassifier:
    """Classify market data into bull/bear/sideways regimes."""

    def __init__(self, sma_short: int = 50, sma_long: int = 200):
        """
        Initialize regime classifier.

        Args:
            sma_short: Short-term moving average period (default: 50)
            sma_long: Long-term moving average period (default: 200)
        """
        self.sma_short = sma_short
        self.sma_long = sma_long

    def classify(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each row in the dataframe into a regime.

        Regime Rules:
        - BULL: price > SMA50 AND SMA50 > SMA200
        - BEAR: price < SMA50 AND SMA50 < SMA200
        - SIDEWAYS: Everything else (choppy/consolidating)

        Args:
            data: DataFrame with 'close' and moving averages

        Returns:
            DataFrame with added 'regime' column
        """
        df = data.copy()

        # Ensure we have the required SMAs
        if f'sma_{self.sma_short}' not in df.columns:
            df[f'sma_{self.sma_short}'] = df['close'].rolling(window=self.sma_short).mean()
        if f'sma_{self.sma_long}' not in df.columns:
            df[f'sma_{self.sma_long}'] = df['close'].rolling(window=self.sma_long).mean()

        price = df['close']
        sma_short = df[f'sma_{self.sma_short}']
        sma_long = df[f'sma_{self.sma_long}']

        # Classify regimes
        bull_condition = (price > sma_short) & (sma_short > sma_long)
        bear_condition = (price < sma_short) & (sma_short < sma_long)

        df['regime'] = 'SIDEWAYS'  # default
        df.loc[bull_condition, 'regime'] = 'BULL'
        df.loc[bear_condition, 'regime'] = 'BEAR'

        return df

    def get_regime_segments(self, data: pd.DataFrame, regime: RegimeType) -> pd.DataFrame:
        """
        Filter data to only include specified regime.

        Args:
            data: DataFrame with 'regime' column
            regime: 'BULL', 'BEAR', or 'SIDEWAYS'

        Returns:
            Filtered DataFrame containing only the specified regime
        """
        if 'regime' not in data.columns:
            data = self.classify(data)

        return data[data['regime'] == regime].copy()

    def get_regime_stats(self, data: pd.DataFrame) -> dict:
        """
        Get statistics about regime distribution in the data.

        Args:
            data: DataFrame with 'regime' column

        Returns:
            Dictionary with regime counts and percentages
        """
        if 'regime' not in data.columns:
            data = self.classify(data)

        total_rows = len(data)
        regime_counts = data['regime'].value_counts()

        stats = {
            'total_rows': total_rows,
            'bull_count': int(regime_counts.get('BULL', 0)),
            'bear_count': int(regime_counts.get('BEAR', 0)),
            'sideways_count': int(regime_counts.get('SIDEWAYS', 0)),
            'bull_pct': float(regime_counts.get('BULL', 0) / total_rows * 100),
            'bear_pct': float(regime_counts.get('BEAR', 0) / total_rows * 100),
            'sideways_pct': float(regime_counts.get('SIDEWAYS', 0) / total_rows * 100),
        }

        return stats
