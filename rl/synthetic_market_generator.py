"""
Sophisticated synthetic market data generator for validating regime-aware trading agents.

This generator creates realistic cryptocurrency price data with:
- Controlled regime sequences (bull/bear/sideways)
- Realistic statistical properties (fat tails, volatility clustering)
- Economic realism (realistic drawdowns, rallies, volatility)
- Technical indicator emergence (SMAs, RSI behave naturally)

Philosophy:
- Calibrate parameters from real historical data (2017-2021)
- Generate new sequences with those parameters
- Test if agent learned transferable patterns vs memorized sequences
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class RegimeParams:
    """Parameters for price generation in a specific market regime."""
    name: str
    drift: float  # Daily expected return (μ)
    base_volatility: float  # Base daily volatility (σ)
    vol_persistence: float  # GARCH persistence (how long volatility clusters)
    jump_probability: float  # Probability of jump/crash events
    jump_size_mean: float  # Mean size of jumps
    jump_size_std: float  # Std of jump sizes


class SyntheticMarketGenerator:
    """
    Generate realistic synthetic crypto market data with controlled regimes.

    Uses a regime-switching stochastic volatility model:
    - Returns follow Student's t-distribution (fat tails)
    - Volatility follows GARCH(1,1) process (clustering)
    - Occasional jumps/crashes (realistic tail events)
    - Regime-dependent parameters calibrated from real data
    """

    def __init__(self, seed: int = 42):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

        # Default regime parameters (calibrated from BTC 2017-2021)
        self.regime_params = {
            'bull': RegimeParams(
                name='bull',
                drift=0.0025,  # ~150% annualized
                base_volatility=0.035,  # 3.5% daily
                vol_persistence=0.85,
                jump_probability=0.02,
                jump_size_mean=0.05,  # Positive jumps
                jump_size_std=0.03
            ),
            'bear': RegimeParams(
                name='bear',
                drift=-0.0015,  # ~-40% annualized
                base_volatility=0.055,  # 5.5% daily (higher vol)
                vol_persistence=0.90,  # Volatility persists longer
                jump_probability=0.05,  # More crashes
                jump_size_mean=-0.08,  # Negative jumps
                jump_size_std=0.04
            ),
            'sideways': RegimeParams(
                name='sideways',
                drift=0.0,  # No trend
                base_volatility=0.025,  # 2.5% daily (lower vol)
                vol_persistence=0.75,
                jump_probability=0.01,
                jump_size_mean=0.0,  # Symmetric jumps
                jump_size_std=0.03
            )
        }

        # GARCH parameters
        self.garch_omega = 0.00001  # Base variance
        self.garch_alpha = 0.10  # Weight on last shock
        # garch_beta is regime.vol_persistence

        # Student's t degrees of freedom (controls fat tails)
        self.t_df = 5  # df=5 gives realistic kurtosis ~7

    def calibrate_from_real_data(self, df: pd.DataFrame, regime_col: str = 'regime'):
        """
        Calibrate regime parameters from real historical data.

        Args:
            df: DataFrame with columns ['close', 'regime']
            regime_col: Name of regime column
        """
        df = df.copy()
        df['returns'] = df['close'].pct_change()

        for regime_name in ['bull', 'bear', 'sideways']:
            regime_data = df[df[regime_col] == regime_name.upper()]

            if len(regime_data) < 30:
                print(f"Warning: Only {len(regime_data)} samples for {regime_name}, using defaults")
                continue

            returns = regime_data['returns'].dropna()

            # Update parameters based on empirical data
            self.regime_params[regime_name].drift = returns.mean()
            self.regime_params[regime_name].base_volatility = returns.std()

            # Estimate jump probability (returns > 3 std)
            threshold = 3 * returns.std()
            jumps = returns[np.abs(returns) > threshold]
            self.regime_params[regime_name].jump_probability = len(jumps) / len(returns)

            if len(jumps) > 0:
                self.regime_params[regime_name].jump_size_mean = jumps.mean()
                self.regime_params[regime_name].jump_size_std = jumps.std()

            print(f"\n{regime_name.upper()} regime calibrated:")
            print(f"  Drift: {self.regime_params[regime_name].drift:.4f}")
            print(f"  Volatility: {self.regime_params[regime_name].base_volatility:.4f}")
            print(f"  Jump prob: {self.regime_params[regime_name].jump_probability:.4f}")

    def generate_scenario(
        self,
        scenario_name: str,
        total_days: int,
        initial_price: float = 10000.0
    ) -> pd.DataFrame:
        """
        Generate a specific test scenario.

        Args:
            scenario_name: Name of predefined scenario
            total_days: Total number of days to generate
            initial_price: Starting price

        Returns:
            DataFrame with columns ['date', 'close', 'regime', ...]
        """
        if scenario_name == 'extended_regimes':
            # Test: Long pure regime periods (never seen in training)
            regime_sequence = (
                [('bull', 365)] +  # 1 year pure bull
                [('sideways', 180)] +  # 6 months sideways
                [('bear', 365)]  # 1 year pure bear
            )

        elif scenario_name == 'rapid_switching':
            # Test: Frequent regime changes (adaptation speed)
            regime_sequence = []
            for _ in range(total_days // 60):
                regime_sequence.extend([
                    ('bull', 20),
                    ('bear', 20),
                    ('sideways', 20)
                ])

        elif scenario_name == 'volatility_stress':
            # Test: Extreme volatility levels
            regime_sequence = [
                ('bull', 180),  # Will use 2x volatility
                ('bear', 180),  # Will use 0.5x volatility
            ]
            # Modify params temporarily
            original_bull_vol = self.regime_params['bull'].base_volatility
            original_bear_vol = self.regime_params['bear'].base_volatility
            self.regime_params['bull'].base_volatility *= 2.0
            self.regime_params['bear'].base_volatility *= 0.5

        elif scenario_name == 'adversarial_bull_trap':
            # Test: Bull market that crashes just before typical eval window
            regime_sequence = [
                ('bull', 85),  # Strong bull
                ('bear', 30),  # Sudden crash
                ('sideways', 90)  # Recovery
            ]

        elif scenario_name == 'black_swan':
            # Test: Extreme events beyond training data
            regime_sequence = [
                ('bull', 120),
                ('bear', 7),  # -70% crash in 1 week
                ('bull', 60),  # V-shaped recovery
                ('bear', 180)
            ]
            # Amplify bear crash
            original_bear_jump = self.regime_params['bear'].jump_size_mean
            self.regime_params['bear'].jump_size_mean = -0.15  # Severe crashes

        elif scenario_name == 'mixed_realistic':
            # Test: Realistic mixed regime sequence
            regime_sequence = [
                ('bull', 120),
                ('sideways', 45),
                ('bull', 90),
                ('bear', 180),
                ('sideways', 60),
                ('bull', 200),
                ('bear', 90),
                ('sideways', 30)
            ]

        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Generate prices
        df = self._generate_prices_from_sequence(
            regime_sequence=regime_sequence,
            initial_price=initial_price
        )

        # Restore original parameters if modified
        if scenario_name == 'volatility_stress':
            self.regime_params['bull'].base_volatility = original_bull_vol
            self.regime_params['bear'].base_volatility = original_bear_vol
        elif scenario_name == 'black_swan':
            self.regime_params['bear'].jump_size_mean = original_bear_jump

        return df

    def _generate_prices_from_sequence(
        self,
        regime_sequence: List[Tuple[str, int]],
        initial_price: float
    ) -> pd.DataFrame:
        """
        Generate prices following a regime sequence.

        Args:
            regime_sequence: List of (regime_name, num_days) tuples
            initial_price: Starting price

        Returns:
            DataFrame with synthetic market data
        """
        all_prices = []
        all_regimes = []
        all_returns = []

        current_price = initial_price
        current_volatility = 0.03  # Initialize GARCH volatility

        for regime_name, num_days in regime_sequence:
            params = self.regime_params[regime_name]

            for _ in range(num_days):
                # GARCH(1,1) volatility update
                # σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
                prev_vol_sq = current_volatility ** 2
                shock = (all_returns[-1] if all_returns else 0.0) ** 2
                new_vol_sq = (
                    self.garch_omega +
                    self.garch_alpha * shock +
                    params.vol_persistence * prev_vol_sq
                )
                current_volatility = np.sqrt(max(new_vol_sq, 0.001))  # Floor at 0.1%

                # Generate return from Student's t-distribution (fat tails)
                # Scale by current volatility and regime base volatility
                t_random = stats.t.rvs(
                    df=self.t_df,
                    scale=params.base_volatility * current_volatility,
                    random_state=self.rng
                )

                daily_return = params.drift + t_random

                # Add jumps/crashes with probability
                if self.rng.random() < params.jump_probability:
                    jump = self.rng.normal(
                        params.jump_size_mean,
                        params.jump_size_std
                    )
                    daily_return += jump

                # Update price
                current_price = current_price * (1 + daily_return)

                all_prices.append(current_price)
                all_regimes.append(regime_name.upper())
                all_returns.append(daily_return)

        # Create DataFrame
        df = pd.DataFrame({
            'close': all_prices,
            'regime': all_regimes,
            'returns': all_returns
        })

        # Add date index
        df['date'] = pd.date_range(
            start='2025-01-01',
            periods=len(df),
            freq='D'
        )

        # Add technical indicators to match real data format
        df = self._add_technical_indicators(df)

        # Add volume (correlated with volatility and price changes)
        df['volume'] = self._generate_volume(df)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators that match real data format."""
        df = df.copy()

        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        df['sma_200'] = df['close'].rolling(window=200, min_periods=1).mean()

        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Add OHLC (simplified: use close for all)
        df['open'] = df['close'] * (1 + self.rng.normal(0, 0.005, len(df)))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(self.rng.normal(0, 0.01, len(df))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(self.rng.normal(0, 0.01, len(df))))

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _generate_volume(self, df: pd.DataFrame) -> np.ndarray:
        """Generate realistic volume correlated with price volatility."""
        base_volume = 1e9  # Base daily volume

        # Volume increases with:
        # 1. Price volatility (higher vol = higher volume)
        # 2. Absolute price changes
        # 3. Regime (bull markets have higher volume)

        vol_factor = np.abs(df['returns']) * 50
        regime_factor = df['regime'].map({
            'BULL': 1.3,
            'BEAR': 1.1,
            'SIDEWAYS': 0.9
        })

        volume = base_volume * (1 + vol_factor) * regime_factor

        # Add noise
        volume = volume * (1 + self.rng.normal(0, 0.2, len(df)))

        return volume.values

    def validate_realism(
        self,
        synthetic_df: pd.DataFrame,
        real_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate that synthetic data has realistic statistical properties.

        Compares:
        - Return distribution (KS test)
        - Kurtosis (fat tails)
        - Volatility clustering (autocorrelation in squared returns)
        - Regime distribution

        Args:
            synthetic_df: Generated synthetic data
            real_df: Real historical data for comparison

        Returns:
            Dictionary of validation metrics
        """
        metrics = {}

        # Compare return distributions
        synth_returns = synthetic_df['returns'].dropna()
        real_returns = real_df['close'].pct_change().dropna()

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(synth_returns, real_returns)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pvalue

        # Kurtosis (should be > 3 for fat tails)
        metrics['synthetic_kurtosis'] = stats.kurtosis(synth_returns)
        metrics['real_kurtosis'] = stats.kurtosis(real_returns)

        # Skewness
        metrics['synthetic_skewness'] = stats.skew(synth_returns)
        metrics['real_skewness'] = stats.skew(real_returns)

        # Volatility clustering (autocorr in squared returns)
        synth_sq_autocorr = pd.Series(synth_returns**2).autocorr(lag=1)
        real_sq_autocorr = pd.Series(real_returns**2).autocorr(lag=1)
        metrics['synthetic_vol_clustering'] = synth_sq_autocorr
        metrics['real_vol_clustering'] = real_sq_autocorr

        # Mean and std
        metrics['synthetic_mean_return'] = synth_returns.mean()
        metrics['real_mean_return'] = real_returns.mean()
        metrics['synthetic_volatility'] = synth_returns.std()
        metrics['real_volatility'] = real_returns.std()

        # Regime distribution (if available)
        if 'regime' in real_df.columns:
            synth_regimes = synthetic_df['regime'].value_counts(normalize=True)
            real_regimes = real_df['regime'].value_counts(normalize=True)
            metrics['regime_distribution_diff'] = np.abs(
                synth_regimes - real_regimes
            ).sum()

        return metrics


if __name__ == "__main__":
    # Example usage
    generator = SyntheticMarketGenerator(seed=42)

    # Generate a test scenario
    print("Generating 'mixed_realistic' scenario...")
    df = generator.generate_scenario(
        scenario_name='mixed_realistic',
        total_days=800,
        initial_price=10000.0
    )

    print(f"\nGenerated {len(df)} days of synthetic data")
    print(f"\nPrice range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")

    print("\nRegime distribution:")
    print(df['regime'].value_counts())

    print("\nReturns statistics:")
    returns = df['returns'].dropna()
    print(f"  Mean: {returns.mean():.6f}")
    print(f"  Std: {returns.std():.6f}")
    print(f"  Kurtosis: {stats.kurtosis(returns):.2f} (>3 indicates fat tails)")
    print(f"  Skewness: {stats.skew(returns):.2f}")

    # Save for inspection
    output_path = 'synthetic_data_sample.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved sample to {output_path}")
