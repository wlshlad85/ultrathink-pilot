# UltraThink Data Management

Enhanced data system with multiple sources, caching, and pre-processed datasets.

## Features

- **Multi-Source Data Fetching** - Yahoo Finance, Binance, CryptoCompare
- **Intelligent Fallback** - Automatically tries alternative sources
- **Smart Caching** - Parquet-based caching for fast loading
- **Pre-processed Datasets** - Ready-to-use training/test splits
- **Dataset Versioning** - Track dataset metadata and provenance

## Quick Start

### 1. Prepare Pre-processed Datasets

```bash
# Prepare all standard datasets (BTC, ETH for 2022-2024)
python3 data/prepare_datasets.py --prepare
```

This creates:
- `btc_2023_train` - Bitcoin 2023 for training
- `btc_2024_test` - Bitcoin 2024 H1 for testing
- `eth_2023_train` - Ethereum 2023 for training
- `eth_2024_test` - Ethereum 2024 H1 for testing
- `btc_2022_2023` - Extended Bitcoin training data

### 2. List Available Datasets

```bash
python3 data/prepare_datasets.py --list
```

### 3. Load a Dataset

```python
from data.prepare_datasets import load_dataset

# Load prepared dataset
df = load_dataset("btc_2023_train")
print(df.head())
```

## Multi-Source Data Fetching

### Basic Usage

```python
from data.data_providers import MultiSourceDataFetcher

# Initialize fetcher
fetcher = MultiSourceDataFetcher(
    preferred_source="yfinance"  # or "binance", "cryptocompare"
)

# Fetch data with automatic fallback
df = fetcher.fetch(
    symbol="BTC-USD",
    start="2023-01-01",
    end="2024-01-01"
)
```

### Available Data Sources

| Source | Symbols | Intervals | Auth Required |
|--------|---------|-----------|---------------|
| **Yahoo Finance** (yfinance) | Stocks, Crypto, Forex | 1m, 5m, 1h, 1d, 1w | No |
| **Binance** (via CCXT) | Crypto pairs | 1m, 5m, 15m, 1h, 1d, 1w | No (for basic) |
| **CryptoCompare** | Crypto | 1m, 1h, 1d | Optional API key |

### Install Additional Sources

```bash
# For Binance and other exchanges
pip install ccxt

# For CryptoCompare
pip install requests

# For enhanced crypto data
pip install python-binance cryptocompare
```

## Pre-processed Datasets

### Standard Datasets

All datasets include:
- OHLCV price data
- 15+ technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.)
- Calculated features (returns, volume ratios, trends)

| Dataset | Symbol | Period | Rows | Purpose |
|---------|--------|--------|------|---------|
| `btc_2023_train` | BTC-USD | 2023 | ~365 | Training |
| `btc_2024_test` | BTC-USD | 2024 H1 | ~180 | Testing |
| `btc_2022_2023` | BTC-USD | 2022-2023 | ~730 | Extended training |
| `eth_2023_train` | ETH-USD | 2023 | ~365 | Training |
| `eth_2024_test` | ETH-USD | 2024 H1 | ~180 | Testing |

### Custom Datasets

Create your own datasets:

```python
from data.prepare_datasets import DatasetConfig, prepare_dataset
from pathlib import Path

config = DatasetConfig(
    name="my_custom_dataset",
    symbol="SPY",  # S&P 500 ETF
    start_date="2020-01-01",
    end_date="2023-12-31",
    interval="1d",
    description="S&P 500 for ML training"
)

metadata = prepare_dataset(config, Path("data/datasets"))
```

## Caching System

Data is automatically cached in `data/cache/` using Parquet format:

```python
# First fetch: downloads from source
df1 = fetcher.fetch("BTC-USD", "2023-01-01", "2024-01-01")

# Second fetch: loads from cache (instant)
df2 = fetcher.fetch("BTC-USD", "2023-01-01", "2024-01-01")
```

Cache key is based on: `provider_symbol_start_end_interval`

### Clear Cache

```bash
rm -rf data/cache/*.parquet
```

## Training with Datasets

### Example: Train on Pre-processed Data

```python
from data.prepare_datasets import load_dataset
from rl.trading_env import TradingEnv
from rl.ppo_agent import PPOAgent

# Load training dataset
train_data = load_dataset("btc_2023_train")

# Create environment (will use cached data)
env = TradingEnv(
    symbol="BTC-USD",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Train agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# ... training loop ...
```

### Example: Train/Test Split

```python
# Train on 2023
python3 rl/train.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --episodes 200

# Test on 2024
python3 rl/evaluate.py \
  --model rl/models/best_model.pth \
  --start 2024-01-01 \
  --end 2024-06-30
```

## Data Quality

### Validation

Datasets are validated for:
- Minimum number of rows (>10)
- No missing critical columns (open, high, low, close, volume)
- Indicators properly calculated
- Date range correctness

### Handling Missing Data

Technical indicators may have NaN values in initial rows (warmup period):

```python
df = load_dataset("btc_2023_train")

# Drop NaN rows (warmup period)
df_clean = df.dropna()

# Or forward-fill indicators
df_filled = df.fillna(method='ffill')
```

## Dataset Metadata

Each dataset has metadata in `data/datasets/datasets_metadata.json`:

```json
{
  "btc_2023_train": {
    "name": "btc_2023_train",
    "symbol": "BTC-USD",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "rows": 365,
    "columns": ["open", "high", "low", "close", "volume", "rsi_14", ...],
    "date_range": {
      "start": "2023-01-01",
      "end": "2023-12-31"
    },
    "price_stats": {
      "min": 16530.25,
      "max": 44700.00,
      "mean": 28450.75,
      "std": 8234.12
    },
    "file_path": "data/datasets/btc_2023_train.parquet",
    "file_size_mb": 0.25,
    "created_at": "2024-10-15T19:30:00"
  }
}
```

## Performance Tips

1. **Use Pre-processed Datasets** - Much faster than fetching on-demand
2. **Enable Caching** - Reuses downloaded data automatically
3. **Parquet Format** - 10x faster than CSV for large datasets
4. **Batch Preparation** - Prepare all datasets once, use many times

## Advanced Usage

### Multiple Symbols

```python
symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

for symbol in symbols:
    df = fetcher.fetch(symbol, "2023-01-01", "2024-01-01")
    # Train agent on each...
```

### Custom Indicators

Add your own indicators to datasets:

```python
df = load_dataset("btc_2023_train")

# Add custom indicator
df['custom_ma'] = df['close'].rolling(window=30).mean()
df['price_to_ma_ratio'] = df['close'] / df['custom_ma']

# Save enhanced dataset
df.to_parquet("data/datasets/btc_2023_enhanced.parquet")
```

### Data Augmentation

For RL training, consider:
- Adding noise to prices
- Time-shifting sequences
- Resampling intervals

## Troubleshooting

### "No data providers available"

Install at least one source:
```bash
pip install yfinance  # Recommended
```

### "All data sources failed"

1. Check internet connection
2. Verify symbol format (e.g., "BTC-USD" not "BTCUSD")
3. Try different source explicitly:
   ```python
   df = fetcher.fetch(..., source="binance")
   ```

### Cache issues

Clear cache and re-fetch:
```bash
rm -rf data/cache/*.parquet
```

## Future Enhancements

- [ ] Alpha Vantage integration
- [ ] On-chain data (Glassnode, etc.)
- [ ] Order book data
- [ ] News sentiment data
- [ ] Economic indicators
- [ ] Dataset versioning with DVC
- [ ] Streaming data support

## File Structure

```
data/
├── README.md                    # This file
├── data_providers.py            # Multi-source fetching
├── prepare_datasets.py          # Dataset preparation
├── cache/                       # Cached raw data
│   └── *.parquet
└── datasets/                    # Pre-processed datasets
    ├── datasets_metadata.json   # Dataset catalog
    ├── btc_2023_train.parquet
    ├── btc_2024_test.parquet
    └── ...
```

## Summary

✅ Multi-source data fetching with fallback
✅ Smart caching for performance
✅ Pre-processed training/test splits
✅ Comprehensive technical indicators
✅ Dataset versioning and metadata
✅ Easy integration with RL training

**Ready to train with high-quality, pre-processed data! 🚀**
