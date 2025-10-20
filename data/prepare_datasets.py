#!/usr/bin/env python3
"""
Dataset preparation for training and evaluation.
Pre-processes and caches datasets for faster experiments.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for a dataset."""

    def __init__(
        self,
        name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        description: str = ""
    ):
        self.name = name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.description = description

    def to_dict(self):
        return {
            "name": self.name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
            "description": self.description
        }


# Pre-defined dataset configurations
DATASETS = [
    DatasetConfig(
        name="btc_2023_train",
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        description="Bitcoin 2023 - Training data"
    ),
    DatasetConfig(
        name="btc_2024_test",
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-06-30",
        description="Bitcoin 2024 H1 - Test data"
    ),
    DatasetConfig(
        name="btc_2022_2023",
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2023-12-31",
        description="Bitcoin 2022-2023 - Extended training"
    ),
    DatasetConfig(
        name="eth_2023_train",
        symbol="ETH-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        description="Ethereum 2023 - Training data"
    ),
    DatasetConfig(
        name="eth_2024_test",
        symbol="ETH-USD",
        start_date="2024-01-01",
        end_date="2024-06-30",
        description="Ethereum 2024 H1 - Test data"
    ),
]


def prepare_dataset(config: DatasetConfig, output_dir: Path) -> dict:
    """
    Prepare a single dataset.

    Args:
        config: Dataset configuration
        output_dir: Output directory

    Returns:
        Metadata dictionary
    """
    logger.info(f"Preparing dataset: {config.name}")

    # Fetch data
    fetcher = DataFetcher(config.symbol)
    df = fetcher.fetch(config.start_date, config.end_date, config.interval)
    fetcher.add_technical_indicators()
    data = fetcher.data

    # Save to parquet (efficient storage)
    output_path = output_dir / f"{config.name}.parquet"
    data.to_parquet(output_path)

    # Generate metadata
    metadata = {
        **config.to_dict(),
        "rows": len(data),
        "columns": list(data.columns),
        "date_range": {
            "start": str(data.index[0].date()),
            "end": str(data.index[-1].date())
        },
        "price_stats": {
            "min": float(data['close'].min()),
            "max": float(data['close'].max()),
            "mean": float(data['close'].mean()),
            "std": float(data['close'].std())
        },
        "file_path": str(output_path.relative_to(output_dir.parent)),
        "file_size_mb": output_path.stat().st_size / (1024 * 1024),
        "created_at": datetime.now().isoformat()
    }

    logger.info(f"  Rows: {metadata['rows']}, Size: {metadata['file_size_mb']:.2f} MB")

    return metadata


def prepare_all_datasets(output_dir: str = "data/datasets"):
    """Prepare all pre-defined datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing {len(DATASETS)} datasets...")
    logger.info(f"Output directory: {output_path.absolute()}")

    all_metadata = {}

    for config in DATASETS:
        try:
            metadata = prepare_dataset(config, output_path)
            all_metadata[config.name] = metadata
        except Exception as e:
            logger.error(f"Failed to prepare {config.name}: {e}")
            continue

    # Save master metadata file
    metadata_path = output_path / "datasets_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"\nDataset preparation complete!")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"\nAvailable datasets:")
    for name, meta in all_metadata.items():
        print(f"  - {name}: {meta['rows']} rows, {meta['file_size_mb']:.2f} MB")


def load_dataset(name: str, datasets_dir: str = "data/datasets") -> pd.DataFrame:
    """
    Load a prepared dataset.

    Args:
        name: Dataset name
        datasets_dir: Directory containing datasets

    Returns:
        DataFrame with market data and indicators
    """
    datasets_path = Path(datasets_dir)

    # Load metadata
    metadata_path = datasets_path / "datasets_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Datasets not found. Run prepare_all_datasets() first.")

    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)

    if name not in all_metadata:
        available = ", ".join(all_metadata.keys())
        raise ValueError(f"Dataset '{name}' not found. Available: {available}")

    # Load data
    metadata = all_metadata[name]
    file_path = Path(metadata['file_path'])

    logger.info(f"Loading dataset: {name}")
    logger.info(f"  Period: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
    logger.info(f"  Rows: {metadata['rows']}")

    df = pd.read_parquet(file_path)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--prepare", action="store_true", help="Prepare all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--load", type=str, help="Load and display a dataset")

    args = parser.parse_args()

    if args.prepare:
        prepare_all_datasets()

    elif args.list:
        metadata_path = Path("data/datasets/datasets_metadata.json")
        if not metadata_path.exists():
            print("No datasets found. Run with --prepare first.")
        else:
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)

            print("\nAvailable Datasets:")
            print("="*60)
            for name, meta in all_metadata.items():
                print(f"\n{name}")
                print(f"  Symbol: {meta['symbol']}")
                print(f"  Period: {meta['date_range']['start']} to {meta['date_range']['end']}")
                print(f"  Rows: {meta['rows']}")
                print(f"  Size: {meta['file_size_mb']:.2f} MB")
                print(f"  Description: {meta['description']}")

    elif args.load:
        df = load_dataset(args.load)
        print(f"\nDataset: {args.load}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
        print(f"\nColumns:")
        print(df.columns.tolist())

    else:
        parser.print_help()
