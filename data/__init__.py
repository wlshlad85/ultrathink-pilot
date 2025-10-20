"""
UltraThink Data Management Module

Multi-source data fetching, caching, and dataset preparation.
"""

from .data_providers import (
    DataProvider,
    YFinanceProvider,
    BinanceProvider,
    CryptoCompareProvider,
    MultiSourceDataFetcher
)

from .prepare_datasets import (
    DatasetConfig,
    prepare_dataset,
    prepare_all_datasets,
    load_dataset,
    DATASETS
)

__all__ = [
    'DataProvider',
    'YFinanceProvider',
    'BinanceProvider',
    'CryptoCompareProvider',
    'MultiSourceDataFetcher',
    'DatasetConfig',
    'prepare_dataset',
    'prepare_all_datasets',
    'load_dataset',
    'DATASETS',
]
