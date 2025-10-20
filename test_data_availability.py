#!/usr/bin/env python3
"""
Test Data Availability for Professional Training
Verifies we can fetch 2017-2024 Bitcoin data before running 1,000-episode training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtesting.data_fetcher import DataFetcher

def test_data_availability():
    """Test if we can fetch all required historical data."""

    print("=" * 80)
    print("TESTING DATA AVAILABILITY FOR PROFESSIONAL TRAINING")
    print("=" * 80)
    print()

    fetcher = DataFetcher("BTC-USD")

    # Test 1: Training data (2017-2021)
    print("Test 1: Fetching training data (2017-2021)...")
    try:
        train_data = fetcher.fetch(
            start_date="2017-01-01",
            end_date="2021-12-31"
        )
        print(f"  [PASS] Fetched {len(train_data)} days")
        print(f"         Date range: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"         Price range: ${train_data['close'].min():.2f} - ${train_data['close'].max():.2f}")
        print()
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

    # Test 2: Validation data (2022)
    print("Test 2: Fetching validation data (2022)...")
    try:
        val_data = fetcher.fetch(
            start_date="2022-01-01",
            end_date="2022-12-31"
        )
        print(f"  [PASS] Fetched {len(val_data)} days")
        print(f"         Date range: {val_data.index[0]} to {val_data.index[-1]}")
        print(f"         Price range: ${val_data['close'].min():.2f} - ${val_data['close'].max():.2f}")
        print()
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

    # Test 3: Test data (2023-2024)
    print("Test 3: Fetching test data (2023-2024)...")
    try:
        test_data = fetcher.fetch(
            start_date="2023-01-01",
            end_date="2024-12-31"
        )
        print(f"  [PASS] Fetched {len(test_data)} days")
        print(f"         Date range: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"         Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
        print()
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

    # Summary
    total_days = len(train_data) + len(val_data) + len(test_data)

    print("=" * 80)
    print("DATA AVAILABILITY TEST RESULTS")
    print("=" * 80)
    print()
    print("[SUCCESS] All data successfully fetched!")
    print()
    print(f"Total Dataset: {total_days} days across 7-8 years")
    print(f"  Training (2017-2021):   {len(train_data)} days ({len(train_data)/total_days*100:.1f}%)")
    print(f"  Validation (2022):      {len(val_data)} days ({len(val_data)/total_days*100:.1f}%)")
    print(f"  Test (2023-2024):       {len(test_data)} days ({len(test_data)/total_days*100:.1f}%)")
    print()
    print("Ready to begin professional 1,000-episode training!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_data_availability()
    sys.exit(0 if success else 1)
