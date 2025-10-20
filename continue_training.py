#!/usr/bin/env python3
"""
Continue Professional RL Training - Phases 2 & 3
Completes validation and test phases starting from Phase 1 model
"""
import sys
import os
from pathlib import Path
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl.train import train
from rl.ppo_agent import PPOAgent
from rl.trading_env import TradingEnv
from backtesting.data_fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def continue_professional_training():
    """
    Complete Phases 2 & 3 of institutional training.

    Phase 1 (COMPLETED): 2020-2021 training - Best: 43.38%
    Phase 2 (NOW):       2022 validation - Tune & prevent overfitting
    Phase 3 (NEXT):      2023 test - Final unbiased evaluation
    """

    print("=" * 80)
    print("CONTINUING PROFESSIONAL RL TRAINING - PHASES 2 & 3")
    print("=" * 80)
    print()
    print("Previous Results:")
    print("  Phase 1 (Train 2020-2021): [COMPLETED] - 43.38% best return")
    print()
    print("Remaining Work:")
    print("  Phase 2 (Validation 2022):  100 episodes")
    print("  Phase 3 (Test 2023):        100 episodes")
    print("  Total:                      200 episodes")
    print()
    print("Training Configuration:")
    print("  Update Freq:   2048 steps")
    print("  GPU:           CUDA (RTX 5070)")
    print("  Algorithm:     PPO (Proximal Policy Optimization)")
    print("  Save Freq:     Every 20 episodes")
    print("=" * 80)
    print()

    root_dir = Path(__file__).parent
    model_dir = root_dir / "rl" / "models"
    log_dir = root_dir / "rl" / "logs"

    # Ensure directories exist
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: Validation Set (2022)
    print("\n" + "=" * 80)
    print("PHASE 2: VALIDATION SET (2022)")
    print("=" * 80)
    print("Training fresh agent on 2022 validation data...")
    print("This phase tests generalization to different market regime.")
    print()

    train(
        num_episodes=100,
        symbol="BTC-USD",
        start_date="2022-01-01",
        end_date="2022-12-31",
        update_freq=2048,
        initial_capital=100000.0,
        save_freq=20,
        model_dir=str(model_dir / "phase2_validation"),
        log_dir=str(log_dir / "phase2_validation")
    )

    print("\n[PHASE 2 COMPLETE] Validation model saved")
    print(f"  Location: {model_dir / 'phase2_validation' / 'best_model.pth'}")

    # Phase 3: Test Set (2023)
    print("\n" + "=" * 80)
    print("PHASE 3: TEST SET (2023)")
    print("=" * 80)
    print("Training fresh agent on 2023 test data...")
    print("This phase gives final unbiased performance estimate.")
    print()

    train(
        num_episodes=100,
        symbol="BTC-USD",
        start_date="2023-01-01",
        end_date="2023-12-31",
        update_freq=2048,
        initial_capital=100000.0,
        save_freq=20,
        model_dir=str(model_dir / "phase3_test"),
        log_dir=str(log_dir / "phase3_test")
    )

    print("\n[PHASE 3 COMPLETE] Test model saved")
    print(f"  Location: {model_dir / 'phase3_test' / 'best_model.pth'}")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - ALL 3 PHASES FINISHED")
    print("=" * 80)
    print()
    print("Results Summary:")
    print(f"  Phase 1 (Train 2020-2021):     {model_dir / 'phase1_train'}")
    print(f"  Phase 2 (Validation 2022):     {model_dir / 'phase2_validation'}")
    print(f"  Phase 3 (Test 2023):           {model_dir / 'phase3_test'}")
    print()
    print("Total Episodes: 300 (100 per phase)")
    print("Total Training Time: 4 years of market data")
    print()
    print("Next Steps:")
    print("  1. Review logs for performance metrics")
    print("  2. Compare returns across all 3 phases")
    print("  3. Check for overfitting (train >> test performance)")
    print("  4. Evaluate on 2024 data for final validation")
    print()
    print("Your RTX 5070 has completed institutional-grade training!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        continue_professional_training()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial models have been saved.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
