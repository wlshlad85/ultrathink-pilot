#!/usr/bin/env python3
"""
Experiment logging utilities for RL trading agent.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ExperimentLogger:
    """Logger for tracking RL experiments in SQLite database."""

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self.experiment_id: Optional[int] = None

    def start_experiment(
        self,
        name: str,
        description: str = "",
        state_dim: int = None,
        action_dim: int = None,
        learning_rate: float = None,
        gamma: float = None,
        train_start_date: str = None,
        train_end_date: str = None,
        test_start_date: str = None,
        test_end_date: str = None,
        **hyperparams
    ) -> int:
        """
        Start a new experiment and return its ID.

        Args:
            name: Experiment name
            description: Optional description
            state_dim: State space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            gamma: Discount factor
            train_start_date: Training data start date
            train_end_date: Training data end date
            test_start_date: Test data start date
            test_end_date: Test data end date
            **hyperparams: Additional hyperparameters to store

        Returns:
            Experiment ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiments (
                name, description, start_time, status,
                state_dim, action_dim, learning_rate, gamma,
                train_start_date, train_end_date,
                test_start_date, test_end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, description, datetime.now(), 'running',
            state_dim, action_dim, learning_rate, gamma,
            train_start_date, train_end_date,
            test_start_date, test_end_date
        ))

        self.experiment_id = cursor.lastrowid

        # Store additional hyperparameters
        for param_name, param_value in hyperparams.items():
            cursor.execute("""
                INSERT INTO hyperparameters (experiment_id, param_name, param_value)
                VALUES (?, ?, ?)
            """, (self.experiment_id, param_name, str(param_value)))

        conn.commit()
        conn.close()

        print(f"✅ Started experiment '{name}' (ID: {self.experiment_id})")
        return self.experiment_id

    def log_episode(
        self,
        episode_num: int,
        train_return: float,
        train_reward: float,
        episode_length: int,
        test_return: Optional[float] = None,
        test_sharpe: Optional[float] = None,
        is_best_model: bool = False,
        avg_policy_loss: Optional[float] = None,
        avg_value_loss: Optional[float] = None
    ):
        """Log metrics for a single episode."""
        if self.experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO episodes (
                experiment_id, episode_num,
                train_return, train_reward, episode_length,
                test_return, test_sharpe, is_best_model,
                avg_policy_loss, avg_value_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment_id, episode_num,
            train_return, train_reward, episode_length,
            test_return, test_sharpe, is_best_model,
            avg_policy_loss, avg_value_loss
        ))

        conn.commit()
        conn.close()

    def end_experiment(
        self,
        final_episode: int,
        best_test_sharpe: float,
        best_test_return: float,
        early_stopped: bool = False,
        notes: str = ""
    ):
        """Mark experiment as complete."""
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE experiments SET
                end_time = ?,
                status = 'completed',
                final_episode = ?,
                best_test_sharpe = ?,
                best_test_return = ?,
                early_stopped = ?,
                notes = ?
            WHERE id = ?
        """, (
            datetime.now(), final_episode,
            best_test_sharpe, best_test_return,
            early_stopped, notes,
            self.experiment_id
        ))

        conn.commit()
        conn.close()

        print(f"✅ Experiment {self.experiment_id} completed")

    def log_regime_analysis(
        self,
        final_value: float,
        total_return: float,
        total_steps: int,
        bull_steps: int,
        bear_steps: int,
        neutral_steps: int,
        bull_buy_prob: float,
        bear_buy_prob: float,
        neutral_buy_prob: float
    ):
        """Log regime-conditional behavior analysis."""
        if self.experiment_id is None:
            raise ValueError("No active experiment.")

        buy_prob_difference = bull_buy_prob - bear_buy_prob

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regime_analysis (
                experiment_id, final_value, total_return, total_steps,
                bull_steps, bear_steps, neutral_steps,
                bull_buy_prob, bear_buy_prob, neutral_buy_prob,
                buy_prob_difference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment_id, final_value, total_return, total_steps,
            bull_steps, bear_steps, neutral_steps,
            bull_buy_prob, bear_buy_prob, neutral_buy_prob,
            buy_prob_difference
        ))

        conn.commit()
        conn.close()

        print(f"✅ Logged regime analysis for experiment {self.experiment_id}")


def query_experiments(db_path: str = "experiments.db"):
    """Display all experiments."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM experiment_summary ORDER BY id DESC")
    results = cursor.fetchall()

    if not results:
        print("No experiments found.")
        conn.close()
        return

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for row in results:
        exp_id, name, status, sharpe, ret, final_ep, early_stop, num_eps, start, end = row
        print(f"\n#{exp_id}: {name}")
        print(f"  Status: {status}")
        print(f"  Episodes: {num_eps}/{final_ep if final_ep else '?'}")
        if sharpe:
            print(f"  Best Sharpe: {sharpe:+.3f}")
        if ret:
            print(f"  Best Return: {ret:+.2f}%")
        if early_stop:
            print(f"  Early stopped: Yes")
        print(f"  Started: {start}")
        if end:
            print(f"  Ended: {end}")

    conn.close()


if __name__ == "__main__":
    # Demo usage
    print("Experiment Logger Utility")
    print("="*80)
    print()
    print("Usage in training script:")
    print()
    print("  from experiment_logger import ExperimentLogger")
    print()
    print("  logger = ExperimentLogger()")
    print("  exp_id = logger.start_experiment(")
    print("      name='Professional 300-episode run',")
    print("      state_dim=53,")
    print("      action_dim=3,")
    print("      learning_rate=3e-4,")
    print("      gamma=0.99")
    print("  )")
    print()
    print("  # In training loop:")
    print("  logger.log_episode(")
    print("      episode_num=ep,")
    print("      train_return=ep_return,")
    print("      train_reward=ep_reward,")
    print("      episode_length=steps")
    print("  )")
    print()
    print("  # After training:")
    print("  logger.end_experiment(")
    print("      final_episode=300,")
    print("      best_test_sharpe=0.5,")
    print("      best_test_return=5.03")
    print("  )")
    print()
    print("="*80)
    print()

    # Show current experiments
    query_experiments()
