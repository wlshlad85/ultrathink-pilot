#!/usr/bin/env python3
"""
Setup experiment tracking database for RL trading agent experiments.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

def create_experiment_database():
    """Create SQLite database schema for tracking experiments."""

    db_path = Path("experiments.db")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            status TEXT DEFAULT 'running',

            -- Model configuration
            state_dim INTEGER,
            action_dim INTEGER,
            learning_rate REAL,
            gamma REAL,

            -- Data split info
            train_start_date TEXT,
            train_end_date TEXT,
            test_start_date TEXT,
            test_end_date TEXT,

            -- Final results
            best_test_sharpe REAL,
            best_test_return REAL,
            final_episode INTEGER,
            early_stopped BOOLEAN,

            -- Notes
            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create episodes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            episode_num INTEGER NOT NULL,

            -- Training metrics
            train_return REAL,
            train_reward REAL,
            episode_length INTEGER,

            -- Test metrics (if evaluated this episode)
            test_return REAL,
            test_sharpe REAL,
            is_best_model BOOLEAN DEFAULT 0,

            -- Policy stats
            avg_policy_loss REAL,
            avg_value_loss REAL,

            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            UNIQUE(experiment_id, episode_num)
        )
    """)

    # Create regime analysis table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regime_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Overall performance
            final_value REAL,
            total_return REAL,
            total_steps INTEGER,

            -- Regime distribution
            bull_steps INTEGER,
            bear_steps INTEGER,
            neutral_steps INTEGER,

            -- BUY probabilities by regime
            bull_buy_prob REAL,
            bear_buy_prob REAL,
            neutral_buy_prob REAL,

            -- Regime awareness score
            buy_prob_difference REAL,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)

    # Create hyperparameters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hyperparameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            param_value TEXT NOT NULL,

            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)

    # Create useful views
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS experiment_summary AS
        SELECT
            e.id,
            e.name,
            e.status,
            e.best_test_sharpe,
            e.best_test_return,
            e.final_episode,
            e.early_stopped,
            COUNT(ep.id) as num_episodes,
            e.start_time,
            e.end_time
        FROM experiments e
        LEFT JOIN episodes ep ON e.id = ep.experiment_id
        GROUP BY e.id
    """)

    conn.commit()
    conn.close()

    print(f"✅ Database created: {db_path.absolute()}")
    print()
    print("Tables created:")
    print("  - experiments: Track training runs")
    print("  - episodes: Store per-episode metrics")
    print("  - regime_analysis: Store regime behavior analysis")
    print("  - hyperparameters: Track model configuration")
    print()
    print("Views created:")
    print("  - experiment_summary: Quick overview of all experiments")
    print()
    print("Example queries:")
    print("  SELECT * FROM experiment_summary;")
    print("  SELECT * FROM episodes WHERE experiment_id = 1 ORDER BY episode_num;")
    print("  SELECT name, best_test_sharpe FROM experiments ORDER BY best_test_sharpe DESC LIMIT 5;")


if __name__ == "__main__":
    create_experiment_database()
