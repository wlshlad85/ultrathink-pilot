#!/usr/bin/env python3
"""
Query experiment database - useful SQL queries for analyzing training runs.
"""

import sqlite3
import sys
from pathlib import Path


def run_query(query: str, db_path: str = "experiments.db"):
    """Run a SQL query and display results."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            print("No results found.")
            return

        # Get column names
        columns = [description[0] for description in cursor.description]

        # Print header
        print("\n" + "-" * 80)
        print(" | ".join(columns))
        print("-" * 80)

        # Print rows
        for row in results:
            print(" | ".join(str(val) if val is not None else "NULL" for val in row))

        print("-" * 80)
        print(f"{len(results)} rows returned")

    except sqlite3.Error as e:
        print(f"❌ SQL Error: {e}")
    finally:
        conn.close()


def main():
    """Interactive query tool."""

    db_path = "experiments.db"

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Run setup_experiment_db.py first.")
        return

    print("="*80)
    print("EXPERIMENT DATABASE QUERY TOOL")
    print("="*80)
    print()

    if len(sys.argv) > 1:
        # Run custom query from command line
        query = " ".join(sys.argv[1:])
        print(f"Running: {query}\n")
        run_query(query)
        return

    # Predefined useful queries
    print("Useful Queries:")
    print()
    print("1. List all experiments")
    print("2. Show best experiments by Sharpe ratio")
    print("3. Show episode progression for latest experiment")
    print("4. Show regime awareness scores")
    print("5. Compare all experiments")
    print("6. Custom SQL query")
    print("0. Exit")
    print()

    choice = input("Enter choice (0-6): ").strip()

    if choice == "1":
        print("\n📊 ALL EXPERIMENTS:")
        run_query("SELECT * FROM experiment_summary ORDER BY id DESC")

    elif choice == "2":
        print("\n🏆 TOP EXPERIMENTS BY SHARPE RATIO:")
        run_query("""
            SELECT
                id, name, best_test_sharpe, best_test_return,
                final_episode, early_stopped
            FROM experiments
            WHERE best_test_sharpe IS NOT NULL
            ORDER BY best_test_sharpe DESC
            LIMIT 10
        """)

    elif choice == "3":
        exp_id = input("Enter experiment ID (or press Enter for latest): ").strip()
        if not exp_id:
            # Get latest experiment ID
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM experiments")
            exp_id = cursor.fetchone()[0]
            conn.close()

        print(f"\n📈 EPISODE PROGRESSION (Experiment {exp_id}):")
        run_query(f"""
            SELECT
                episode_num, train_return, train_reward,
                test_return, test_sharpe, is_best_model
            FROM episodes
            WHERE experiment_id = {exp_id}
            ORDER BY episode_num
        """)

    elif choice == "4":
        print("\n🎯 REGIME AWARENESS ANALYSIS:")
        run_query("""
            SELECT
                e.id, e.name,
                r.bull_buy_prob, r.bear_buy_prob,
                r.buy_prob_difference,
                r.total_return
            FROM experiments e
            JOIN regime_analysis r ON e.id = r.experiment_id
            ORDER BY r.buy_prob_difference DESC
        """)

    elif choice == "5":
        print("\n📊 EXPERIMENT COMPARISON:")
        run_query("""
            SELECT
                id, name,
                final_episode,
                best_test_sharpe,
                best_test_return,
                early_stopped,
                status
            FROM experiments
            ORDER BY id DESC
        """)

    elif choice == "6":
        print("\nEnter SQL query (end with semicolon):")
        query_lines = []
        while True:
            line = input()
            query_lines.append(line)
            if line.strip().endswith(";"):
                break

        query = "\n".join(query_lines)
        run_query(query)

    elif choice == "0":
        print("Exiting.")
        return

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
