#!/usr/bin/env python3
"""Quick database inspection tool."""

import sqlite3
import time
import sys

def inspect_database(db_path='experiments.db', retry=3):
    """Inspect the experiments database with retry logic."""

    for attempt in range(retry):
        try:
            # Try read-only access
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=5.0)
            cursor = conn.cursor()

            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f'📊 Database has {table_count} tables')

            # Get experiment count
            cursor.execute('SELECT COUNT(*) FROM experiments')
            exp_count = cursor.fetchone()[0]
            print(f'🔬 Total experiments tracked: {exp_count}')

            if exp_count > 0:
                # Show top experiments by Sharpe ratio
                cursor.execute('''
                    SELECT id, name, status,
                           COALESCE(ROUND(best_test_sharpe, 3), 0) as sharpe,
                           COALESCE(ROUND(best_test_return, 2), 0) as return_pct,
                           COALESCE(final_episode, 0) as episodes
                    FROM experiments
                    ORDER BY best_test_sharpe DESC NULLS LAST
                    LIMIT 10
                ''')

                print('\n🏆 Top Experiments by Sharpe Ratio:')
                print('=' * 85)
                print(f"{'ID':>3} | {'Name':<36} | {'Status':<8} | {'Sharpe':>6} | {'Return%':>7} | {'Eps':>4}")
                print('-' * 85)
                for row in cursor.fetchall():
                    print(f"{row[0]:3d} | {row[1]:<36} | {row[2]:<8} | {row[3]:6.3f} | {row[4]:7.2f} | {row[5]:4d}")

                # Get episode count
                cursor.execute('SELECT COUNT(*) FROM episodes')
                ep_count = cursor.fetchone()[0]
                print(f'\n📈 Total episodes recorded: {ep_count}')

                # Get regime analysis count
                cursor.execute('SELECT COUNT(*) FROM regime_analysis')
                regime_count = cursor.fetchone()[0]
                print(f'🎯 Regime analyses: {regime_count}')

            else:
                print('\n⚠️ No experiments found in database yet!')

            conn.close()
            return True

        except sqlite3.OperationalError as e:
            if 'locked' in str(e) and attempt < retry - 1:
                print(f'Database locked, retrying in 1 second... (attempt {attempt + 1}/{retry})')
                time.sleep(1)
            else:
                print(f'❌ Error: {e}')
                return False
        except Exception as e:
            print(f'❌ Unexpected error: {e}')
            return False

    return False

if __name__ == '__main__':
    success = inspect_database()
    sys.exit(0 if success else 1)
