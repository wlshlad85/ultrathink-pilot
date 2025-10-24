#!/usr/bin/env python3
"""
SQLite to TimescaleDB Migration Script
Migrates experiment tracking data from ml_experiments.db (SQLite) to TimescaleDB
"""

import sqlite3
import psycopg2
import psycopg2.extras
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLiteToTimescaleMigrator:
    def __init__(self, sqlite_path, postgres_config):
        """
        Initialize migrator

        Args:
            sqlite_path: Path to SQLite database
            postgres_config: Dict with host, port, database, user, password
        """
        self.sqlite_path = sqlite_path
        self.postgres_config = postgres_config
        self.sqlite_conn = None
        self.pg_conn = None

    def connect(self):
        """Establish database connections"""
        try:
            # Connect to SQLite
            logger.info(f"Connecting to SQLite: {self.sqlite_path}")
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row

            # Connect to PostgreSQL/TimescaleDB
            logger.info(f"Connecting to TimescaleDB: {self.postgres_config['host']}")
            self.pg_conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            self.pg_conn.autocommit = False

            logger.info("Database connections established")

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def close(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.pg_conn:
            self.pg_conn.close()
        logger.info("Database connections closed")

    def verify_schemas(self):
        """Verify SQLite and TimescaleDB schemas exist"""
        try:
            # Check SQLite tables
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            sqlite_tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"SQLite tables found: {sqlite_tables}")

            # Check TimescaleDB tables
            cursor = self.pg_conn.cursor()
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            pg_tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"TimescaleDB tables found: {pg_tables}")

            return True

        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return False

    def migrate_experiments(self):
        """Migrate experiments table"""
        logger.info("Migrating experiments table...")

        try:
            # Get SQLite data
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT id, name, description, experiment_type, status,
                       git_commit, git_branch, python_version, random_seed,
                       start_time, end_time, duration_seconds, tags,
                       notes, metadata, created_at, updated_at
                FROM experiments
            """)

            # Get all hyperparameters and organize by experiment_id
            hyperparam_cursor = self.sqlite_conn.cursor()
            hyperparam_cursor.execute("""
                SELECT experiment_id, param_name, param_value, param_type
                FROM hyperparameters
                ORDER BY experiment_id
            """)

            hyperparams_by_exp = {}
            for hp_row in hyperparam_cursor:
                exp_id = hp_row['experiment_id']
                if exp_id not in hyperparams_by_exp:
                    hyperparams_by_exp[exp_id] = {}

                # Convert param_value to appropriate type
                param_value = hp_row['param_value']
                param_type = hp_row['param_type']

                if param_type == 'float':
                    try:
                        param_value = float(param_value)
                    except:
                        pass
                elif param_type == 'int':
                    try:
                        param_value = int(param_value)
                    except:
                        pass
                elif param_type == 'bool':
                    param_value = param_value.lower() in ('true', '1', 'yes')

                hyperparams_by_exp[exp_id][hp_row['param_name']] = param_value

            # Insert into TimescaleDB
            pg_cursor = self.pg_conn.cursor()
            count = 0

            for row in sqlite_cursor:
                # Get hyperparameters for this experiment
                config = hyperparams_by_exp.get(row['id'], {})

                # Parse tags JSON if present
                try:
                    tags_list = json.loads(row['tags']) if row['tags'] else []
                except:
                    tags_list = []

                # Determine model type from hyperparameters or name
                model_type = config.get('model_type', 'ppo_agent')
                if 'bull' in row['name'].lower():
                    model_type = 'bull_specialist'
                elif 'bear' in row['name'].lower():
                    model_type = 'bear_specialist'
                elif 'sideways' in row['name'].lower():
                    model_type = 'sideways_specialist'

                # Add additional tags from name
                if 'experiment' in row['name'].lower() and 'experiment' not in tags_list:
                    tags_list.append('experiment')
                if 'professional' in row['name'].lower() and 'professional' not in tags_list:
                    tags_list.append('professional')

                # Add description and notes to config if present
                if row['description']:
                    config['description'] = row['description']
                if row['notes']:
                    config['notes'] = row['notes']
                if row['git_commit']:
                    config['git_commit'] = row['git_commit']
                if row['git_branch']:
                    config['git_branch'] = row['git_branch']
                if row['python_version']:
                    config['python_version'] = row['python_version']

                pg_cursor.execute("""
                    INSERT INTO experiments
                    (id, experiment_name, model_type, config, status, random_seed,
                     created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        experiment_name = EXCLUDED.experiment_name,
                        status = EXCLUDED.status
                """, (
                    row['id'],
                    row['name'],
                    model_type,
                    json.dumps(config),
                    row['status'] or 'completed',
                    row['random_seed'],
                    row['created_at'] or datetime.now(),
                    row['updated_at'] or row['end_time'] or datetime.now()
                ))

                count += 1

            self.pg_conn.commit()
            logger.info(f"Migrated {count} experiments")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Experiment migration failed: {e}")
            raise

    def migrate_metrics(self):
        """Migrate metrics table to experiment_metrics hypertable"""
        logger.info("Migrating metrics to experiment_metrics...")

        try:
            # Get SQLite data
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT experiment_id, metric_name, value,
                       step, episode, timestamp, metric_type,
                       epoch, value_json, split, phase
                FROM metrics
                ORDER BY experiment_id, timestamp
            """)

            # Batch insert into TimescaleDB for performance
            pg_cursor = self.pg_conn.cursor()
            batch_size = 1000
            batch = []
            count = 0

            for row in sqlite_cursor:
                # Convert timestamp
                try:
                    timestamp = datetime.fromisoformat(row['timestamp'])
                except:
                    timestamp = datetime.now()

                # Create metadata from additional fields
                metadata = {}
                if row['metric_type']:
                    metadata['metric_type'] = row['metric_type']
                if row['epoch']:
                    metadata['epoch'] = row['epoch']
                if row['value_json']:
                    try:
                        metadata['value_json'] = json.loads(row['value_json'])
                    except:
                        metadata['value_json'] = row['value_json']
                if row['split']:
                    metadata['split'] = row['split']
                if row['phase']:
                    metadata['phase'] = row['phase']

                batch.append((
                    timestamp,
                    row['experiment_id'],
                    row['metric_name'],
                    row['value'],  # Changed from metric_value to value
                    row['step'],
                    row['episode'],
                    json.dumps(metadata) if metadata else None
                ))

                if len(batch) >= batch_size:
                    psycopg2.extras.execute_batch(pg_cursor, """
                        INSERT INTO experiment_metrics
                        (time, experiment_id, metric_name, metric_value, step, episode, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, batch)
                    count += len(batch)
                    batch = []
                    self.pg_conn.commit()
                    logger.info(f"Migrated {count} metrics...")

            # Insert remaining batch
            if batch:
                psycopg2.extras.execute_batch(pg_cursor, """
                    INSERT INTO experiment_metrics
                    (time, experiment_id, metric_name, metric_value, step, episode, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, batch)
                count += len(batch)
                self.pg_conn.commit()

            logger.info(f"Migrated {count} total metrics")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Metrics migration failed: {e}")
            raise

    def migrate_models(self):
        """Migrate models table to model_checkpoints"""
        logger.info("Migrating models to model_checkpoints...")

        try:
            # Get SQLite data
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT experiment_id, checkpoint_path, episode_num,
                       train_metric, val_metric, test_metric,
                       is_best, created_at, checkpoint_size_mb,
                       name, version, architecture_type, state_dim,
                       action_dim, global_step, hyperparameters, metadata
                FROM models
            """)

            # Insert into TimescaleDB
            pg_cursor = self.pg_conn.cursor()
            count = 0

            for row in sqlite_cursor:
                # Use checkpoint_size_mb if available, otherwise calculate
                file_size_mb = row['checkpoint_size_mb']
                if not file_size_mb and row['checkpoint_path'] and os.path.exists(row['checkpoint_path']):
                    file_size_mb = os.path.getsize(row['checkpoint_path']) / (1024 * 1024)

                # Use train_metric as sharpe_ratio fallback
                sharpe_ratio = row['train_metric'] if row['train_metric'] else None

                pg_cursor.execute("""
                    INSERT INTO model_checkpoints
                    (experiment_id, checkpoint_path, episode, sharpe_ratio,
                     is_best, created_at, file_size_mb)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    row['experiment_id'],
                    row['checkpoint_path'],
                    row['episode_num'],
                    sharpe_ratio,
                    row['is_best'] == 1 if row['is_best'] is not None else False,
                    row['created_at'] or datetime.now(),
                    file_size_mb
                ))

                count += 1

            self.pg_conn.commit()
            logger.info(f"Migrated {count} model checkpoints")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Model migration failed: {e}")
            raise

    def migrate_datasets(self):
        """Migrate datasets table to dataset_versions"""
        logger.info("Migrating datasets to dataset_versions...")

        try:
            # Get SQLite data
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT id, name, version, symbol, start_date, end_date,
                       record_count, data_hash, created_at, metadata
                FROM datasets
            """)

            # Insert into TimescaleDB
            pg_cursor = self.pg_conn.cursor()
            count = 0

            for row in sqlite_cursor:
                pg_cursor.execute("""
                    INSERT INTO dataset_versions
                    (id, dataset_name, version, symbol, start_date, end_date,
                     record_count, data_hash, created_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_name, version) DO NOTHING
                """, (
                    row['id'],
                    row['name'],
                    row['version'],
                    row['symbol'],
                    row['start_date'],
                    row['end_date'],
                    row['record_count'],
                    row['data_hash'],
                    row['created_at'] or datetime.now(),
                    row['metadata']
                ))

                count += 1

            self.pg_conn.commit()
            logger.info(f"Migrated {count} datasets")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Dataset migration failed: {e}")
            raise

    def validate_migration(self):
        """Validate migration by comparing counts"""
        logger.info("Validating migration...")

        try:
            validations = []

            # Count experiments
            sqlite_cursor = self.sqlite_conn.cursor()
            pg_cursor = self.pg_conn.cursor()

            sqlite_cursor.execute("SELECT COUNT(*) FROM experiments")
            sqlite_exp_count = sqlite_cursor.fetchone()[0]

            pg_cursor.execute("SELECT COUNT(*) FROM experiments")
            pg_exp_count = pg_cursor.fetchone()[0]

            validations.append(("Experiments", sqlite_exp_count, pg_exp_count))

            # Count metrics
            sqlite_cursor.execute("SELECT COUNT(*) FROM metrics")
            sqlite_metrics_count = sqlite_cursor.fetchone()[0]

            pg_cursor.execute("SELECT COUNT(*) FROM experiment_metrics")
            pg_metrics_count = pg_cursor.fetchone()[0]

            validations.append(("Metrics", sqlite_metrics_count, pg_metrics_count))

            # Count models
            sqlite_cursor.execute("SELECT COUNT(*) FROM models")
            sqlite_models_count = sqlite_cursor.fetchone()[0]

            pg_cursor.execute("SELECT COUNT(*) FROM model_checkpoints")
            pg_models_count = pg_cursor.fetchone()[0]

            validations.append(("Models", sqlite_models_count, pg_models_count))

            # Print validation results
            logger.info("\nValidation Results:")
            logger.info("=" * 60)
            all_valid = True
            for table, sqlite_count, pg_count in validations:
                match = "✓" if sqlite_count == pg_count else "✗"
                logger.info(f"{table:20} SQLite: {sqlite_count:6} | TimescaleDB: {pg_count:6} {match}")
                if sqlite_count != pg_count:
                    all_valid = False
            logger.info("=" * 60)

            if all_valid:
                logger.info("✓ Migration validation passed!")
            else:
                logger.warning("✗ Migration validation failed - counts don't match")

            return all_valid

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def run_migration(self, validate_only=False):
        """Run full migration process"""
        try:
            self.connect()

            if not self.verify_schemas():
                logger.error("Schema verification failed - aborting migration")
                return False

            if validate_only:
                return self.validate_migration()

            logger.info("Starting migration...")

            self.migrate_experiments()
            self.migrate_metrics()
            self.migrate_models()

            # Try to migrate datasets if table exists
            try:
                self.migrate_datasets()
            except Exception as e:
                logger.warning(f"Dataset migration skipped: {e}")

            # Validate
            is_valid = self.validate_migration()

            if is_valid:
                logger.info("✓ Migration completed successfully!")
            else:
                logger.warning("⚠ Migration completed with validation warnings")

            return is_valid

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite experiment data to TimescaleDB')
    parser.add_argument('--sqlite-path', type=str,
                       default='ml_experiments.db',
                       help='Path to SQLite database')
    parser.add_argument('--pg-host', type=str,
                       default='localhost',
                       help='PostgreSQL host')
    parser.add_argument('--pg-port', type=int,
                       default=5432,
                       help='PostgreSQL port')
    parser.add_argument('--pg-database', type=str,
                       default='ultrathink_experiments',
                       help='PostgreSQL database name')
    parser.add_argument('--pg-user', type=str,
                       default='ultrathink',
                       help='PostgreSQL user')
    parser.add_argument('--pg-password', type=str,
                       help='PostgreSQL password (or use PG_PASSWORD env var)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing migration, do not migrate')

    args = parser.parse_args()

    # Get password from env if not provided
    pg_password = args.pg_password or os.getenv('PG_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    if not pg_password:
        logger.error("PostgreSQL password required (use --pg-password or PG_PASSWORD env var)")
        sys.exit(1)

    # Build config
    postgres_config = {
        'host': args.pg_host,
        'port': args.pg_port,
        'database': args.pg_database,
        'user': args.pg_user,
        'password': pg_password
    }

    # Check if SQLite database exists
    if not os.path.exists(args.sqlite_path):
        logger.error(f"SQLite database not found: {args.sqlite_path}")
        sys.exit(1)

    # Run migration
    migrator = SQLiteToTimescaleMigrator(args.sqlite_path, postgres_config)
    success = migrator.run_migration(validate_only=args.validate_only)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
