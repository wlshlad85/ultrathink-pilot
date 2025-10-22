"""Utilities for migrating SQLite tracking data into TimescaleDB.

The production environment stores historical experiment metadata in a
TimescaleDB/PostgreSQL instance while local development typically uses the
lightweight SQLite database that ships with :mod:`ml_persistence`.  This module
bridges both worlds by copying compatible columns from the SQLite schema into a
TimescaleDB database without assuming perfectly identical schemas.

The migrator performs the following safeguards:

* Discover shared columns between the source and destination tables.
* Gracefully skip columns that only exist on one side (fixing crashes such as
  ``no such column: hyperparameters``).
* Coerce values into appropriate PostgreSQL types (timestamps, JSON, numerics,
  booleans) before insertion.
* Batch inserts to avoid keeping the entire dataset in memory.

The entrypoint can be executed directly::

    python -m ml_persistence.migrations.sqlite_to_timescale \
        --sqlite-path ml_experiments.db \
        --pg-host localhost --pg-database ultrathink_experiments \
        --pg-user ultrathink --pg-password changeme_in_production

For CLI backwards compatibility both ``--pg-*`` and the older
``--postgres-*`` flags are accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency during import time
    from psycopg import connect, sql
except ImportError:  # pragma: no cover - provide a lazy failure later
    connect = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class TableMapping:
    """Describe how a SQLite table maps onto a TimescaleDB table."""

    source: str
    target: str
    column_mapping: Dict[str, str] | None = None
    conflict_columns: Sequence[str] | None = None
    skip_source_columns: Sequence[str] | None = None

    def normalised_column_mapping(self) -> Dict[str, str]:
        mapping = dict(self.column_mapping or {})
        if self.skip_source_columns:
            for col in self.skip_source_columns:
                mapping.setdefault(col, None)
        return mapping


def compute_column_mapping(
    source_columns: Iterable[str],
    target_columns: Iterable[str],
    explicit_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return a list of usable (source, target) column pairs.

    Parameters
    ----------
    source_columns:
        Column names reported by SQLite.
    target_columns:
        Column names present on the TimescaleDB table.
    explicit_mapping:
        Optional overrides mapping source column names to new names.  A value of
        ``None`` indicates that the column should be skipped.

    Returns
    -------
    tuple
        ``(usable, missing)`` where ``usable`` is a list of (source, target)
        column pairs ready for insertion and ``missing`` enumerates the columns
        that could not be matched.
    """

    explicit_mapping = explicit_mapping or {}
    target_set = set(target_columns)
    usable: List[Tuple[str, str]] = []
    missing: List[Tuple[str, str]] = []

    for source_col in source_columns:
        if source_col in explicit_mapping and explicit_mapping[source_col] is None:
            missing.append((source_col, "<skipped>"))
            continue

        target_col = explicit_mapping.get(source_col, source_col)
        if target_col in target_set:
            usable.append((source_col, target_col))
        else:
            missing.append((source_col, target_col))
    return usable, missing


def _parse_datetime(value: str | int | float) -> Optional[datetime]:
    """Try to convert the SQLite value into a :class:`datetime` instance."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                if fmt is None:
                    return datetime.fromisoformat(candidate)
                return datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def convert_value(value, data_type: str):
    """Coerce ``value`` into a representation suitable for PostgreSQL."""

    if value is None:
        return None

    normalised = (data_type or "").lower()

    if "timestamp" in normalised or normalised in {"date", "time"}:
        parsed = _parse_datetime(value)
        return parsed if parsed is not None else value

    if normalised in {"json", "jsonb"}:
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value

    if normalised in {"integer", "smallint", "bigint"}:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return int(value)

    if normalised in {"real", "double precision", "numeric", "decimal"}:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return float(value)

    if normalised in {"boolean", "bool"}:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "t", "1", "yes", "y"}:
                return True
            if lowered in {"false", "f", "0", "no", "n"}:
                return False
        return bool(value)

    return value


def _ensure_psycopg_available() -> None:
    if connect is None or sql is None:  # pragma: no cover - runtime guard
        raise ImportError(
            "psycopg is required for TimescaleDB migrations."
            " Install with `pip install psycopg[binary]`."
        )


@dataclass
class SQLiteToTimescaleMigrator:
    """Move experiment tracking data from SQLite into TimescaleDB."""

    sqlite_path: Path
    pg_dsn: str
    schema: str = "public"
    table_mappings: Sequence[TableMapping] = field(default_factory=list)
    batch_size: int = 500
    dry_run: bool = False

    def __post_init__(self):
        if not self.table_mappings:
            self.table_mappings = [
                TableMapping("experiments", "experiments", conflict_columns=["id"]),
                TableMapping("models", "model_checkpoints", conflict_columns=["id"]),
                TableMapping("datasets", "dataset_versions", conflict_columns=["id"]),
                TableMapping("metrics", "experiment_metrics", conflict_columns=["id"]),
                TableMapping(
                    "hyperparameters",
                    "experiment_hyperparameters",
                    conflict_columns=["id"],
                ),
                TableMapping("artifacts", "artifacts", conflict_columns=["id"]),
                TableMapping(
                    "experiment_datasets",
                    "experiment_datasets",
                    conflict_columns=["experiment_id", "dataset_id"],
                ),
            ]

    def migrate(self, only_tables: Optional[Sequence[str]] = None) -> Dict[str, int]:
        """Run the migration and return inserted row counts per table."""

        results: Dict[str, int] = {}
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row

        _ensure_psycopg_available()

        with connect(self.pg_dsn) as pg_conn:
            for mapping in self.table_mappings:
                if only_tables and mapping.source not in only_tables:
                    logger.debug("Skipping table %s (not in selection)", mapping.source)
                    continue
                inserted = self._migrate_table(sqlite_conn, pg_conn, mapping)
                results[mapping.source] = inserted
            if self.dry_run:
                pg_conn.rollback()
            else:
                pg_conn.commit()

        sqlite_conn.close()
        return results

    def _fetch_sqlite_columns(self, conn: sqlite3.Connection, table: str) -> List[str]:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        logger.debug("SQLite columns for %s: %s", table, columns)
        return columns

    def _fetch_pg_columns(self, pg_conn, table: str) -> Dict[str, str]:
        query = sql.SQL(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """
        )
        with pg_conn.cursor() as cur:
            cur.execute(query, (self.schema, table))
            columns = {row[0]: row[1] for row in cur.fetchall()}
        if not columns:
            logger.warning("Target table %s.%s not found", self.schema, table)
        else:
            logger.debug("PostgreSQL columns for %s.%s: %s", self.schema, table, columns)
        return columns

    def _migrate_table(self, sqlite_conn, pg_conn, mapping: TableMapping) -> int:
        logger.info(
            "Migrating table %s -> %s.%s", mapping.source, self.schema, mapping.target
        )

        source_columns = self._fetch_sqlite_columns(sqlite_conn, mapping.source)
        target_columns = self._fetch_pg_columns(pg_conn, mapping.target)
        usable, missing = compute_column_mapping(
            source_columns, target_columns.keys(), mapping.normalised_column_mapping()
        )

        if missing:
            logger.debug("Columns skipped for %s: %s", mapping.source, missing)

        if not usable:
            logger.warning(
                "No shared columns for table %s -> %s; skipping", mapping.source, mapping.target
            )
            return 0

        select_cols = ", ".join(f'"{src}"' for src, _ in usable)
        select_sql = f"SELECT {select_cols} FROM {mapping.source}"
        total_rows = self._count_rows(sqlite_conn, mapping.source)

        if total_rows == 0:
            logger.info("No rows to migrate for %s", mapping.source)
            return 0

        with sqlite_conn:  # ensure the cursor observes a consistent snapshot
            sqlite_cursor = sqlite_conn.cursor()
            inserted_total = 0
            for offset in range(0, total_rows, self.batch_size):
                sqlite_cursor.execute(
                    f"{select_sql} LIMIT ? OFFSET ?", (self.batch_size, offset)
                )
                rows = sqlite_cursor.fetchall()
                prepared_rows = [
                    self._prepare_row(row, usable, target_columns) for row in rows
                ]
                inserted = self._insert_rows(pg_conn, mapping, usable, prepared_rows)
                inserted_total += inserted
        logger.info(
            "Migrated %d/%d rows from %s", inserted_total, total_rows, mapping.source
        )
        return inserted_total

    def _prepare_row(
        self,
        row: sqlite3.Row,
        column_pairs: Sequence[Tuple[str, str]],
        target_columns: Dict[str, str],
    ) -> Tuple:
        values: List = []
        for source_col, target_col in column_pairs:
            data_type = target_columns.get(target_col, "")
            values.append(convert_value(row[source_col], data_type))
        return tuple(values)

    def _insert_rows(
        self,
        pg_conn,
        mapping: TableMapping,
        column_pairs: Sequence[Tuple[str, str]],
        prepared_rows: Sequence[Tuple],
    ) -> int:
        if not prepared_rows:
            return 0

        column_idents = sql.SQL(", ").join(
            sql.Identifier(target) for _, target in column_pairs
        )
        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in column_pairs)
        base_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
            sql.Identifier(self.schema),
            sql.Identifier(mapping.target),
            column_idents,
            placeholders,
        )

        conflict_clause = sql.SQL("")
        if mapping.conflict_columns:
            valid_conflicts = [
                col for col in mapping.conflict_columns if col in [t for _, t in column_pairs]
            ]
            if not valid_conflicts:
                # fall back to checking direct existence in target schema
                target_cols = {col for _, col in column_pairs}
                valid_conflicts = [
                    col for col in (mapping.conflict_columns or []) if col in target_cols
                ]
            if valid_conflicts:
                conflict_clause = sql.SQL(" ON CONFLICT ({}) DO NOTHING").format(
                    sql.SQL(", ").join(sql.Identifier(col) for col in valid_conflicts)
                )

        query = base_query + conflict_clause

        if self.dry_run:
            logger.debug("Dry-run enabled; skipping insert into %s", mapping.target)
            return len(prepared_rows)

        with pg_conn.cursor() as cur:
            cur.executemany(query, prepared_rows)
        return len(prepared_rows)

    def _count_rows(self, sqlite_conn, table: str) -> int:
        cursor = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}")
        result = cursor.fetchone()
        return int(result[0]) if result else 0


def _build_default_dsn(args: argparse.Namespace) -> str:
    return (
        f"host={args.pg_host} port={args.pg_port} dbname={args.pg_database} "
        f"user={args.pg_user} password={args.pg_password}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SQLite data to TimescaleDB")
    parser.add_argument("--sqlite-path", default="ml_experiments.db")
    parser.add_argument("--pg-host", "--postgres-host", dest="pg_host", default="localhost")
    parser.add_argument("--pg-port", "--postgres-port", dest="pg_port", default=5432, type=int)
    parser.add_argument(
        "--pg-database", "--postgres-db", dest="pg_database", default="ultrathink_experiments"
    )
    parser.add_argument(
        "--pg-user", "--postgres-user", dest="pg_user", default="ultrathink"
    )
    parser.add_argument(
        "--pg-password",
        "--postgres-password",
        dest="pg_password",
        default="changeme_in_production",
    )
    parser.add_argument("--schema", default="public")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--tables",
        nargs="*",
        help="Optional subset of source tables to migrate",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        logger.error("SQLite database %s does not exist", sqlite_path)
        return 1

    dsn = _build_default_dsn(args)
    migrator = SQLiteToTimescaleMigrator(
        sqlite_path=sqlite_path,
        pg_dsn=dsn,
        schema=args.schema,
        batch_size=args.batch_size,
        dry_run=args.dry_run or args.validate_only,
    )

    if args.validate_only:
        logger.info("Validation-only mode: checking connections and schemas")
        try:
            migrator.migrate(only_tables=args.tables)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.error("Validation failed: %s", exc)
            return 1
        logger.info("Validation successful")
        return 0

    try:
        results = migrator.migrate(only_tables=args.tables)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.error("Migration failed: %s", exc)
        return 1

    for table, count in results.items():
        logger.info("Migrated %d rows for table %s", count, table)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry point
    raise SystemExit(main())
