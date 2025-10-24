"""
Storage backend for A/B testing results.
Integrates with TimescaleDB for efficient time-series storage and analysis.
"""
import asyncio
import asyncpg
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ABTestStorageBackend:
    """
    TimescaleDB storage backend for A/B test results.
    Handles batch inserts and provides query methods for analysis.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize storage backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        # Use environment variables if not provided
        self.host = host or os.getenv('TIMESCALE_HOST', 'localhost')
        self.port = port or int(os.getenv('TIMESCALE_PORT', '5432'))
        self.database = database or os.getenv('TIMESCALE_DB', 'ultrathink_experiments')
        self.user = user or os.getenv('TIMESCALE_USER', 'ultrathink')
        self.password = password or os.getenv('TIMESCALE_PASSWORD', 'ultrathink_changeme')

        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Establish connection pool to TimescaleDB."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}/{self.database}")

        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from TimescaleDB")

    async def store_ab_results(self, results: List[Dict[str, Any]]):
        """
        Store A/B test results in bulk.

        Args:
            results: List of result dictionaries
        """
        if not results:
            return

        if not self.pool:
            raise RuntimeError("Storage backend not connected. Call connect() first.")

        insert_query = """
            INSERT INTO ab_test_results (
                time, test_id, request_id, assigned_group, symbol,
                control_model, control_action, control_confidence, control_latency_ms,
                treatment_model, treatment_action, treatment_confidence, treatment_latency_ms,
                actions_match, confidence_delta, latency_delta_ms,
                features
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        """

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for result in results:
                    await conn.execute(
                        insert_query,
                        result.get('timestamp'),
                        result.get('test_id'),
                        result.get('request_id'),
                        result.get('assigned_group'),
                        result.get('symbol'),
                        result.get('control_model'),
                        result.get('control_action'),
                        result.get('control_confidence'),
                        result.get('control_latency_ms'),
                        result.get('treatment_model'),
                        result.get('treatment_action'),
                        result.get('treatment_confidence'),
                        result.get('treatment_latency_ms'),
                        result.get('actions_match'),
                        result.get('confidence_delta'),
                        result.get('latency_delta_ms'),
                        result.get('features')  # JSONB
                    )

        logger.info(f"Stored {len(results)} A/B test results")

    async def get_test_stats(
        self,
        test_id: str,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get statistics for an A/B test.

        Args:
            test_id: Test identifier
            hours_back: Number of hours to look back

        Returns:
            Dictionary with test statistics
        """
        if not self.pool:
            raise RuntimeError("Storage backend not connected")

        query = """
            SELECT
                COUNT(*) FILTER (WHERE assigned_group = 'control') AS control_count,
                COUNT(*) FILTER (WHERE assigned_group = 'treatment') AS treatment_count,
                COUNT(*) FILTER (WHERE assigned_group = 'shadow') AS shadow_count,

                AVG(control_confidence) AS avg_control_confidence,
                AVG(treatment_confidence) FILTER (WHERE treatment_confidence IS NOT NULL) AS avg_treatment_confidence,

                AVG(control_latency_ms) AS avg_control_latency,
                AVG(treatment_latency_ms) FILTER (WHERE treatment_latency_ms IS NOT NULL) AS avg_treatment_latency,

                STDDEV(control_latency_ms) AS stddev_control_latency,
                STDDEV(treatment_latency_ms) FILTER (WHERE treatment_latency_ms IS NOT NULL) AS stddev_treatment_latency,

                -- Shadow mode metrics
                SUM(CASE WHEN actions_match = TRUE THEN 1 ELSE 0 END)::FLOAT /
                    NULLIF(SUM(CASE WHEN actions_match IS NOT NULL THEN 1 ELSE 0 END), 0) AS agreement_rate,

                AVG(confidence_delta) FILTER (WHERE confidence_delta IS NOT NULL) AS avg_confidence_delta,
                STDDEV(confidence_delta) FILTER (WHERE confidence_delta IS NOT NULL) AS stddev_confidence_delta,

                AVG(latency_delta_ms) FILTER (WHERE latency_delta_ms IS NOT NULL) AS avg_latency_delta,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_delta_ms) FILTER (WHERE latency_delta_ms IS NOT NULL) AS p95_latency_delta

            FROM ab_test_results
            WHERE test_id = $1
            AND time > NOW() - ($2 || ' hours')::INTERVAL
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, test_id, hours_back)

            if not row:
                return {"test_id": test_id, "error": "No data found"}

            return {
                "test_id": test_id,
                "time_window_hours": hours_back,
                "total_samples": (row['control_count'] or 0) + (row['treatment_count'] or 0) + (row['shadow_count'] or 0),
                "control_count": row['control_count'] or 0,
                "treatment_count": row['treatment_count'] or 0,
                "shadow_count": row['shadow_count'] or 0,
                "metrics": {
                    "control": {
                        "avg_confidence": float(row['avg_control_confidence']) if row['avg_control_confidence'] else None,
                        "avg_latency_ms": float(row['avg_control_latency']) if row['avg_control_latency'] else None,
                        "stddev_latency_ms": float(row['stddev_control_latency']) if row['stddev_control_latency'] else None,
                    },
                    "treatment": {
                        "avg_confidence": float(row['avg_treatment_confidence']) if row['avg_treatment_confidence'] else None,
                        "avg_latency_ms": float(row['avg_treatment_latency']) if row['avg_treatment_latency'] else None,
                        "stddev_latency_ms": float(row['stddev_treatment_latency']) if row['stddev_treatment_latency'] else None,
                    },
                    "comparison": {
                        "agreement_rate": float(row['agreement_rate']) if row['agreement_rate'] else None,
                        "avg_confidence_delta": float(row['avg_confidence_delta']) if row['avg_confidence_delta'] else None,
                        "stddev_confidence_delta": float(row['stddev_confidence_delta']) if row['stddev_confidence_delta'] else None,
                        "avg_latency_delta_ms": float(row['avg_latency_delta']) if row['avg_latency_delta'] else None,
                        "p95_latency_delta_ms": float(row['p95_latency_delta']) if row['p95_latency_delta'] else None,
                    }
                }
            }

    async def verify_traffic_split(
        self,
        test_id: str,
        expected_split: float,
        hours_back: int = 1,
        tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Verify that traffic split is accurate.

        Args:
            test_id: Test identifier
            expected_split: Expected treatment percentage (0.0-1.0)
            hours_back: Hours to analyze
            tolerance: Acceptable deviation (default ±2%)

        Returns:
            Verification results
        """
        if not self.pool:
            raise RuntimeError("Storage backend not connected")

        query = """
            SELECT
                COUNT(*) FILTER (WHERE assigned_group = 'control') AS control_count,
                COUNT(*) FILTER (WHERE assigned_group = 'treatment') AS treatment_count,
                COUNT(*) FILTER (WHERE assigned_group IN ('control', 'treatment')) AS total_count
            FROM ab_test_results
            WHERE test_id = $1
            AND time > NOW() - ($2 || ' hours')::INTERVAL
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, test_id, hours_back)

            total = row['total_count'] or 0
            treatment = row['treatment_count'] or 0

            if total == 0:
                return {
                    "test_id": test_id,
                    "error": "No samples found in time window"
                }

            actual_split = treatment / total
            delta = abs(actual_split - expected_split)
            within_tolerance = delta <= tolerance

            return {
                "test_id": test_id,
                "time_window_hours": hours_back,
                "sample_size": total,
                "expected_split": expected_split,
                "actual_split": actual_split,
                "delta": delta,
                "delta_pct": delta * 100,
                "within_tolerance": within_tolerance,
                "tolerance": tolerance
            }

    async def get_action_distribution(
        self,
        test_id: str,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get distribution of actions for control vs treatment.

        Args:
            test_id: Test identifier
            hours_back: Hours to analyze

        Returns:
            Action distribution statistics
        """
        if not self.pool:
            raise RuntimeError("Storage backend not connected")

        query = """
            SELECT
                assigned_group,
                control_action,
                treatment_action,
                COUNT(*) as count
            FROM ab_test_results
            WHERE test_id = $1
            AND time > NOW() - ($2 || ' hours')::INTERVAL
            GROUP BY assigned_group, control_action, treatment_action
            ORDER BY assigned_group, count DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, test_id, hours_back)

            distribution = {
                "control": {"BUY": 0, "SELL": 0, "HOLD": 0},
                "treatment": {"BUY": 0, "SELL": 0, "HOLD": 0}
            }

            for row in rows:
                group = row['assigned_group']
                action = row['control_action'] if group == 'control' else row['treatment_action']
                count = row['count']

                if action and action in distribution.get(group, {}):
                    distribution[group][action] += count

            return {
                "test_id": test_id,
                "time_window_hours": hours_back,
                "distribution": distribution
            }


# Singleton instance
_storage_backend: Optional[ABTestStorageBackend] = None


async def get_storage_backend() -> ABTestStorageBackend:
    """Get or create storage backend singleton."""
    global _storage_backend

    if _storage_backend is None:
        _storage_backend = ABTestStorageBackend()
        await _storage_backend.connect()

    return _storage_backend


async def shutdown_storage_backend():
    """Shutdown storage backend."""
    global _storage_backend

    if _storage_backend:
        await _storage_backend.disconnect()
        _storage_backend = None
