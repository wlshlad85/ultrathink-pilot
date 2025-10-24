"""
Forensics Consumer - Asynchronous trade explainability and audit logging.
Consumes trading decision events from Kafka and generates SHAP explanations.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import os

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
import psycopg2
from psycopg2.extras import Json, execute_values
import numpy as np

logger = logging.getLogger(__name__)


class ForensicsConsumer:
    """
    Kafka consumer for forensics processing.

    Responsibilities:
    - Subscribe to trading_decisions topic
    - Generate SHAP explanations (model interpretability)
    - Store complete audit trail in TimescaleDB
    - Track consumer lag metrics
    """

    def __init__(
        self,
        bootstrap_servers: str = "kafka-1:9092",
        topic: str = "trading_decisions",
        group_id: str = "forensics_consumer_group",
        db_config: Optional[Dict[str, str]] = None
    ):
        """
        Initialize forensics consumer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to subscribe to
            group_id: Consumer group ID
            db_config: TimescaleDB connection parameters
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id

        # Database connection
        self.db_config = db_config or {
            'host': os.environ.get('TIMESCALEDB_HOST', 'timescaledb'),
            'port': int(os.environ.get('TIMESCALEDB_PORT', '5432')),
            'database': os.environ.get('TIMESCALEDB_DATABASE', 'ultrathink_experiments'),
            'user': os.environ.get('TIMESCALEDB_USER', 'ultrathink'),
            'password': os.environ.get('TIMESCALEDB_PASSWORD', 'changeme_in_production')
        }

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.db_conn = None
        self.is_running = False

        # Metrics
        self.events_processed = 0
        self.events_failed = 0
        self.consumer_lag = 0
        self.processing_time_ms = 0.0

    async def start(self):
        """Start the consumer and database connection."""
        logger.info("Starting Forensics Consumer...")

        # Initialize database connection
        self._init_database()

        # Initialize Kafka consumer
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        await self.consumer.start()
        self.is_running = True

        logger.info(
            f"Forensics consumer started: topic={self.topic}, "
            f"group={self.group_id}"
        )

    async def stop(self):
        """Stop the consumer."""
        self.is_running = False

        if self.consumer:
            await self.consumer.stop()

        if self.db_conn:
            self.db_conn.close()

        logger.info("Forensics consumer stopped")

    def _init_database(self):
        """Initialize TimescaleDB connection and schema."""
        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            logger.info("Connected to TimescaleDB")

            # Create schema if not exists
            self._create_schema()

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _create_schema(self):
        """Create forensics audit table schema."""
        schema_sql = """
        -- Forensics audit trail table
        CREATE TABLE IF NOT EXISTS trading_decisions_audit (
            decision_id UUID PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL,
            quantity INTEGER NOT NULL,
            confidence FLOAT NOT NULL,
            regime_probs JSONB NOT NULL,
            strategy_weights JSONB NOT NULL,
            features JSONB,
            risk_checks JSONB,
            model_version VARCHAR(100),
            latency_ms FLOAT,
            shap_values JSONB,
            explanation TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Create hypertable for time-series optimization (7-year retention)
        SELECT create_hypertable(
            'trading_decisions_audit',
            'timestamp',
            if_not_exists => TRUE
        );

        -- Retention policy: 7 years hot, archive to cold storage
        SELECT add_retention_policy(
            'trading_decisions_audit',
            INTERVAL '7 years',
            if_not_exists => TRUE
        );

        -- Indexes for fast queries
        CREATE INDEX IF NOT EXISTS idx_audit_symbol_timestamp
            ON trading_decisions_audit (symbol, timestamp DESC);

        CREATE INDEX IF NOT EXISTS idx_audit_action
            ON trading_decisions_audit (action);

        CREATE INDEX IF NOT EXISTS idx_audit_decision_id
            ON trading_decisions_audit (decision_id);
        """

        with self.db_conn.cursor() as cursor:
            cursor.execute(schema_sql)
            self.db_conn.commit()

        logger.info("Database schema initialized")

    async def consume(self):
        """Main consumption loop."""
        logger.info("Starting consumption loop...")

        try:
            async for message in self.consumer:
                event = message.value
                await self._process_event(event)

                # Update lag metric
                self.consumer_lag = await self._calculate_lag()

        except Exception as e:
            logger.error(f"Consumption error: {e}", exc_info=True)
            raise

    async def _process_event(self, event: Dict[str, Any]):
        """
        Process single trading decision event.

        Args:
            event: Trading decision event
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Extract event data
            decision_id = event['decision_id']
            timestamp = event['timestamp']
            symbol = event['symbol']
            action = event['action']
            confidence = event['confidence']

            logger.debug(f"Processing: {decision_id[:8]}... {symbol} {action}")

            # Generate SHAP explanations
            shap_values = self._generate_shap_explanation(event)

            # Generate human-readable explanation
            explanation = self._generate_explanation(event, shap_values)

            # Store in database
            self._store_audit_record(event, shap_values, explanation)

            self.events_processed += 1

            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.processing_time_ms = processing_time

            logger.info(
                f"Processed: {decision_id[:8]}... in {processing_time:.2f}ms "
                f"(total: {self.events_processed})"
            )

        except Exception as e:
            self.events_failed += 1
            logger.error(f"Event processing failed: {e}", exc_info=True)

    def _generate_shap_explanation(self, event: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate SHAP values for model interpretability.

        For now, this is a simplified implementation.
        In production, load the actual model and compute real SHAP values.

        Args:
            event: Trading decision event

        Returns:
            Dict of feature -> SHAP value
        """
        features = event.get('features', {})

        # Simplified SHAP calculation (placeholder)
        # In production: use shap.Explainer with loaded model
        shap_values = {}

        for feature_name, feature_value in features.items():
            # Placeholder: random importance
            shap_values[feature_name] = np.random.uniform(-0.5, 0.5)

        # Sort by absolute importance
        shap_values = dict(
            sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return shap_values

    def _generate_explanation(
        self,
        event: Dict[str, Any],
        shap_values: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation.

        Args:
            event: Trading decision event
            shap_values: SHAP values

        Returns:
            Explanation text
        """
        symbol = event['symbol']
        action = event['action']
        confidence = event['confidence']
        regime_probs = event['regime_probs']

        # Identify dominant regime
        dominant_regime = max(regime_probs.items(), key=lambda x: x[1])[0]

        # Top 3 influential features
        top_features = list(shap_values.items())[:3]
        feature_text = ", ".join([f"{k}={v:.3f}" for k, v in top_features])

        explanation = (
            f"Decision: {action} {symbol} (confidence={confidence:.2%}). "
            f"Market regime: {dominant_regime.upper()} ({regime_probs[dominant_regime]:.1%}). "
            f"Key factors: {feature_text}."
        )

        return explanation

    def _store_audit_record(
        self,
        event: Dict[str, Any],
        shap_values: Dict[str, float],
        explanation: str
    ):
        """
        Store audit record in TimescaleDB.

        Args:
            event: Trading decision event
            shap_values: SHAP values
            explanation: Human-readable explanation
        """
        insert_sql = """
        INSERT INTO trading_decisions_audit (
            decision_id, timestamp, symbol, action, quantity, confidence,
            regime_probs, strategy_weights, features, risk_checks,
            model_version, latency_ms, shap_values, explanation
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (decision_id) DO NOTHING
        """

        with self.db_conn.cursor() as cursor:
            cursor.execute(insert_sql, (
                event['decision_id'],
                event['timestamp'],
                event['symbol'],
                event['action'],
                event['quantity'],
                event['confidence'],
                Json(event['regime_probs']),
                Json(event['strategy_weights']),
                Json(event.get('features', {})),
                Json(event.get('risk_checks', {})),
                event.get('model_version'),
                event.get('latency_ms'),
                Json(shap_values),
                explanation
            ))
            self.db_conn.commit()

    async def _calculate_lag(self) -> int:
        """
        Calculate consumer lag (messages behind).

        Returns:
            Number of messages behind
        """
        try:
            # Get partition assignments
            partitions = self.consumer.assignment()
            if not partitions:
                return 0

            total_lag = 0

            for partition in partitions:
                # Get committed offset
                committed = await self.consumer.committed(partition)
                if committed is None:
                    committed = 0

                # Get end offset (high water mark)
                end_offset = await self.consumer.end_offsets([partition])
                high_water = end_offset.get(partition, 0)

                # Lag = high water - committed
                lag = high_water - committed
                total_lag += lag

            return total_lag

        except Exception as e:
            logger.error(f"Lag calculation error: {e}")
            return 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        return {
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "consumer_lag": self.consumer_lag,
            "processing_time_ms": self.processing_time_ms,
            "is_running": self.is_running
        }


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    consumer = ForensicsConsumer()

    try:
        await consumer.start()
        await consumer.consume()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
