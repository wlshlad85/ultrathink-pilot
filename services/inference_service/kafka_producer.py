"""
Kafka Producer for Trading Decision Events.
Async, non-blocking event emission for forensics decoupling.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4
import os

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from models import PredictResponse

logger = logging.getLogger(__name__)


class TradingEventProducer:
    """
    Async Kafka producer for trading decision events.

    Features:
    - Non-blocking async emission
    - In-memory buffering on Kafka unavailable
    - JSON serialization with compression
    - Automatic retries with exponential backoff
    """

    def __init__(
        self,
        bootstrap_servers: str = "kafka-1:9092",
        topic: str = "trading_decisions",
        buffer_size: int = 10000,
        compression_type: str = "gzip"
    ):
        """
        Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name for trading decisions
            buffer_size: Max events to buffer when Kafka unavailable
            compression_type: Compression algorithm (gzip, snappy, lz4, zstd)
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.buffer_size = buffer_size
        self.compression_type = compression_type

        self.producer: Optional[AIOKafkaProducer] = None
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.is_connected = False
        self._lock = asyncio.Lock()

        # Metrics
        self.events_sent = 0
        self.events_buffered = 0
        self.events_dropped = 0
        self.connection_failures = 0

    async def start(self):
        """Start the Kafka producer and connection."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type=self.compression_type,
                acks='all',  # Wait for all replicas
                max_in_flight_requests_per_connection=5,
                request_timeout_ms=5000,
                retry_backoff_ms=100,
                max_batch_size=65536,  # 64KB batches
                linger_ms=10  # Wait up to 10ms to batch messages
            )

            await self.producer.start()
            self.is_connected = True
            logger.info(f"Kafka producer started: {self.bootstrap_servers}, topic={self.topic}")

            # Start background task to flush buffered events
            asyncio.create_task(self._flush_buffer())

        except Exception as e:
            self.connection_failures += 1
            logger.error(f"Failed to start Kafka producer: {e}")
            self.is_connected = False

    async def stop(self):
        """Stop the Kafka producer."""
        if self.producer:
            try:
                await self.producer.stop()
                self.is_connected = False
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping producer: {e}")

    async def emit_decision(
        self,
        response: PredictResponse,
        features: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Emit trading decision to Kafka (async, non-blocking).

        This method returns immediately, even if Kafka is unavailable.
        Events are buffered in memory if Kafka is down.

        Args:
            response: Prediction response
            features: Optional feature dict for forensics

        Returns:
            True if sent/buffered successfully, False if dropped
        """
        try:
            # Build event payload matching schema from technical-spec.md
            event = self._build_event(response, features)

            # Try to send immediately (non-blocking)
            if self.is_connected:
                asyncio.create_task(self._send_event(event))
                return True
            else:
                # Buffer event if Kafka unavailable
                return await self._buffer_event(event)

        except Exception as e:
            logger.error(f"Error emitting decision: {e}")
            return False

    def _build_event(
        self,
        response: PredictResponse,
        features: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Build trading decision event payload."""
        decision_id = str(uuid4())

        # Extract regime probs
        regime_probs = {
            "bull": response.regime_probabilities.bull,
            "bear": response.regime_probabilities.bear,
            "sideways": response.regime_probabilities.sideways
        }

        # Extract strategy weights
        strategy_weights = {
            "bull_specialist": response.strategy_weights.bull_specialist,
            "bear_specialist": response.strategy_weights.bear_specialist,
            "sideways_specialist": response.strategy_weights.sideways_specialist
        }

        # Risk checks
        risk_checks = {
            "position_limit_ok": True,
            "concentration_ok": True,
            "correlation_ok": True
        }

        if response.risk_validation:
            risk_checks["position_limit_ok"] = response.risk_validation.approved

        event = {
            "event_type": "trading_decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_id": decision_id,
            "symbol": response.symbol,
            "action": response.action,
            "quantity": response.recommended_quantity,
            "confidence": response.confidence,
            "regime_probs": regime_probs,
            "strategy_weights": strategy_weights,
            "features": features or {},
            "risk_checks": risk_checks,
            "model_version": response.metadata.model_version,
            "latency_ms": response.metadata.latency_ms
        }

        return event

    async def _send_event(self, event: Dict[str, Any]):
        """Send event to Kafka (async, with retries)."""
        try:
            # Send to Kafka (non-blocking)
            await self.producer.send_and_wait(
                self.topic,
                value=event,
                key=event['symbol'].encode('utf-8')
            )

            self.events_sent += 1

            logger.debug(
                f"Event sent: {event['symbol']} {event['action']} "
                f"(decision_id={event['decision_id'][:8]}...)"
            )

        except KafkaError as e:
            self.connection_failures += 1
            self.is_connected = False
            logger.error(f"Kafka send error: {e}")

            # Buffer failed event
            await self._buffer_event(event)

        except Exception as e:
            logger.error(f"Unexpected send error: {e}")

    async def _buffer_event(self, event: Dict[str, Any]) -> bool:
        """Buffer event in memory when Kafka unavailable."""
        try:
            self.buffer.put_nowait(event)
            self.events_buffered += 1
            logger.warning(
                f"Event buffered (queue size: {self.buffer.qsize()}/{self.buffer_size})"
            )
            return True

        except asyncio.QueueFull:
            self.events_dropped += 1
            logger.error("Buffer full, event dropped!")
            return False

    async def _flush_buffer(self):
        """Background task to flush buffered events when Kafka recovers."""
        while True:
            try:
                if not self.buffer.empty() and self.is_connected:
                    # Try to flush buffer
                    while not self.buffer.empty():
                        event = await self.buffer.get()
                        await self._send_event(event)
                        await asyncio.sleep(0.01)  # Rate limit flushing

                    logger.info(f"Buffer flushed: {self.events_buffered} events")
                    self.events_buffered = 0

                # Check connection health
                if not self.is_connected:
                    logger.warning("Kafka disconnected, attempting reconnect...")
                    await self.start()

            except Exception as e:
                logger.error(f"Buffer flush error: {e}")

            # Sleep between flush attempts
            await asyncio.sleep(5)

    def get_metrics(self) -> Dict[str, int]:
        """Get producer metrics."""
        return {
            "events_sent": self.events_sent,
            "events_buffered": self.events_buffered,
            "events_dropped": self.events_dropped,
            "connection_failures": self.connection_failures,
            "buffer_size": self.buffer.qsize(),
            "is_connected": self.is_connected
        }


# Global producer instance
_producer: Optional[TradingEventProducer] = None


async def init_producer(
    bootstrap_servers: Optional[str] = None
) -> TradingEventProducer:
    """
    Initialize global Kafka producer.

    Args:
        bootstrap_servers: Kafka broker addresses (or use env var KAFKA_BOOTSTRAP_SERVERS)

    Returns:
        Initialized producer
    """
    global _producer

    if _producer is None:
        if bootstrap_servers is None:
            bootstrap_servers = os.environ.get(
                'KAFKA_BOOTSTRAP_SERVERS',
                'kafka-1:9092'
            )

        _producer = TradingEventProducer(bootstrap_servers=bootstrap_servers)
        await _producer.start()

    return _producer


async def get_producer() -> Optional[TradingEventProducer]:
    """Get global producer instance."""
    return _producer


async def shutdown_producer():
    """Shutdown global producer."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
