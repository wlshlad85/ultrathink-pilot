#!/bin/bash

# kafka_topics.sh - Create Kafka topics for UltraThink Pilot trading system
# Author: event-architecture-specialist agent
# Date: 2025-10-24

set -e

LOG_FILE="/home/rich/ultrathink-pilot/agent-coordination/logs/event-architecture.log"
TIMESTAMP=

# Logging function
log() {
    echo "[] " | tee -a ""
}

log "========================================"
log "Starting Kafka topic creation"
log "========================================"

# Wait for Kafka cluster to be ready
log "Waiting for Kafka cluster to be ready..."
RETRIES=30
RETRY_COUNT=0

while [  -lt  ]; do
    if docker exec ultrathink-kafka-1 kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null; then
        log "Kafka broker 1 is ready"
        break
    fi
    RETRY_COUNT=1
    log "Waiting for Kafka... attempt /"
    sleep 2
done

if [  -eq  ]; then
    log "ERROR: Kafka cluster failed to start within timeout"
    exit 1
fi

# Topic configuration
declare -A TOPICS=(
    ["trading_decisions"]="3 2"
    ["market_data"]="5 2"
    ["forensics_events"]="2 2"
)

# Create topics
log "Creating Kafka topics..."

for topic in ""; do
    read -r partitions replication <<< ""
    
    log "Creating topic:  (partitions=, replication=)"
    
    if docker exec ultrathink-kafka-1 kafka-topics --create         --bootstrap-server localhost:9092         --topic ""         --partitions ""         --replication-factor ""         --config retention.ms=604800000         --if-not-exists 2>&1 | tee -a ""; then
        log "SUCCESS: Topic  created"
    else
        log "ERROR: Failed to create topic "
        exit 1
    fi
done

# Verify topics
log ""
log "Verifying topic creation..."
log ""

if docker exec ultrathink-kafka-1 kafka-topics --list     --bootstrap-server localhost:9092 2>&1 | tee -a ""; then
    log ""
    log "Topic list retrieved successfully"
else
    log "ERROR: Failed to list topics"
    exit 1
fi

# Describe topics for verification
log ""
log "Topic details:"
log ""

for topic in ""; do
    log "--- Topic:  ---"
    docker exec ultrathink-kafka-1 kafka-topics --describe         --bootstrap-server localhost:9092         --topic "" 2>&1 | tee -a ""
    log ""
done

log "========================================"
log "Kafka topic creation completed successfully"
log "========================================"

exit 0
