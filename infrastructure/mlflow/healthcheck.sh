#!/bin/bash
# MLflow Health Check Script
# Verifies MLflow server is responding correctly

set -e

# Check if MLflow server is responding
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)

if [ "$response" = "200" ]; then
    echo "MLflow is healthy"
    exit 0
else
    echo "MLflow health check failed with status: $response"
    exit 1
fi
