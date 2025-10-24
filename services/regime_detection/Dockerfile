# Probabilistic Regime Detection Service
# Agent: regime-detection-specialist
# Mission: Eliminate 15% portfolio disruption through probabilistic regime classification

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY probabilistic_regime_detector.py .
COPY regime_api.py .

# Create directories for model storage
RUN mkdir -p /app/models /app/logs

# Environment variables (overridable)
ENV TIMESCALEDB_HOST=timescaledb \
    TIMESCALEDB_PORT=5432 \
    TIMESCALEDB_DATABASE=ultrathink_experiments \
    TIMESCALEDB_USER=ultrathink \
    TIMESCALEDB_PASSWORD=changeme_in_production \
    PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run FastAPI application
CMD ["uvicorn", "regime_api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
