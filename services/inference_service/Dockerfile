FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch and GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Create models directory (will be mounted as volume)
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV MOCK_DATA_SERVICE=true
ENV MOCK_REGIME_DETECTION=true
ENV MOCK_META_CONTROLLER=true
ENV MOCK_RISK_MANAGER=true

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with single worker for GPU
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
