# UltraThink Pilot Trading System - Dockerfile
# Multi-stage build for optimized image size

# Build argument to choose CPU or GPU variant
ARG VARIANT=cpu
ARG PYTHON_VERSION=3.11

# Base stage - common dependencies
FROM python:${PYTHON_VERSION}-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# CPU variant stage
FROM base AS cpu-builder

# Install CPU version of PyTorch and other dependencies
RUN pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# GPU variant stage (PyTorch Nightly with CUDA 12.8 for RTX 50 series sm_120 support)
FROM base AS gpu-builder

# Install PyTorch Nightly with CUDA 12.8 (supports sm_120 Blackwell architecture)
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir -r requirements.txt

# Final stage - select based on build argument
FROM ${VARIANT}-builder AS final

# Copy application code
COPY agents/ ./agents/
COPY backtesting/ ./backtesting/
COPY rl/ ./rl/
COPY orchestration/ ./orchestration/
COPY policy/ ./policy/
COPY ml_persistence/ ./ml_persistence/
COPY eval/ ./eval/
COPY services/ ./services/
COPY tests/ ./tests/
COPY examples/ ./examples/
COPY *.py ./

# Create directories for data, models, and logs
RUN mkdir -p data/cache rl/models logs && \
    chmod -R 755 /app

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

USER trader

# Set Python path
ENV PYTHONPATH=/app

# Health check (optional - checks if Python imports work)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import gymnasium; import pandas" || exit 1

# Default command - run tests to verify installation
CMD ["pytest", "tests/", "-v"]

# Alternative entry points (can be overridden at runtime):
# Backtesting: docker run ultrathink-pilot python run_backtest.py --symbol BTC-USD --start 2023-01-01 --end 2024-01-01
# RL Training: docker run ultrathink-pilot python rl/train.py --episodes 100
# Shell access: docker run -it ultrathink-pilot /bin/bash
