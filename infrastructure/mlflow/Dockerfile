# Custom MLflow Image with PostgreSQL Support
# Base: Official MLflow v2.9.2
# Purpose: Add psycopg2-binary for TimescaleDB connectivity
# Agent: database-migration-specialist (Agent 8)
# Date: 2025-10-25

FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install PostgreSQL adapter for TimescaleDB backend
RUN pip install --no-cache-dir psycopg2-binary==2.9.9

# Install additional dependencies for better performance
RUN pip install --no-cache-dir \
    sqlalchemy==2.0.23 \
    alembic==1.12.1

# Verify installation
RUN python -c "import psycopg2; print('psycopg2 version:', psycopg2.__version__)"
RUN python -c "import sqlalchemy; print('SQLAlchemy version:', sqlalchemy.__version__)"

# Create directory for artifacts
RUN mkdir -p /mlflow/artifacts && chmod 777 /mlflow/artifacts

# Health check script
COPY healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Keep original entrypoint
# Default command will be overridden by docker-compose
