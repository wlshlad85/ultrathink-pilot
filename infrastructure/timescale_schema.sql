-- UltraThink Pilot - TimescaleDB Schema
-- Migration from SQLite to TimescaleDB for experiment tracking

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Experiments table (replaces SQLite experiments table)
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'ppo_agent', 'bull_specialist', 'bear_specialist', etc.
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'running', -- 'running', 'completed', 'failed', 'stopped'
    random_seed INTEGER,
    git_commit VARCHAR(40),
    tags TEXT[],
    notes TEXT
);

CREATE INDEX idx_experiments_name ON experiments(experiment_name);
CREATE INDEX idx_experiments_model_type ON experiments(model_type);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_created_at ON experiments(created_at DESC);
CREATE INDEX idx_experiments_tags ON experiments USING GIN(tags);

-- Hypertable for time-series metrics (automatic partitioning)
CREATE TABLE IF NOT EXISTS experiment_metrics (
    time TIMESTAMPTZ NOT NULL,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    step INTEGER,
    epoch INTEGER,
    episode INTEGER,
    metadata JSONB
);

-- Convert to hypertable (TimescaleDB-specific)
SELECT create_hypertable('experiment_metrics', 'time', if_not_exists => TRUE);

-- Indexes for fast queries
CREATE INDEX idx_exp_metrics_exp_id ON experiment_metrics(experiment_id, time DESC);
CREATE INDEX idx_exp_metrics_name ON experiment_metrics(experiment_id, metric_name, time DESC);
CREATE INDEX idx_exp_metrics_episode ON experiment_metrics(experiment_id, episode, time DESC) WHERE episode IS NOT NULL;

-- Model checkpoints tracking
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    checkpoint_path VARCHAR(500) NOT NULL,
    episode INTEGER,
    sharpe_ratio DOUBLE PRECISION,
    total_return DOUBLE PRECISION,
    validation_loss DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_best BOOLEAN DEFAULT FALSE,
    is_production BOOLEAN DEFAULT FALSE,
    file_size_mb DOUBLE PRECISION,
    model_hash VARCHAR(64),
    metadata JSONB
);

CREATE INDEX idx_checkpoints_exp_id ON model_checkpoints(experiment_id, created_at DESC);
CREATE INDEX idx_checkpoints_is_best ON model_checkpoints(experiment_id, is_best) WHERE is_best = TRUE;
CREATE INDEX idx_checkpoints_is_production ON model_checkpoints(is_production) WHERE is_production = TRUE;
CREATE INDEX idx_checkpoints_episode ON model_checkpoints(experiment_id, episode DESC);

-- Hyperparameters table (normalized from config JSONB)
CREATE TABLE IF NOT EXISTS experiment_hyperparameters (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    param_name VARCHAR(100) NOT NULL,
    param_value TEXT NOT NULL,
    param_type VARCHAR(20), -- 'int', 'float', 'str', 'bool'
    PRIMARY KEY (experiment_id, param_name)
);

CREATE INDEX idx_hyperparams_name ON experiment_hyperparameters(param_name);

-- Regime detection history (for probabilistic regime detector)
CREATE TABLE IF NOT EXISTS regime_history (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prob_bull DOUBLE PRECISION CHECK (prob_bull >= 0 AND prob_bull <= 1),
    prob_bear DOUBLE PRECISION CHECK (prob_bear >= 0 AND prob_bear <= 1),
    prob_sideways DOUBLE PRECISION CHECK (prob_sideways >= 0 AND prob_sideways <= 1),
    entropy DOUBLE PRECISION, -- uncertainty measure
    detected_regime VARCHAR(20), -- 'bull', 'bear', 'sideways', 'mixed'
    metadata JSONB,
    CONSTRAINT valid_probabilities CHECK (
        ABS((prob_bull + prob_bear + prob_sideways) - 1.0) < 0.001
    )
);

SELECT create_hypertable('regime_history', 'time', if_not_exists => TRUE);
CREATE INDEX idx_regime_symbol ON regime_history(symbol, time DESC);

-- Dataset versions (for tracking data used in experiments)
CREATE TABLE IF NOT EXISTS dataset_versions (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(100) NOT NULL,
    symbol VARCHAR(20),
    start_date DATE,
    end_date DATE,
    record_count INTEGER,
    data_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    UNIQUE(dataset_name, version)
);

CREATE INDEX idx_dataset_symbol ON dataset_versions(symbol);
CREATE INDEX idx_dataset_dates ON dataset_versions(start_date, end_date);

-- Experiment-Dataset relationship (many-to-many)
CREATE TABLE IF NOT EXISTS experiment_datasets (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    usage_type VARCHAR(50), -- 'train', 'validation', 'test'
    PRIMARY KEY (experiment_id, dataset_id, usage_type)
);

-- Trading decisions log (for forensics and analysis)
CREATE TABLE IF NOT EXISTS trading_decisions (
    time TIMESTAMPTZ NOT NULL,
    experiment_id INTEGER REFERENCES experiments(id),
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    quantity DOUBLE PRECISION,
    price DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    portfolio_value DOUBLE PRECISION,
    regime VARCHAR(20),
    features JSONB,
    model_version VARCHAR(100),
    metadata JSONB
);

SELECT create_hypertable('trading_decisions', 'time', if_not_exists => TRUE);
CREATE INDEX idx_decisions_exp_id ON trading_decisions(experiment_id, time DESC);
CREATE INDEX idx_decisions_symbol ON trading_decisions(symbol, time DESC);
CREATE INDEX idx_decisions_action ON trading_decisions(action, time DESC);

-- Continuous aggregates for metrics (automatic materialized views)
CREATE MATERIALIZED VIEW IF NOT EXISTS experiment_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    experiment_id,
    metric_name,
    time_bucket('1 hour', time) AS bucket,
    AVG(metric_value) AS avg_value,
    MAX(metric_value) AS max_value,
    MIN(metric_value) AS min_value,
    STDDEV(metric_value) AS stddev_value,
    COUNT(*) AS count
FROM experiment_metrics
GROUP BY experiment_id, metric_name, bucket
WITH NO DATA;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('experiment_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Data retention policies (keep detailed data for 90 days, compressed after 7 days)
SELECT add_retention_policy('experiment_metrics', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('trading_decisions', INTERVAL '365 days', if_not_exists => TRUE);

-- Compression policies (compress old data to save space)
-- DISABLED: SELECT add_compression_policy('experiment_metrics', INTERVAL '7 days', if_not_exists => TRUE);
-- DISABLED: SELECT add_compression_policy('regime_history', INTERVAL '30 days', if_not_exists => TRUE);

-- Helper function to get latest metric value
CREATE OR REPLACE FUNCTION get_latest_metric(exp_id INTEGER, metric VARCHAR)
RETURNS DOUBLE PRECISION AS $$
    SELECT metric_value
    FROM experiment_metrics
    WHERE experiment_id = exp_id AND metric_name = metric
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Helper function to get experiment summary
CREATE OR REPLACE FUNCTION get_experiment_summary(exp_id INTEGER)
RETURNS TABLE(
    metric_name VARCHAR,
    latest_value DOUBLE PRECISION,
    avg_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    min_value DOUBLE PRECISION
) AS $$
    SELECT
        metric_name,
        (SELECT metric_value FROM experiment_metrics WHERE experiment_id = exp_id AND metric_name = m.metric_name ORDER BY time DESC LIMIT 1) AS latest_value,
        AVG(metric_value) AS avg_value,
        MAX(metric_value) AS max_value,
        MIN(metric_value) AS min_value
    FROM experiment_metrics m
    WHERE experiment_id = exp_id
    GROUP BY metric_name;
$$ LANGUAGE SQL STABLE;

-- Create user for MLflow
CREATE USER mlflow WITH PASSWORD 'mlflow_changeme';
GRANT ALL PRIVILEGES ON DATABASE ultrathink_experiments TO mlflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow;

-- Grant permissions to ultrathink user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ultrathink;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ultrathink;

-- Comments for documentation
COMMENT ON TABLE experiments IS 'Main experiments tracking table - replaces SQLite experiments';
COMMENT ON TABLE experiment_metrics IS 'Time-series metrics (Hypertable) - supports 20+ concurrent writes';
COMMENT ON TABLE model_checkpoints IS 'Model checkpoint versioning with automated cleanup';
COMMENT ON TABLE regime_history IS 'Probabilistic regime detection history (Hypertable)';
COMMENT ON TABLE dataset_versions IS 'Dataset versioning for reproducibility';
COMMENT ON TABLE trading_decisions IS 'Trading decision audit trail (Hypertable)';

-- Initial data for testing
INSERT INTO experiments (experiment_name, model_type, config, status, tags)
VALUES ('test_migration', 'ppo_agent', '{"learning_rate": 0.0003}', 'completed', ARRAY['test'])
ON CONFLICT DO NOTHING;

-- Verify setup
SELECT 'TimescaleDB schema created successfully' AS status;
SELECT format('Hypertables created: %s', COUNT(*)) FROM timescaledb_information.hypertables;
