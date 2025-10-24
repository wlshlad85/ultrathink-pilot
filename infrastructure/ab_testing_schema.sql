-- A/B Testing Schema for TimescaleDB
-- Tracks A/B test configurations and results for safe model rollouts

-- A/B Test Configurations
CREATE TABLE IF NOT EXISTS ab_test_configs (
    test_id VARCHAR(100) PRIMARY KEY,
    mode VARCHAR(50) NOT NULL, -- 'traffic_split', 'shadow', 'disabled'
    control_model VARCHAR(255) NOT NULL,
    treatment_model VARCHAR(255) NOT NULL,
    traffic_split DOUBLE PRECISION NOT NULL CHECK (traffic_split >= 0 AND traffic_split <= 1),
    enabled BOOLEAN DEFAULT TRUE,
    shadow_mode BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),
    metadata JSONB
);

CREATE INDEX idx_ab_configs_enabled ON ab_test_configs(enabled) WHERE enabled = TRUE;
CREATE INDEX idx_ab_configs_created_at ON ab_test_configs(created_at DESC);

-- A/B Test Results (Hypertable for time-series data)
CREATE TABLE IF NOT EXISTS ab_test_results (
    time TIMESTAMPTZ NOT NULL,
    test_id VARCHAR(100) NOT NULL,
    request_id VARCHAR(100) NOT NULL,
    assigned_group VARCHAR(20) NOT NULL, -- 'control', 'treatment', 'shadow'

    -- Symbol context
    symbol VARCHAR(20),

    -- Control model results
    control_model VARCHAR(255) NOT NULL,
    control_action VARCHAR(10),
    control_confidence DOUBLE PRECISION,
    control_latency_ms DOUBLE PRECISION,

    -- Treatment model results (null if not in shadow mode or treatment group)
    treatment_model VARCHAR(255),
    treatment_action VARCHAR(10),
    treatment_confidence DOUBLE PRECISION,
    treatment_latency_ms DOUBLE PRECISION,

    -- Comparison metrics (for shadow mode)
    actions_match BOOLEAN,
    confidence_delta DOUBLE PRECISION,
    latency_delta_ms DOUBLE PRECISION,

    -- Feature data (optional, can be large)
    features JSONB,

    -- Additional metadata
    metadata JSONB
);

-- Convert to hypertable for automatic partitioning and efficient time-series queries
SELECT create_hypertable('ab_test_results', 'time', if_not_exists => TRUE);

-- Indexes for fast queries
CREATE INDEX idx_ab_results_test_id ON ab_test_results(test_id, time DESC);
CREATE INDEX idx_ab_results_request_id ON ab_test_results(request_id);
CREATE INDEX idx_ab_results_assigned_group ON ab_test_results(test_id, assigned_group, time DESC);
CREATE INDEX idx_ab_results_symbol ON ab_test_results(symbol, time DESC) WHERE symbol IS NOT NULL;
CREATE INDEX idx_ab_results_actions_match ON ab_test_results(test_id, actions_match) WHERE actions_match IS NOT NULL;

-- Continuous aggregate for hourly A/B test metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS ab_test_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    test_id,
    assigned_group,
    time_bucket('1 hour', time) AS bucket,

    -- Sample sizes
    COUNT(*) AS sample_count,

    -- Control metrics
    AVG(control_confidence) AS avg_control_confidence,
    AVG(control_latency_ms) AS avg_control_latency_ms,
    STDDEV(control_latency_ms) AS stddev_control_latency_ms,

    -- Treatment metrics
    AVG(treatment_confidence) AS avg_treatment_confidence,
    AVG(treatment_latency_ms) AS avg_treatment_latency_ms,
    STDDEV(treatment_latency_ms) AS stddev_treatment_latency_ms,

    -- Comparison metrics (shadow mode)
    SUM(CASE WHEN actions_match = TRUE THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS agreement_rate,
    AVG(confidence_delta) AS avg_confidence_delta,
    AVG(latency_delta_ms) AS avg_latency_delta_ms,

    -- Action distribution
    SUM(CASE WHEN control_action = 'BUY' THEN 1 ELSE 0 END) AS control_buy_count,
    SUM(CASE WHEN control_action = 'SELL' THEN 1 ELSE 0 END) AS control_sell_count,
    SUM(CASE WHEN control_action = 'HOLD' THEN 1 ELSE 0 END) AS control_hold_count,
    SUM(CASE WHEN treatment_action = 'BUY' THEN 1 ELSE 0 END) AS treatment_buy_count,
    SUM(CASE WHEN treatment_action = 'SELL' THEN 1 ELSE 0 END) AS treatment_sell_count,
    SUM(CASE WHEN treatment_action = 'HOLD' THEN 1 ELSE 0 END) AS treatment_hold_count

FROM ab_test_results
GROUP BY test_id, assigned_group, bucket
WITH NO DATA;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('ab_test_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily aggregate for longer-term analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS ab_test_metrics_daily
WITH (timescaledb.continuous) AS
SELECT
    test_id,
    assigned_group,
    time_bucket('1 day', time) AS bucket,

    COUNT(*) AS sample_count,

    -- Overall metrics
    AVG(control_confidence) AS avg_control_confidence,
    AVG(treatment_confidence) AS avg_treatment_confidence,
    AVG(control_latency_ms) AS avg_control_latency_ms,
    AVG(treatment_latency_ms) AS avg_treatment_latency_ms,

    -- Agreement rate
    SUM(CASE WHEN actions_match = TRUE THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS agreement_rate,

    -- Statistical measures
    STDDEV(confidence_delta) AS stddev_confidence_delta,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence_delta) AS median_confidence_delta,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_delta_ms) AS p95_latency_delta_ms

FROM ab_test_results
WHERE actions_match IS NOT NULL  -- Only shadow mode results
GROUP BY test_id, assigned_group, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('ab_test_metrics_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Data retention policy (keep A/B test results for 90 days)
SELECT add_retention_policy('ab_test_results', INTERVAL '90 days', if_not_exists => TRUE);

-- Helper function to get A/B test summary
CREATE OR REPLACE FUNCTION get_ab_test_summary(test_id_param VARCHAR)
RETURNS TABLE(
    metric VARCHAR,
    control_value DOUBLE PRECISION,
    treatment_value DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    delta_pct DOUBLE PRECISION
) AS $$
WITH stats AS (
    SELECT
        COUNT(*) FILTER (WHERE assigned_group = 'control') AS control_count,
        COUNT(*) FILTER (WHERE assigned_group = 'treatment') AS treatment_count,
        COUNT(*) FILTER (WHERE assigned_group = 'shadow') AS shadow_count,

        AVG(control_confidence) AS avg_control_conf,
        AVG(treatment_confidence) AS avg_treatment_conf,

        AVG(control_latency_ms) AS avg_control_latency,
        AVG(treatment_latency_ms) AS avg_treatment_latency,

        SUM(CASE WHEN actions_match = TRUE THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(SUM(CASE WHEN actions_match IS NOT NULL THEN 1 ELSE 0 END), 0) AS agreement
    FROM ab_test_results
    WHERE test_id = test_id_param
    AND time > NOW() - INTERVAL '24 hours'
)
SELECT 'sample_size_control'::VARCHAR, control_count, treatment_count,
       treatment_count - control_count,
       CASE WHEN control_count > 0 THEN (treatment_count - control_count)::FLOAT / control_count * 100 ELSE NULL END
FROM stats
UNION ALL
SELECT 'avg_confidence', avg_control_conf, avg_treatment_conf,
       avg_treatment_conf - avg_control_conf,
       CASE WHEN avg_control_conf > 0 THEN (avg_treatment_conf - avg_control_conf) / avg_control_conf * 100 ELSE NULL END
FROM stats
UNION ALL
SELECT 'avg_latency_ms', avg_control_latency, avg_treatment_latency,
       avg_treatment_latency - avg_control_latency,
       CASE WHEN avg_control_latency > 0 THEN (avg_treatment_latency - avg_control_latency) / avg_control_latency * 100 ELSE NULL END
FROM stats
UNION ALL
SELECT 'agreement_rate', agreement, NULL, NULL, NULL
FROM stats;
$$ LANGUAGE SQL STABLE;

-- Helper function to check if traffic split is accurate
CREATE OR REPLACE FUNCTION verify_traffic_split(test_id_param VARCHAR, hours_back INTEGER DEFAULT 1)
RETURNS TABLE(
    expected_split DOUBLE PRECISION,
    actual_split DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    within_tolerance BOOLEAN
) AS $$
WITH split_data AS (
    SELECT
        tc.traffic_split AS expected,
        COUNT(*) FILTER (WHERE r.assigned_group = 'treatment')::FLOAT /
            NULLIF(COUNT(*) FILTER (WHERE r.assigned_group IN ('control', 'treatment')), 0) AS actual
    FROM ab_test_configs tc
    LEFT JOIN ab_test_results r ON tc.test_id = r.test_id
    WHERE tc.test_id = test_id_param
    AND r.time > NOW() - (hours_back || ' hours')::INTERVAL
    GROUP BY tc.traffic_split
)
SELECT
    expected,
    actual,
    ABS(actual - expected) AS delta,
    ABS(actual - expected) < 0.02 AS within_tolerance  -- Within ±2%
FROM split_data;
$$ LANGUAGE SQL STABLE;

-- Comments for documentation
COMMENT ON TABLE ab_test_configs IS 'A/B test configurations for model deployment experiments';
COMMENT ON TABLE ab_test_results IS 'A/B test results (Hypertable) - stores predictions from both control and treatment models';
COMMENT ON MATERIALIZED VIEW ab_test_metrics_hourly IS 'Hourly aggregated A/B test metrics for monitoring';
COMMENT ON MATERIALIZED VIEW ab_test_metrics_daily IS 'Daily aggregated A/B test metrics for trend analysis';

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE ab_test_configs TO ultrathink;
GRANT ALL PRIVILEGES ON TABLE ab_test_results TO ultrathink;
GRANT SELECT ON ab_test_metrics_hourly TO ultrathink;
GRANT SELECT ON ab_test_metrics_daily TO ultrathink;

-- Example: Create a sample A/B test
INSERT INTO ab_test_configs (test_id, mode, control_model, treatment_model, traffic_split, description)
VALUES (
    'test_shadow_new_model',
    'shadow',
    'universal',
    'bull_specialist_v2',
    0.05,
    'Shadow mode test for new bull specialist model - comparing predictions without affecting production'
) ON CONFLICT (test_id) DO NOTHING;

-- Verify setup
SELECT 'A/B Testing schema created successfully' AS status;
SELECT format('A/B test hypertables: %s', COUNT(*))
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'ab_test_results';
