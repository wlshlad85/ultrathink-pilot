-- ============================================================================
-- UltraThink Pilot - Database Validation Script
-- Purpose: Validate data integrity and continuous aggregate functionality
-- Usage: psql -U ultrathink -d ultrathink_experiments -f validate_database.sql
-- ============================================================================

\echo '========================================='
\echo 'UltraThink Pilot Database Validation'
\echo '========================================='

\echo ''
\echo '1. Checking TimescaleDB Extension...'
SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';

\echo ''
\echo '2. Verifying Hypertables...'
SELECT hypertable_schema, hypertable_name, num_dimensions, num_chunks
FROM timescaledb_information.hypertables
ORDER BY hypertable_name;

\echo ''
\echo '3. Validating Experiment Count...'
SELECT
    COUNT(*) as total_experiments,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
    COUNT(CASE WHEN status = 'running' THEN 1 END) as running,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
FROM experiments;

\echo ''
\echo '4. Validating Metrics Count...'
SELECT
    COUNT(*) as total_metrics,
    COUNT(DISTINCT experiment_id) as experiments_with_metrics,
    COUNT(DISTINCT metric_name) as unique_metric_names,
    MIN(time) as earliest_metric,
    MAX(time) as latest_metric
FROM experiment_metrics;

\echo ''
\echo '5. Validating Model Checkpoints...'
SELECT
    COUNT(*) as total_checkpoints,
    COUNT(DISTINCT experiment_id) as experiments_with_checkpoints,
    COUNT(CASE WHEN is_best THEN 1 END) as best_checkpoints,
    COUNT(CASE WHEN is_production THEN 1 END) as production_checkpoints
FROM model_checkpoints;

\echo ''
\echo '6. Checking Continuous Aggregates...'
SELECT
    view_name,
    materialization_hypertable_schema,
    materialization_hypertable_name
FROM timescaledb_information.continuous_aggregates
ORDER BY view_name;

\echo ''
\echo '7. Validating Continuous Aggregate Data...'

\echo ''
\echo '   7a. metrics_5min...'
SELECT
    COUNT(*) as total_buckets,
    COUNT(DISTINCT experiment_id) as experiments,
    COUNT(DISTINCT metric_name) as metric_names,
    MIN(bucket) as earliest_bucket,
    MAX(bucket) as latest_bucket
FROM metrics_5min;

\echo ''
\echo '   7b. episode_stats_hourly...'
SELECT
    COUNT(*) as total_buckets,
    COUNT(DISTINCT experiment_id) as experiments,
    MIN(bucket) as earliest_bucket,
    MAX(bucket) as latest_bucket,
    ROUND(AVG(avg_return)::numeric, 4) as overall_avg_return,
    ROUND(AVG(win_rate)::numeric, 2) as overall_win_rate
FROM episode_stats_hourly;

\echo ''
\echo '   7c. metrics_daily...'
SELECT
    COUNT(*) as total_buckets,
    COUNT(DISTINCT experiment_id) as experiments,
    COUNT(DISTINCT metric_name) as metric_names,
    MIN(bucket) as earliest_bucket,
    MAX(bucket) as latest_bucket
FROM metrics_daily;

\echo ''
\echo '   7d. episode_stats_daily...'
SELECT
    COUNT(*) as total_buckets,
    COUNT(DISTINCT experiment_id) as experiments,
    MIN(bucket) as earliest_bucket,
    MAX(bucket) as latest_bucket,
    ROUND(AVG(avg_return)::numeric, 4) as overall_avg_return,
    ROUND(AVG(win_rate)::numeric, 2) as overall_win_rate
FROM episode_stats_daily;

\echo ''
\echo '8. Checking Refresh Policies...'
SELECT
    hypertable_name,
    job_id,
    schedule_interval,
    config->>'start_offset' as start_offset,
    config->>'end_offset' as end_offset
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_refresh_continuous_aggregate'
ORDER BY hypertable_name;

\echo ''
\echo '9. Validating Indexes...'
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
FROM pg_indexes
WHERE schemaname = 'public'
    AND tablename IN ('experiment_metrics', 'metrics_5min', 'metrics_daily',
                      'episode_stats_hourly', 'episode_stats_daily')
ORDER BY tablename, indexname;

\echo ''
\echo '10. Database Size Summary...'
SELECT
    pg_size_pretty(pg_database_size(current_database())) as total_database_size;

SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                   pg_relation_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;

\echo ''
\echo '11. Performance Test: Raw vs Aggregate Query...'
\echo '    (Measuring 5-minute bucket query performance)'

\timing on

\echo ''
\echo '   Query 1: Raw data (slower)...'
SELECT
    time_bucket('5 minutes', time) as bucket,
    AVG(metric_value) as avg_value
FROM experiment_metrics
WHERE metric_name = 'episode_return_pct'
    AND experiment_id = (SELECT MAX(id) FROM experiments)
GROUP BY bucket
ORDER BY bucket DESC
LIMIT 10;

\echo ''
\echo '   Query 2: Continuous aggregate (faster)...'
SELECT
    bucket,
    avg_value
FROM metrics_5min
WHERE metric_name = 'episode_return_pct'
    AND experiment_id = (SELECT MAX(id) FROM experiments)
ORDER BY bucket DESC
LIMIT 10;

\timing off

\echo ''
\echo '========================================='
\echo 'Validation Complete!'
\echo '========================================='
\echo ''
\echo 'Expected Values (from Phase 1 Report):'
\echo '  - Experiments: 10'
\echo '  - Metrics: 12,335'
\echo '  - Model Checkpoints: 59'
\echo ''
\echo 'If values differ significantly, investigate:'
\echo '  1. Check migration logs'
\echo '  2. Verify continuous aggregate refresh'
\echo '  3. Review data retention policies'
\echo '========================================='
