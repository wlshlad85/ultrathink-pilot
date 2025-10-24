# Grafana Quickstart Guide - UltraThink Pilot

**Grafana UI:** http://localhost:3000

---

## First-Time Setup

### 1. Initial Login

1. Open browser and navigate to: **http://localhost:3000**
2. Default credentials:
   - **Username:** `admin`
   - **Password:** `admin`
3. You'll be prompted to set a new password on first login

### 2. Add Data Sources

#### Add Prometheus

1. Click the gear icon (⚙️) → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Configure:
   - **Name:** Prometheus
   - **URL:** `http://prometheus:9090`
   - Leave other settings as default
5. Click **Save & Test** (should see green checkmark)

#### Add TimescaleDB (PostgreSQL)

1. Click **Add data source** again
2. Select **PostgreSQL**
3. Configure:
   - **Name:** TimescaleDB
   - **Host:** `timescaledb:5432`
   - **Database:** `ultrathink_experiments`
   - **User:** `ultrathink`
   - **Password:** [check `infrastructure/.env` file]
   - **SSL Mode:** `disable`
   - **Version:** `12+`
   - **TimescaleDB:** ✅ Enable
   - **Min time interval:** `1s`
4. Click **Save & Test** (should see green checkmark)

---

## Create Your First Dashboard

### Training Performance Dashboard

1. Click **+** → **Dashboard** → **Add new panel**
2. Select **TimescaleDB** as data source
3. Click **Code** (SQL editor)

#### Panel 1: Episode Returns Over Time

```sql
SELECT
    timestamp as time,
    value as "Return %"
FROM experiment_metrics
WHERE
    metric_name = 'episode_return_pct'
    AND experiment_id = $experiment_id
    AND $__timeFilter(timestamp)
ORDER BY timestamp
```

**Variables to add:**
- Variable name: `experiment_id`
- Type: `Query`
- Data source: `TimescaleDB`
- Query: `SELECT DISTINCT experiment_id FROM experiment_metrics ORDER BY experiment_id DESC`

#### Panel 2: Average Return (Rolling Window)

```sql
SELECT
    timestamp as time,
    AVG(value) OVER (ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as "Avg Return (10 ep)"
FROM experiment_metrics
WHERE
    metric_name = 'episode_return_pct'
    AND experiment_id = $experiment_id
    AND $__timeFilter(timestamp)
ORDER BY timestamp
```

#### Panel 3: Episode Length Distribution

```sql
SELECT
    timestamp as time,
    value as "Steps"
FROM experiment_metrics
WHERE
    metric_name = 'episode_length'
    AND experiment_id = $experiment_id
    AND $__timeFilter(timestamp)
ORDER BY timestamp
```

#### Panel 4: Win Rate

```sql
SELECT
    time_bucket('10 minutes', timestamp) as time,
    (SUM(CASE WHEN value > 0 THEN 1 ELSE 0 END)::float / COUNT(*) * 100) as "Win Rate %"
FROM experiment_metrics
WHERE
    metric_name = 'episode_return_pct'
    AND experiment_id = $experiment_id
    AND $__timeFilter(timestamp)
GROUP BY time
ORDER BY time
```

---

## Pre-Built Query Examples

### View Experiment Performance

**Query:**
```sql
SELECT
    e.id,
    e.experiment_name,
    e.status,
    COUNT(DISTINCT m.episode) as episodes_completed,
    AVG(CASE WHEN m.metric_name = 'episode_return_pct' THEN m.value END) as avg_return,
    MAX(CASE WHEN m.metric_name = 'episode_return_pct' THEN m.value END) as best_return
FROM experiments e
LEFT JOIN experiment_metrics m ON e.id = m.experiment_id
WHERE e.id = 14
GROUP BY e.id, e.experiment_name, e.status
```

### Model Checkpoints by Performance

**Query:**
```sql
SELECT
    created_at as time,
    episode_num as "Episode",
    val_metric as "Validation Metric",
    CASE WHEN is_best THEN 'Best' ELSE 'Regular' END as "Type"
FROM model_checkpoints
WHERE experiment_id = $experiment_id
ORDER BY created_at
```

### Hourly Training Throughput

**Query:**
```sql
SELECT
    time_bucket('1 hour', timestamp) as time,
    COUNT(DISTINCT episode) as "Episodes per Hour"
FROM experiment_metrics
WHERE
    experiment_id = $experiment_id
    AND metric_name = 'episode_return_pct'
    AND $__timeFilter(timestamp)
GROUP BY time
ORDER BY time
```

---

## System Monitoring Dashboard (Prometheus)

### Panel 1: Container CPU Usage

1. Data source: **Prometheus**
2. Query:
```promql
rate(container_cpu_usage_seconds_total{container_label_com_docker_compose_project="infrastructure"}[5m]) * 100
```

### Panel 2: Memory Usage

```promql
container_memory_usage_bytes{container_label_com_docker_compose_project="infrastructure"} / 1024 / 1024
```

### Panel 3: Network I/O

```promql
rate(container_network_receive_bytes_total{container_label_com_docker_compose_project="infrastructure"}[5m])
```

---

## Import Dashboard from JSON

You can create a dashboard JSON file and import it:

1. Click **+** → **Import**
2. Paste JSON or upload file
3. Select data source
4. Click **Import**

**Example Dashboard JSON Structure:**
```json
{
  "dashboard": {
    "title": "UltraThink Training Metrics",
    "panels": [
      {
        "title": "Episode Returns",
        "type": "timeseries",
        "datasource": "TimescaleDB",
        "targets": [
          {
            "rawSql": "SELECT timestamp as time, value FROM experiment_metrics WHERE metric_name = 'episode_return_pct' AND experiment_id = 14"
          }
        ]
      }
    ]
  }
}
```

---

## Useful Grafana Features

### Variables

Create dashboard variables for dynamic filtering:

1. Dashboard settings (gear icon) → **Variables** → **Add variable**
2. Common variables:
   - `$experiment_id` - Filter by experiment
   - `$metric_name` - Select metric to display
   - `$time_range` - Custom time range

### Annotations

Mark important events on your dashboards:

1. Dashboard settings → **Annotations** → **Add annotation query**
2. Example: Mark when models are saved
```sql
SELECT
    created_at as time,
    'Model ' || id || ' saved (ep ' || episode_num || ')' as text,
    'checkpoint' as tags
FROM model_checkpoints
WHERE experiment_id = $experiment_id
```

### Alerts

Set up alerts for training issues:

1. Edit panel → **Alert** tab → **Create alert**
2. Example: Alert if win rate drops below 40%
3. Configure notification channels (email, Slack, etc.)

---

## Common SQL Patterns for TimescaleDB

### Time-series aggregation

```sql
SELECT
    time_bucket('5 minutes', timestamp) as time,
    metric_name,
    AVG(value) as avg_value,
    STDDEV(value) as std_value
FROM experiment_metrics
WHERE experiment_id = $experiment_id
  AND $__timeFilter(timestamp)
GROUP BY time, metric_name
ORDER BY time
```

### Latest value per metric

```sql
SELECT DISTINCT ON (metric_name)
    metric_name,
    value,
    timestamp
FROM experiment_metrics
WHERE experiment_id = $experiment_id
ORDER BY metric_name, timestamp DESC
```

### Percentage change over time

```sql
SELECT
    timestamp as time,
    metric_name,
    value,
    (value - LAG(value) OVER (PARTITION BY metric_name ORDER BY timestamp)) / LAG(value) OVER (PARTITION BY metric_name ORDER BY timestamp) * 100 as pct_change
FROM experiment_metrics
WHERE experiment_id = $experiment_id
  AND $__timeFilter(timestamp)
ORDER BY timestamp
```

---

## Troubleshooting

### Data source connection failed

1. Check Docker containers are running:
   ```bash
   docker ps
   ```

2. Verify TimescaleDB is accessible:
   ```bash
   docker exec ultrathink-timescaledb psql -U ultrathink -d ultrathink_experiments -c "SELECT 1;"
   ```

3. Check Prometheus is running:
   ```bash
   curl http://localhost:9090/-/healthy
   ```

### No data in panels

1. Verify experiment ID exists:
   ```sql
   SELECT id, experiment_name FROM experiments ORDER BY id DESC LIMIT 5;
   ```

2. Check metric data exists:
   ```sql
   SELECT COUNT(*) FROM experiment_metrics WHERE experiment_id = 14;
   ```

3. Ensure time range includes data timestamps

### Query timeout

1. Add indexes if querying large datasets
2. Use time_bucket() for aggregations
3. Limit results with WHERE clauses
4. Use continuous aggregates for pre-computed metrics

---

## Next Steps

1. **Create custom dashboards** for your specific metrics
2. **Set up alerts** for training anomalies
3. **Export dashboards** as JSON for version control
4. **Share dashboards** with team members

---

## Additional Resources

- **Grafana Docs:** https://grafana.com/docs/grafana/latest/
- **TimescaleDB Docs:** https://docs.timescale.com/
- **Prometheus Docs:** https://prometheus.io/docs/

---

**Need help?** Check the main documentation or create an issue in the project repository.

**Dashboard Examples:** See `infrastructure/grafana_dashboards/` directory (to be created in Phase 2)
