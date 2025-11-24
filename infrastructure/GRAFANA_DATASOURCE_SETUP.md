# Grafana Manual DataSource Setup

**Why Manual Setup?**
Grafana provisioning files don't support environment variable substitution for passwords. To maintain security, the TimescaleDB datasource must be configured manually after Grafana starts.

## Quick Setup (5 minutes)

### 1. Access Grafana
```
URL: http://localhost:3000
Username: admin
Password: (from your .env file - GRAFANA_ADMIN_PASSWORD)
```

### 2. Add TimescaleDB Datasource

**Navigate:** Configuration (⚙️) → Data Sources → Add data source

**Select:** PostgreSQL

**Configuration:**
```
Name: TimescaleDB
Host: timescaledb:5432
Database: ultrathink_experiments
User: ultrathink
Password: (from your .env file - POSTGRES_PASSWORD)
SSL Mode: disable
Version: 15
TimescaleDB: ✓ (enable)
```

**Click:** Save & Test

You should see: ✅ "Database Connection OK"

### 3. Set as Default (Optional)

Click the ⭐ icon to make it the default datasource.

## Automated Setup (Future)

**Options for production:**
1. **Grafana Environment Variables:** Use `GF_DATABASE_PASSWORD__FILE`
2. **Docker Secrets:** Mount password as secret file
3. **Vault Integration:** Dynamic secret injection
4. **API Configuration:** POST to Grafana API at startup

For MVP, manual setup is secure and straightforward.

## Troubleshooting

### "Cannot connect to database"
- Check TimescaleDB is running: `docker ps | grep timescaledb`
- Verify password: `cat infrastructure/.env | grep POSTGRES_PASSWORD`
- Check container network: `docker network ls | grep ultrathink`

### "Invalid password"
- Ensure you're using POSTGRES_PASSWORD from .env
- Password must match what TimescaleDB was started with
- Don't use the old `changeme_in_production`

### "Connection refused"
- TimescaleDB may still be starting: wait 30s
- Check health: `docker logs ultrathink-timescaledb | tail -20`
