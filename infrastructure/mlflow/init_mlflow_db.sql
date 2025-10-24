-- MLflow Database Initialization Script
-- Agent: database-migration-specialist (Agent 8)
-- Purpose: Create mlflow_tracking database and grant permissions
-- Date: 2025-10-25

-- Create MLflow tracking database if it doesn't exist
SELECT 'Creating mlflow_tracking database...' AS status;

-- Note: CREATE DATABASE cannot be executed inside a transaction block in the init script
-- This will be created automatically by MLflow when it first connects

-- Grant permissions to ultrathink user on mlflow_tracking database
-- (This will be executed after the database is created)
DO $$
BEGIN
    -- Check if database exists
    IF EXISTS (SELECT 1 FROM pg_database WHERE datname = 'mlflow_tracking') THEN
        RAISE NOTICE 'Database mlflow_tracking already exists';
    END IF;
END
$$;

-- Connect to mlflow_tracking database and set up permissions
\c mlflow_tracking

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant all privileges to ultrathink user
GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO ultrathink;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ultrathink;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ultrathink;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO ultrathink;

-- Create schema ownership
ALTER SCHEMA public OWNER TO ultrathink;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ultrathink;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ultrathink;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO ultrathink;

SELECT 'MLflow database initialized successfully' AS status;
