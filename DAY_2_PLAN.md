# UltraThink Pilot - Day 2 Execution Plan
**Date:** 2025-11-25 (Tomorrow)
**Focus:** Complete Password Hardening + Begin API Authentication
**Goal:** Eliminate all remaining hardcoded passwords, test secrets system
**Status:** 6-Week MVP - Week 1, Day 2 of 5

---

## 📊 Day 1 Recap (Completed Today)

✅ **Completed:**
- Security audit (47 vulnerabilities documented)
- Created `.env.template` and `generate_secrets.py`
- Fixed `infrastructure/docker-compose.yml` (all passwords → env vars)
- Fixed port conflict (risk-manager: 8001→8003)
- Committed and pushed to `claude/analyze-repo-create-todos-01KXRPGZAEL8i5RehdbxcmJa`

**Files Modified:** 1
**Files Created:** 3
**Lines Changed:** +937/-18
**Security Score:** 30% → 50%

---

## 🎯 Day 2 Objectives

### Primary Goal
**Eliminate ALL remaining hardcoded passwords from the codebase**

### Success Criteria
- [ ] Zero instances of `changeme_in_production` in active code
- [ ] All Python services read passwords from environment only
- [ ] Password generator tested and documented
- [ ] All changes committed and pushed
- [ ] Services can start with generated .env file

### Time Budget: 6-8 hours

---

## 📋 Task Breakdown (Prioritized)

### **BLOCK 1: Quick Wins (90 minutes)**
*Start here for immediate progress*

#### Task 1.1: Test Password Generator (15 min)
```bash
# Verify the script works
cd /home/user/ultrathink-pilot
python3 scripts/generate_secrets.py

# Generate test .env file
python3 scripts/generate_secrets.py --output /tmp/test.env

# Verify output
cat /tmp/test.env | grep -E "PASSWORD|API_KEY" | wc -l
# Expected: 12+ lines

# Check no placeholders remain
grep "GENERATE" /tmp/test.env
# Expected: empty (all generated)
```

**Deliverable:** Working .env file in `/tmp/test.env`
**Validation:** All passwords are 32+ chars, all API keys have `uk_` prefix

---

#### Task 1.2: Update docker-compose.enhanced.yml (30 min)
**File:** `infrastructure/docker-compose.enhanced.yml`
**Changes Needed:** Same pattern as docker-compose.yml

**Password Locations (6 instances):**
```yaml
Line 9:   POSTGRES_PASSWORD: changeme_in_production
Line 55:  MLFLOW_BACKEND_STORE_URI: postgresql://...changeme_in_production...
Line 62:  --backend-store-uri postgresql://...changeme_in_production...
Line 416: TIMESCALEDB_PASSWORD: changeme_in_production
Line 531: TIMESCALE_PASSWORD: changeme_in_production
Line 623: TIMESCALEDB_PASSWORD: changeme_in_production
```

**Search Command:**
```bash
cd /home/user/ultrathink-pilot/infrastructure
grep -n "changeme_in_production" docker-compose.enhanced.yml
```

**Replace Pattern:** Use same approach as docker-compose.yml:
- `changeme_in_production` → `${POSTGRES_PASSWORD}`
- Add validation: `${POSTGRES_PASSWORD:?must be set in .env}`

**Validation:**
```bash
# After changes
grep "changeme_in_production" docker-compose.enhanced.yml
# Expected: empty
```

**Deliverable:** Updated `docker-compose.enhanced.yml`

---

#### Task 1.3: Fix Grafana Datasource Config (15 min)
**File:** `infrastructure/grafana/provisioning/datasources/datasources.yml`
**Current (Line 25):**
```yaml
password: changeme_in_production
```

**Problem:** Grafana provisioning files don't support env var substitution directly

**Solution Options:**

**Option A: Template approach** (Recommended)
```bash
# Create template file
cp datasources.yml datasources.yml.template

# Use envsubst at container startup
datasources:
  - name: TimescaleDB
    password: ${POSTGRES_PASSWORD}

# Update Grafana entrypoint to process template
```

**Option B: Docker secret** (Simpler for MVP)
```yaml
# Use Grafana's secret file reference
password: $__file{/run/secrets/postgres_password}
```

**Option C: Remove datasource provisioning** (Fastest)
```yaml
# Remove datasource from provisioning
# Add datasource manually via Grafana UI
# Document in README
```

**MVP Recommendation:** Option C (document manual setup)

**Deliverable:** Updated datasource config or documentation

---

#### Task 1.4: Quick Verification (10 min)
```bash
# Scan entire codebase for remaining hardcoded passwords
cd /home/user/ultrathink-pilot
grep -r "changeme_in_production" \
  --exclude-dir=node_modules \
  --exclude-dir=.git \
  --exclude="*.md" \
  --exclude="SECURITY_AUDIT*" \
  --include="*.yml" \
  --include="*.yaml"

# Should only show:
# - .env.example (acceptable - it's a template)
# - Documentation files (acceptable)
```

**Deliverable:** List of remaining instances

---

#### Task 1.5: Commit Block 1 (10 min)
```bash
git add -A
git commit -m "security: Complete infrastructure password hardening

- Test and validate password generation script
- Update docker-compose.enhanced.yml (remove 6 password instances)
- Fix Grafana datasource provisioning
- Verify no hardcoded passwords in configs

Part of: Week 1 Day 2 security hardening"

git push origin claude/analyze-repo-create-todos-01KXRPGZAEL8i5RehdbxcmJa
```

---

### **BLOCK 2: Python Services (3 hours)**
*Core security work - requires careful testing*

#### Task 2.1: Fix forensics_consumer (45 min)

**Files:**
- `services/forensics_consumer/forensics_consumer.py:58`
- `services/forensics_consumer/forensics_api.py:58`

**Current Pattern:**
```python
'password': os.environ.get('TIMESCALEDB_PASSWORD', 'changeme_in_production')
```

**New Pattern:**
```python
'password': os.environ['TIMESCALEDB_PASSWORD']  # Will raise KeyError if not set
```

**Steps:**
1. Read both files
2. Locate all `os.environ.get(..., 'changeme_in_production')` calls
3. Replace with `os.environ['VAR']` or `os.getenv('VAR')` with validation
4. Add startup validation:
   ```python
   # At module level or __main__
   required_vars = ['TIMESCALEDB_PASSWORD', 'KAFKA_BOOTSTRAP_SERVERS']
   missing = [v for v in required_vars if not os.getenv(v)]
   if missing:
       raise RuntimeError(f"Missing required env vars: {missing}")
   ```
5. Test import: `python3 -c "import services.forensics_consumer.forensics_api"`

**Validation:**
```bash
# Check for remaining defaults
grep -n "changeme_in_production" services/forensics_consumer/*.py
# Expected: empty

# Check for proper validation
grep -n "os.environ\[" services/forensics_consumer/*.py
# Expected: 2+ matches
```

**Deliverable:** 2 files updated, tested imports

---

#### Task 2.2: Fix regime_detection (45 min)

**Files:**
- `services/regime_detection/regime_api.py:50`
- `services/regime_detection/Dockerfile:32` (ENV default)

**Current Pattern:**
```python
'password': os.getenv('TIMESCALEDB_PASSWORD', 'changeme_in_production')
```

**Dockerfile Issue:**
```dockerfile
ENV TIMESCALEDB_PASSWORD=changeme_in_production
```

**Steps:**
1. Update `regime_api.py` - same pattern as forensics_consumer
2. **Remove ENV line from Dockerfile** (rely on docker-compose)
3. Add validation at startup
4. Update README.md to document required env vars

**Dockerfile Change:**
```dockerfile
# Before
ENV TIMESCALEDB_PASSWORD=changeme_in_production

# After
# TIMESCALEDB_PASSWORD must be provided via docker-compose environment
```

**Validation:**
```bash
grep "changeme" services/regime_detection/regime_api.py
grep "changeme" services/regime_detection/Dockerfile
# Both should return: empty
```

**Deliverable:** 2 files updated (Python + Dockerfile)

---

#### Task 2.3: Fix meta_controller (45 min)

**Files:**
- `services/meta_controller/meta_controller_v2.py:573`
- `services/meta_controller/api.py:148`

**Current Pattern:**
```python
password: str = 'changeme_in_production'  # Function parameter default
```

**Tricky Part:** This is a function parameter, not env var access

**Solution:**
```python
# Before
def connect_db(password: str = 'changeme_in_production'):
    ...

# After
def connect_db(password: str = None):
    if password is None:
        password = os.environ['POSTGRES_PASSWORD']
    ...
```

**Alternative (Better):**
```python
# Use environment at call site
password = os.environ['POSTGRES_PASSWORD']
connect_db(password=password)
```

**Steps:**
1. Identify all functions with hardcoded password defaults
2. Change defaults to `None`
3. Add env var lookup inside function or at call site
4. Ensure all callers provide password or can access env

**Validation:**
```bash
grep -n "changeme_in_production" services/meta_controller/*.py
# Expected: empty

grep -n "POSTGRES_PASSWORD" services/meta_controller/*.py
# Expected: 2+ matches showing env var usage
```

**Deliverable:** 2 files updated with tested changes

---

#### Task 2.4: Fix inference_service (30 min)

**File:**
- `services/inference_service/ab_storage.py:44`

**Current Pattern:**
```python
self.password = password or os.getenv('TIMESCALE_PASSWORD', 'ultrathink_changeme')
```

**Different Default!** Not `changeme_in_production`

**Steps:**
1. Update to: `os.environ['TIMESCALE_PASSWORD']`
2. Update docker-compose.yml to provide `TIMESCALE_PASSWORD`
3. Add to `.env.template`
4. Validate class instantiation

**Validation:**
```bash
grep -i "changeme" services/inference_service/ab_storage.py
# Expected: empty
```

**Deliverable:** 1 file updated + docker-compose updated

---

#### Task 2.5: Fix risk-manager Port Configuration (15 min)

**Issue:** risk-manager code may still reference port 8001

**Files to Check:**
- `services/risk_manager/*.py` (search for `:8001`)
- `services/risk_manager/README.md`
- Any config files

**Changes:**
```python
# If found
app.run(port=8001)  # Before
app.run(port=8003)  # After

# Better - use env var
app.run(port=int(os.getenv('PORT', 8003)))
```

**Validation:**
```bash
cd services/risk_manager
grep -rn "8001" .
# Check each match to see if it's the service port
```

**Deliverable:** Port references updated to 8003

---

#### Task 2.6: Commit Block 2 (15 min)
```bash
git add services/
git commit -m "security: Remove hardcoded passwords from all Python services

Updated services:
- forensics_consumer (2 files): Remove 'changeme_in_production' defaults
- regime_detection (2 files): Remove Dockerfile ENV + Python defaults
- meta_controller (2 files): Fix function parameter defaults
- inference_service (1 file): Remove 'ultrathink_changeme' default
- risk_manager: Update port references 8001 → 8003

All services now require passwords via environment variables.
Services will fail-fast on startup if passwords not provided.

Breaking change: Services will not start without proper .env file

Part of: Week 1 Day 2 security hardening"

git push origin claude/analyze-repo-create-todos-01KXRPGZAEL8i5RehdbxcmJa
```

---

### **BLOCK 3: Integration Testing (90 min)**
*Verify everything works together*

#### Task 3.1: Generate Production-Style .env (15 min)
```bash
cd /home/user/ultrathink-pilot

# Generate secrets
python3 scripts/generate_secrets.py --output infrastructure/.env

# Verify
cat infrastructure/.env | head -20

# Add OpenAI key manually (if you have one)
echo "OPENAI_API_KEY=sk-your-key-here" >> infrastructure/.env
```

**Deliverable:** `infrastructure/.env` with real passwords

---

#### Task 3.2: Validate Docker Compose Syntax (10 min)
```bash
cd infrastructure

# Validate main compose file
docker-compose config > /dev/null
echo $?  # Should be 0 (success)

# Validate enhanced compose file
docker-compose -f docker-compose.enhanced.yml config > /dev/null
echo $?  # Should be 0
```

**If errors:** Review docker-compose files for syntax issues

**Deliverable:** Valid compose file syntax

---

#### Task 3.3: Test Service Startup (30 min)
```bash
cd infrastructure

# Start only TimescaleDB first (test database password)
docker-compose up -d timescaledb

# Check logs
docker-compose logs timescaledb | tail -20
# Look for: "database system is ready to accept connections"

# Test connection
docker exec ultrathink-timescaledb pg_isready -U ultrathink
# Expected: "accepting connections"

# Start MLflow (test connection string construction)
docker-compose up -d mlflow

# Check MLflow logs
docker-compose logs mlflow | tail -20
# Look for successful connection to TimescaleDB

# Start Grafana (test admin password)
docker-compose up -d grafana

# Check admin password is not default
docker-compose exec grafana grafana-cli admin reset-admin-password --password-from-stdin <<< "test"
# Should fail or require current password

# Stop all
docker-compose down
```

**Expected Results:**
- ✅ TimescaleDB starts successfully
- ✅ MLflow connects to database
- ✅ Grafana requires strong password
- ❌ Should NOT be able to login with admin/admin

**Deliverable:** Verified service startup with secure credentials

---

#### Task 3.4: Security Validation Tests (20 min)

**Create test script:**
```bash
# Create tests/security_validation.sh
cat > tests/security_validation.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Security Validation Tests ==="

# Test 1: No hardcoded passwords in active code
echo "Test 1: Checking for hardcoded passwords..."
FOUND=$(grep -r "changeme_in_production" \
  --exclude-dir=node_modules \
  --exclude-dir=.git \
  --exclude="*.md" \
  --include="*.py" \
  --include="*.yml" \
  --include="*.yaml" \
  . | grep -v "\.env\.template" | grep -v "\.env\.example" || true)

if [ -n "$FOUND" ]; then
  echo "❌ FAIL: Found hardcoded passwords:"
  echo "$FOUND"
  exit 1
else
  echo "✅ PASS: No hardcoded passwords found"
fi

# Test 2: .env file exists
echo "Test 2: Checking .env file..."
if [ ! -f "infrastructure/.env" ]; then
  echo "❌ FAIL: infrastructure/.env not found"
  exit 1
else
  echo "✅ PASS: .env file exists"
fi

# Test 3: All required passwords in .env
echo "Test 3: Checking required passwords..."
REQUIRED_VARS=(
  "POSTGRES_PASSWORD"
  "GRAFANA_ADMIN_PASSWORD"
  "GF_SECURITY_SECRET_KEY"
)

for VAR in "${REQUIRED_VARS[@]}"; do
  if ! grep -q "^${VAR}=" infrastructure/.env; then
    echo "❌ FAIL: Missing $VAR in .env"
    exit 1
  fi
done
echo "✅ PASS: All required passwords present"

# Test 4: Password strength
echo "Test 4: Checking password strength..."
PASSWORD=$(grep "^POSTGRES_PASSWORD=" infrastructure/.env | cut -d= -f2 | tr -d '"')
if [ ${#PASSWORD} -lt 32 ]; then
  echo "❌ FAIL: POSTGRES_PASSWORD too short (${#PASSWORD} chars)"
  exit 1
else
  echo "✅ PASS: Passwords meet 32-char minimum"
fi

echo ""
echo "=== All Tests Passed ==="
EOF

chmod +x tests/security_validation.sh
```

**Run tests:**
```bash
cd /home/user/ultrathink-pilot
./tests/security_validation.sh
```

**Deliverable:** Security validation test suite

---

#### Task 3.5: Document Changes (15 min)

**Update README or create SECURITY_SETUP.md:**
```markdown
# Security Setup Guide

## Quick Start

1. Generate secrets:
   ```bash
   python3 scripts/generate_secrets.py --output infrastructure/.env
   ```

2. Add your OpenAI key:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key" >> infrastructure/.env
   ```

3. Start services:
   ```bash
   cd infrastructure
   docker-compose up -d
   ```

## Security Requirements

- All passwords must be 32+ characters
- Passwords must be stored in `infrastructure/.env` (NOT committed to git)
- Use `generate_secrets.py` to create cryptographically secure passwords
- Rotate passwords every 90 days

## Troubleshooting

### "POSTGRES_PASSWORD must be set in .env file"
Run: `python3 scripts/generate_secrets.py --output infrastructure/.env`

### "Cannot connect to database"
Check: `cat infrastructure/.env | grep POSTGRES_PASSWORD`

### Services fail to start
Validate: `docker-compose config` (in infrastructure/)
```

**Deliverable:** Updated security documentation

---

#### Task 3.6: Commit Block 3 (10 min)
```bash
git add -A
git commit -m "security: Add integration tests and documentation

Added:
- Security validation test suite (tests/security_validation.sh)
- SECURITY_SETUP.md with quick start guide
- Verified service startup with secure credentials

Tested:
- TimescaleDB starts with env var password
- MLflow connects using dynamic connection string
- Grafana requires strong admin password
- All docker-compose files validate successfully

Status: Week 1 Day 2 complete - 70% security hardened"

git push origin claude/analyze-repo-create-todos-01KXRPGZAEL8i5RehdbxcmJa
```

---

## 📊 Day 2 Success Metrics

### Primary Metrics
- [ ] **Zero hardcoded passwords** in active code (excluding docs/examples)
- [ ] **All Python services** read passwords from environment
- [ ] **Services start successfully** with generated .env
- [ ] **Tests pass**: security_validation.sh exits 0

### Secondary Metrics
- [ ] **3-4 commits** pushed to branch
- [ ] **10+ files modified** (Python services + configs)
- [ ] **Documentation updated** (README or SECURITY_SETUP.md)
- [ ] **Port conflict verified fixed** (risk-manager on 8003)

### Security Score Target
**Day 1:** 50%
**Day 2 Goal:** 70-75%
**Remaining:** API authentication (Day 3), SQL injection fixes (Day 4)

---

## 🚨 Blockers & Contingencies

### Potential Issues

**Issue 1: Services won't start without passwords**
- **Impact:** Can't test without .env file
- **Solution:** Generate .env early (Task 3.1)
- **Workaround:** Use docker-compose with `.env.example` temporarily

**Issue 2: Grafana datasource provisioning complex**
- **Impact:** 30-60 min debugging
- **Solution:** Use Option C (manual setup, document it)
- **Escalation:** Skip for MVP, handle in Week 2

**Issue 3: Python import errors after changes**
- **Impact:** Services may not start
- **Solution:** Test imports after each service fix
- **Validation:** `python3 -c "import services.X.Y"`

### Time Contingency
- **Optimistic:** 5 hours (everything works first try)
- **Realistic:** 6-8 hours (expected)
- **Pessimistic:** 10 hours (major debugging needed)

**If running long:**
- Skip docker-compose.enhanced.yml (focus on main file)
- Skip Grafana datasource fix (document manual setup)
- Skip integration testing (defer to Day 3)

---

## 📋 Day 2 Checklist (Print This)

### Morning (3 hours)
- [ ] Task 1.1: Test password generator (15m)
- [ ] Task 1.2: Fix docker-compose.enhanced.yml (30m)
- [ ] Task 1.3: Fix Grafana datasource (15m)
- [ ] Task 1.4: Verify no remaining hardcoded passwords (10m)
- [ ] Task 1.5: Commit Block 1 (10m)
- [ ] BREAK (10m)
- [ ] Task 2.1: Fix forensics_consumer (45m)
- [ ] Task 2.2: Fix regime_detection (45m)

### Afternoon (3 hours)
- [ ] Task 2.3: Fix meta_controller (45m)
- [ ] Task 2.4: Fix inference_service (30m)
- [ ] Task 2.5: Fix risk-manager port (15m)
- [ ] Task 2.6: Commit Block 2 (15m)
- [ ] BREAK (15m)
- [ ] Task 3.1: Generate production .env (15m)
- [ ] Task 3.2: Validate docker-compose (10m)
- [ ] Task 3.3: Test service startup (30m)
- [ ] Task 3.4: Run security validation (20m)
- [ ] Task 3.5: Update documentation (15m)
- [ ] Task 3.6: Commit Block 3 (10m)

### Wrap-Up
- [ ] Review all commits
- [ ] Push to remote
- [ ] Update Day 3 plan (preview)
- [ ] Celebrate progress! 🎉

---

## 🔜 Day 3 Preview (API Authentication)

**Focus:** Implement API key authentication for all services

**Key Tasks:**
1. Design authentication middleware (FastAPI)
2. Generate service API keys
3. Add authentication to 8 service endpoints
4. Update service-to-service HTTP calls
5. Test authenticated requests
6. Add rate limiting

**Estimated Effort:** 8-10 hours

---

## 📞 Support & Escalation

**Questions during Day 2?**
- Check: `SECURITY_AUDIT_2025-11-24.md` for context
- Review: `infrastructure/.env.template` for examples
- Search: Previous commits for patterns

**Blockers?**
- Skip non-critical tasks (Grafana provisioning)
- Focus on Python services (highest impact)
- Document issues for later resolution

**Need Help?**
- Claude Code is available to assist
- Refer to security audit for remediation guidance
- Check Docker logs for debugging

---

**END OF DAY 2 PLAN**

*Remember: Progress > Perfection. Get the critical security fixes done, test them, and document blockers for later.*
