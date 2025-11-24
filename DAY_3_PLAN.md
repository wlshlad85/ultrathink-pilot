# UltraThink Pilot - Day 3 Execution Plan
**Date:** 2025-11-25 (Today)
**Focus:** API Key Authentication & Service Security
**Goal:** Protect all service APIs with authentication
**Status:** 6-Week MVP - Week 1, Day 3 of 5

---

## 📊 Day 2 Recap (Completed Yesterday)

✅ **Completed:**
- All hardcoded passwords removed from codebase
- Docker infrastructure secured with environment variables
- Python services secured (7 files across 4 services)
- Port conflict resolved (risk-manager: 8001→8003)
- Security validation tests (9/10 passing)

**Security Score:** 50% → 75%

---

## 🎯 Day 3 Objectives

### Primary Goal
**Implement API key authentication for all microservices**

### Success Criteria
- [ ] Reusable authentication middleware created
- [ ] All 8 service APIs require valid API keys
- [ ] Service-to-service calls include API keys
- [ ] Rate limiting implemented
- [ ] Authentication tests passing
- [ ] All changes committed and pushed

### Time Budget: 6-8 hours

---

## 📋 Task Breakdown (Prioritized)

### **BLOCK 1: Authentication Foundation (2 hours)**

#### Task 1: Design Authentication Middleware (30 min)

**Create:** `services/common_utils/auth_middleware.py`

**Features:**
- FastAPI dependency for API key validation
- Support for header-based keys: `X-API-Key: uk_...`
- Environment variable-based key storage
- Optional bypass for health checks
- Logging of authentication attempts

**Implementation:**
```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os
import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key from request header.

    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    if api_key is None:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )

    # Get expected key from environment
    expected_key = os.environ.get('SERVICE_API_KEY')
    if not expected_key:
        logger.error("SERVICE_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )

    if api_key != expected_key:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key
```

**Deliverable:** `services/common_utils/auth_middleware.py`

---

#### Task 2: Create Rate Limiting Middleware (45 min)

**Create:** `services/common_utils/rate_limiter.py`

**Features:**
- In-memory rate limiting (Redis optional for production)
- Configurable limits per minute
- Per-API-key or per-IP tracking
- Sliding window algorithm

**Implementation:**
```python
from fastapi import Request, HTTPException, status
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, identifier: str):
        """Check if request is within rate limit."""
        async with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old requests
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if ts > minute_ago
            ]

            # Check limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
                )

            # Record request
            self.requests[identifier].append(now)

async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting."""
    # Get identifier (API key or IP)
    identifier = request.headers.get('X-API-Key', request.client.host)

    # Check rate limit
    await rate_limiter.check_rate_limit(identifier)

    response = await call_next(request)
    return response

# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=100)
```

**Deliverable:** `services/common_utils/rate_limiter.py`

---

#### Task 3: Update .env with Service API Keys (15 min)

Our password generator already created API keys! Verify they're in `.env`:

```bash
# These were generated in Day 2:
DATA_SERVICE_API_KEY=uk_...
INFERENCE_SERVICE_API_KEY=uk_...
RISK_MANAGER_API_KEY=uk_...
REGIME_DETECTION_API_KEY=uk_...
META_CONTROLLER_API_KEY=uk_...
```

**Update docker-compose files** to pass these to services:
```yaml
environment:
  SERVICE_API_KEY: ${DATA_SERVICE_API_KEY}
```

**Deliverable:** Updated docker-compose files with API key environment variables

---

#### Task 4: Create Authentication Tests (30 min)

**Create:** `tests/test_authentication.py`

**Test Cases:**
1. Request without API key → 401
2. Request with invalid API key → 401
3. Request with valid API key → 200
4. Health endpoint bypasses auth → 200
5. Rate limit enforcement → 429

**Deliverable:** Comprehensive authentication test suite

---

### **BLOCK 2: Service Integration (3 hours)**

#### Task 5: Add Auth to data-service (30 min)

**File:** `services/data_service/api.py`

**Changes:**
```python
from common_utils.auth_middleware import verify_api_key

# Add to protected endpoints
@app.get("/api/v1/features", dependencies=[Depends(verify_api_key)])
async def get_features(...):
    ...

# Health check - NO AUTH
@app.get("/health")
async def health():
    ...
```

**Test:**
```bash
# Should fail
curl http://localhost:8000/api/v1/features

# Should succeed
curl -H "X-API-Key: ${DATA_SERVICE_API_KEY}" http://localhost:8000/api/v1/features
```

**Deliverable:** data-service with API key authentication

---

#### Task 6: Add Auth to inference-service (30 min)

**File:** `services/inference_service/inference_api.py`

**Changes:**
```python
from common_utils.auth_middleware import verify_api_key

@app.post("/api/v1/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

**Deliverable:** inference-service with authentication

---

#### Task 7: Add Auth to risk-manager (30 min)

**File:** `services/risk_manager/main.py`

**Changes:**
```python
from common_utils.auth_middleware import verify_api_key

@app.post("/api/v1/risk/check", dependencies=[Depends(verify_api_key)])
async def check_risk(...):
    ...
```

**Deliverable:** risk-manager with authentication

---

#### Task 8: Add Auth to regime-detection (30 min)

**File:** `services/regime_detection/regime_api.py`

**Changes:**
```python
from common_utils.auth_middleware import verify_api_key

@app.post("/regime/probabilities", dependencies=[Depends(verify_api_key)])
async def get_probabilities(...):
    ...
```

**Deliverable:** regime-detection with authentication

---

#### Task 9: Add Auth to meta-controller (30 min)

**File:** `services/meta_controller/api.py`

**Changes:**
```python
from common_utils.auth_middleware import verify_api_key

@app.get("/meta/weights", dependencies=[Depends(verify_api_key)])
async def get_weights(...):
    ...
```

**Deliverable:** meta-controller with authentication

---

#### Task 10: Update Service-to-Service Calls (1 hour)

**Services that make HTTP calls:**
- inference-service → data-service
- inference-service → regime-detection
- inference-service → meta-controller
- inference-service → risk-manager

**Pattern:**
```python
headers = {
    'X-API-Key': os.environ['TARGET_SERVICE_API_KEY'],
    'Content-Type': 'application/json'
}

response = await client.post(
    f"{DATA_SERVICE_URL}/api/v1/features",
    headers=headers,
    json=payload
)
```

**Files to update:**
- `services/inference_service/service_clients.py`
- Any other inter-service communication

**Deliverable:** All service-to-service calls include API keys

---

### **BLOCK 3: Testing & Documentation (2 hours)**

#### Task 11: Integration Testing (1 hour)

**Create:** `tests/test_api_authentication_e2e.py`

**Test Scenarios:**
1. **Unauthenticated Request Flow:**
   - Client → Inference Service (no key) → 401

2. **Authenticated Request Flow:**
   - Client → Inference Service (valid key) → 200
   - Inference → Data Service (with key) → 200
   - Inference → Regime Detection (with key) → 200

3. **Rate Limiting:**
   - Send 150 requests → 429 after 100

4. **Health Checks:**
   - All health endpoints accessible without auth

**Run Tests:**
```bash
pytest tests/test_api_authentication_e2e.py -v
```

**Deliverable:** Comprehensive E2E authentication tests

---

#### Task 12: Update Documentation (30 min)

**Update:** `services/*/README.md`

Add authentication section:
```markdown
## Authentication

All API endpoints require authentication via API key.

### Request Format

```
curl -H "X-API-Key: your-api-key-here" \
     http://localhost:8000/api/v1/endpoint
```

### Getting API Keys

API keys are generated during deployment:
1. Generate: `python3 scripts/generate_secrets.py --output infrastructure/.env`
2. Keys are prefixed with `uk_` (ultrathink key)
3. Keys are 32 characters long

### Error Responses

- `401 Unauthorized`: Missing or invalid API key
- `429 Too Many Requests`: Rate limit exceeded (100 req/min)
```

**Deliverable:** Updated service documentation

---

#### Task 13: Create API Key Management Guide (30 min)

**Create:** `API_KEY_MANAGEMENT.md`

**Content:**
- How to generate API keys
- How to rotate keys
- How to revoke access
- Best practices
- Troubleshooting

**Deliverable:** API key management documentation

---

### **BLOCK 4: Commit & Push (30 min)**

#### Task 14: Final Validation & Commit

**Run all tests:**
```bash
# Security validation
bash tests/security_validation.sh

# Authentication tests
pytest tests/test_authentication.py -v
pytest tests/test_api_authentication_e2e.py -v

# Verify no unauthenticated access
curl http://localhost:8000/api/v1/features
# Should return 401
```

**Commit:**
```bash
git add -A
git commit -m "security: Complete Day 3 - API key authentication

Implemented:
- FastAPI authentication middleware with API key validation
- Rate limiting middleware (100 req/min)
- Authentication on 8 service endpoints
- Service-to-service API key propagation
- Comprehensive authentication tests

Services secured:
- data-service
- inference-service
- risk-manager
- regime-detection
- meta-controller
- online-learning
- forensics-consumer
- training-orchestrator

Security improvements:
- All APIs require X-API-Key header
- Health endpoints exempt from auth
- Rate limiting per API key/IP
- Failed auth attempts logged

Breaking changes:
- All API calls require valid API key
- Service-to-service calls need keys configured

Part of: Week 1 Day 3 - API security
Progress: 75% → 90% security complete"

git push origin claude/analyze-repo-create-todos-01KXRPGZAEL8i5RehdbxcmJa
```

---

## 📊 Day 3 Success Metrics

### Primary Metrics
- [ ] **8 services** protected with API key auth
- [ ] **Rate limiting** implemented and tested
- [ ] **Service-to-service** calls authenticated
- [ ] **Tests passing**: authentication test suite
- [ ] **Documentation** updated with auth examples

### Security Score Target
**Day 2:** 75%
**Day 3 Goal:** 90%
**Remaining:** SQL injection fixes (Day 4), final verification (Day 5)

---

## 🚨 Potential Issues & Solutions

### Issue 1: Import errors (common_utils not in path)
**Solution:** Add `__init__.py` to common_utils, or copy middleware to each service

### Issue 2: API keys not in environment
**Solution:** Verify `.env` loaded, check docker-compose environment section

### Issue 3: Service startup fails with auth
**Solution:** Make health endpoint bypass auth, check logs

### Issue 4: Rate limiting too aggressive
**Solution:** Increase limit or add whitelist for internal services

---

## 📋 Day 3 Checklist

### Morning (3 hours)
- [ ] Task 1: Design auth middleware (30m)
- [ ] Task 2: Create rate limiter (45m)
- [ ] Task 3: Update .env with API keys (15m)
- [ ] Task 4: Create auth tests (30m)
- [ ] Task 5: Add auth to data-service (30m)
- [ ] Task 6: Add auth to inference-service (30m)

### Afternoon (3 hours)
- [ ] Task 7: Add auth to risk-manager (30m)
- [ ] Task 8: Add auth to regime-detection (30m)
- [ ] Task 9: Add auth to meta-controller (30m)
- [ ] Task 10: Update service-to-service calls (1h)
- [ ] Task 11: Integration testing (1h)

### Wrap-Up (1 hour)
- [ ] Task 12: Update documentation (30m)
- [ ] Task 13: Create API key guide (30m)
- [ ] Task 14: Commit and push (30m)

---

## 🔜 Day 4 Preview

**Focus:** SQL Injection Fixes & Input Validation

**Key Tasks:**
1. Audit all SQL queries for injection vulnerabilities
2. Replace f-strings with parameterized queries
3. Add input validation decorators
4. Test with SQL injection payloads
5. Add comprehensive input sanitization

**Estimated Effort:** 6-8 hours

---

**END OF DAY 3 PLAN**

*Authentication is critical for production. Take time to test thoroughly and ensure all services are properly secured.*
