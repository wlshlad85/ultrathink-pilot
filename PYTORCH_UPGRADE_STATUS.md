# PyTorch Nightly Upgrade Status

**Date**: 2025-10-24 16:09 UTC
**Task**: Upgrade GPU services from PyTorch 2.1.0 to PyTorch 2.10+ nightly for RTX 5070 (sm_120) support

## Current Status: FINAL REBUILD (Iteration 3) - TARGETING LATEST NIGHTLY WITH sm_120

### First Build Completed (16:12 UTC):
- ✅ PyTorch 2.6.0.dev20241112+cu121 installed successfully
- ❌ Dependencies failed: `|| true` flag caused silent failure
- **Issue**: pip exited early on torch version conflict, never installed numpy/celery

### Dockerfile Fix Applied:
- Changed `pip3 install -r requirements.txt || true`
- To: `grep -v '^torch' requirements.txt > requirements_no_torch.txt && pip3 install -r requirements_no_torch.txt`
- **Effect**: Filters out torch line, installs remaining dependencies

### Build Iteration 2 Result (16:22 UTC):
- ✅ PyTorch 2.6.0.dev20241112+cu121 installed
- ❌ **STILL HAS sm_120 WARNING** - Nov 12 build doesn't support Blackwell!
- **Issue**: RTX 5070 announced Dec 2024, Nov build predates hardware

### Root Cause Analysis:
- PyTorch nightly from Nov 12, 2024 compiled with: sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
- **Missing**: sm_120 (Blackwell / RTX 5070)
- **Solution**: Need LATEST nightly (Oct 24, 2025) with sm_120 support

### Final Rebuild (Iteration 3):
- **meta-controller**: Background task 9ef669 (started 16:30 UTC)
- **training-orchestrator**: Background task c961ce (started 16:30 UTC)
- **Updated flags**: `--pre --upgrade` to force LATEST nightly
- **Expected**: 5-7 minutes, Oct 2025 build should have sm_120

### Changes Made:

**1. Requirements Updated:**
- `services/meta_controller/requirements.txt`: Changed `torch==2.1.0` → `torch>=2.10.0.dev`
- `services/training_orchestrator/requirements.txt`: Changed `torch==2.1.0` → `torch>=2.10.0.dev`

**2. Dockerfiles Updated:**
- Added explicit PyTorch nightly install: `pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/cu121`
- Install torch first, then remaining dependencies

**3. Services Stopped:**
- `docker compose stop meta-controller training-orchestrator` (completed)

**4. Rebuild Started:**
- `docker compose build --no-cache meta-controller` (running)
- `docker compose build --no-cache training-orchestrator` (running)

## Next Steps (When Builds Complete):

### 1. Check Build Status:
```bash
# Check meta-controller
docker logs ultrathink-meta-controller 2>&1 | grep -i "built"

# Check training-orchestrator
docker logs ultrathink-training-orchestrator 2>&1 | grep -i "built"
```

### 2. Deploy Services:
```bash
cd /home/rich/ultrathink-pilot/infrastructure
docker compose up -d meta-controller training-orchestrator
```

### 3. Validate GPU (CRITICAL):
```bash
# Test meta-controller - should show NO sm_120 warning
docker exec ultrathink-meta-controller python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Test training-orchestrator
docker exec ultrathink-training-orchestrator python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected Output:**
- PyTorch: 2.10.0+cu121 (or higher)
- CUDA: True
- GPU: NVIDIA GeForce RTX 5070
- **NO WARNING about sm_120 incompatibility**

### 4. End-to-End Test:
```bash
# Send test message through pipeline
docker exec ultrathink-kafka-1 kafka-console-producer --bootstrap-server localhost:9092 --topic market_data <<EOF
{"symbol": "TEST-USD", "close": 50000, "prev_close": 49000, "volatility": 0.02, "volume_ratio": 1.3, "trend_strength": 0.85, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)"}
EOF

# Check logs
docker logs ultrathink-regime-detection --tail 5
docker logs ultrathink-meta-controller --tail 5
```

### 5. Update Documentation:
- Update `docs/PHASE2_COMPLETION_REPORT.md` with PyTorch version
- Note performance improvements
- Document any issues

## Rollback Plan (If Builds Fail):

### Restore PyTorch 2.1.0:
```bash
# Revert requirements.txt changes
cd /home/rich/ultrathink-pilot
git checkout services/meta_controller/requirements.txt
git checkout services/meta_controller/Dockerfile
git checkout services/training_orchestrator/requirements.txt
git checkout services/training_orchestrator/Dockerfile

# Rebuild
cd infrastructure
docker compose build meta-controller training-orchestrator
docker compose up -d meta-controller training-orchestrator
```

## Files Modified:
- `services/meta_controller/requirements.txt`
- `services/meta_controller/Dockerfile`
- `services/training_orchestrator/requirements.txt`
- `services/training_orchestrator/Dockerfile`

## Background Tasks:
- 46c8c7: meta-controller build
- 038efc: training-orchestrator build

**Check status**: `docker ps -a | grep -E "(meta-controller|training-orchestrator)"`
