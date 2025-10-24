# Docker Setup for UltraThink Pilot

This guide explains how to run the UltraThink Pilot trading system using Docker.

## Quick Start

### 1. Build the Docker Image

**CPU Version (default):**
```bash
docker build -t ultrathink-pilot:cpu .
```

**GPU Version (requires NVIDIA Docker runtime):**
```bash
docker build --build-arg VARIANT=gpu -t ultrathink-pilot:gpu .
```

### 2. Run with Docker Compose

**Run tests (default):**
```bash
docker-compose up ultrathink-cpu
```

**Run backtesting:**
```bash
docker-compose --profile backtest up backtest
```

**Run RL training (CPU):**
```bash
docker-compose --profile rl-train up rl-train-cpu
```

**Run RL training (GPU):**
```bash
docker-compose --profile rl-train-gpu up rl-train-gpu
```

**Development shell:**
```bash
docker-compose --profile dev run --rm dev
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=4000

# Backtesting Parameters
SYMBOL=BTC-USD
START_DATE=2023-01-01
END_DATE=2024-01-01
INITIAL_CAPITAL=100000
COMMISSION=0.001

# RL Training Parameters
EPISODES=100
```

## Docker Commands

### Building

```bash
# Build CPU version
docker build -t ultrathink-pilot:cpu .

# Build GPU version
docker build --build-arg VARIANT=gpu -t ultrathink-pilot:gpu .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t ultrathink-pilot:cpu .
```

### Running

**Run tests:**
```bash
docker run --rm ultrathink-pilot:cpu pytest tests/ -v
```

**Run backtest:**
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=sk-your-key \
  ultrathink-pilot:cpu \
  python run_backtest.py --symbol BTC-USD --start 2023-01-01 --end 2024-01-01
```

**Run RL training:**
```bash
docker run --rm \
  -v $(pwd)/rl/models:/app/rl/models \
  -v $(pwd)/ml_experiments.db:/app/ml_experiments.db \
  ultrathink-pilot:cpu \
  python rl/train.py --episodes 100 --symbol BTC-USD
```

**Interactive shell:**
```bash
docker run -it --rm \
  -v $(pwd):/app \
  ultrathink-pilot:cpu \
  /bin/bash
```

## GPU Support

To use GPU acceleration for RL training:

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Docker runtime** installed:
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Run with GPU

```bash
# Using docker run
docker run --rm --gpus all \
  -v $(pwd)/rl/models:/app/rl/models \
  ultrathink-pilot:gpu \
  python rl/train.py --episodes 100

# Using docker-compose
docker-compose --profile rl-train-gpu up rl-train-gpu
```

## Docker Compose Profiles

The `docker-compose.yml` file defines several profiles for different use cases:

| Profile | Service | Description |
|---------|---------|-------------|
| `default` | ultrathink-cpu | Run tests (CPU) |
| `cpu` | ultrathink-cpu | Run tests (CPU) |
| `gpu` | ultrathink-gpu | Run tests (GPU) |
| `backtest` | backtest | Run backtesting |
| `rl-train` | rl-train-cpu | Train RL agent (CPU) |
| `rl-train-gpu` | rl-train-gpu | Train RL agent (GPU) |
| `dev` | dev | Development shell |

## Volume Mounts

The Docker setup uses the following volume mounts for data persistence:

- `./data:/app/data` - Market data cache
- `./rl/models:/app/rl/models` - Trained RL model checkpoints
- `./logs:/app/logs` - Application logs
- `./ml_experiments.db:/app/ml_experiments.db` - ML experiment tracking database

## Customization

### Using Custom Requirements

Edit `requirements.txt` and rebuild:
```bash
docker build -t ultrathink-pilot:cpu .
```

### Using Different Base Image

Edit the `Dockerfile` and change the `PYTHON_VERSION` argument:
```dockerfile
ARG PYTHON_VERSION=3.12  # Change to 3.12
```

Then rebuild:
```bash
docker build --build-arg PYTHON_VERSION=3.12 -t ultrathink-pilot:cpu .
```

## Troubleshooting

### Image is too large

The image includes PyTorch which is large (~2GB). To reduce size:
- Use CPU version instead of GPU
- Use multi-stage builds (already implemented)
- Remove unnecessary dependencies from `requirements.txt`

### Permission errors

The container runs as non-root user `trader` (UID 1000). Ensure mounted volumes have correct permissions:
```bash
sudo chown -R 1000:1000 data/ rl/models/ logs/
```

### GPU not detected

Verify NVIDIA Docker runtime:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Build fails due to missing dependencies

Ensure you have sufficient disk space and internet connection. The build downloads ~2GB of dependencies.

## Examples

### Full backtesting workflow

```bash
# 1. Build image
docker build -t ultrathink-pilot:cpu .

# 2. Set environment variables
export OPENAI_API_KEY=sk-your-key
export SYMBOL=BTC-USD
export START_DATE=2023-01-01
export END_DATE=2024-01-01

# 3. Run backtest
docker-compose --profile backtest up backtest

# 4. Check results
ls -lh data/
```

### Full RL training workflow (GPU)

```bash
# 1. Build GPU image
docker build --build-arg VARIANT=gpu -t ultrathink-pilot:gpu .

# 2. Set environment variables
export EPISODES=200
export SYMBOL=ETH-USD
export START_DATE=2023-01-01
export END_DATE=2024-01-01

# 3. Run training
docker-compose --profile rl-train-gpu up rl-train-gpu

# 4. Check trained models
ls -lh rl/models/
```

## Best Practices

1. **Use `.env` file** for configuration instead of hardcoding values
2. **Mount volumes** for data persistence (models, logs, database)
3. **Use GPU version** for RL training if available (10-50x faster)
4. **Run tests** after building to verify installation
5. **Use profiles** in docker-compose for different workflows
6. **Clean up** old images regularly: `docker image prune -a`

## Next Steps

- See `CLAUDE.md` for detailed project documentation
- See `ML_PERSISTENCE_INTRODUCTION.md` for ML experiment tracking
- See `README.md` for project overview
- Run `docker-compose --profile dev run --rm dev` for interactive development
