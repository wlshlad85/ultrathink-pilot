FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch Nightly with CUDA 12.8 for RTX 5070 Blackwell (sm_120) support
RUN pip3 install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Copy requirements and install remaining dependencies (skip torch as already installed)
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements_no_torch.txt && \
    pip3 install --no-cache-dir -r requirements_no_torch.txt

# Copy application code
COPY meta_controller.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka-1:9092
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Run the meta-controller service
CMD ["python3", "-u", "meta_controller.py"]
