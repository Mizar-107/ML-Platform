# Training Docker Image
# Optimized for LLM fine-tuning with PyTorch and CUDA

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Build arguments
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.2

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create non-root user
RUN useradd -m -u 1000 training
USER training
WORKDIR /home/training

# Install Python packages
COPY --chown=training:training requirements.txt .

RUN pip install --user --upgrade pip && \
    pip install --user \
    torch==${PYTORCH_VERSION} \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --user -r requirements.txt

# Copy source code
COPY --chown=training:training src/ ./src/
COPY --chown=training:training configs/ ./configs/

# Set Python path
ENV PATH="/home/training/.local/bin:${PATH}" \
    PYTHONPATH="/home/training:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "-m", "src.training.trainers.lora_trainer"]
