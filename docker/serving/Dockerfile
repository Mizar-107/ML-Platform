# Serving Docker Image
# Optimized for high-throughput inference with vLLM

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Build arguments
ARG PYTHON_VERSION=3.10
ARG VLLM_VERSION=0.3.0

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create non-root user
RUN useradd -m -u 1000 serving
USER serving
WORKDIR /home/serving

# Install vLLM and dependencies
RUN pip install --user --upgrade pip && \
    pip install --user \
    vllm==${VLLM_VERSION} \
    ray \
    prometheus-client \
    structlog

# Copy source code
COPY --chown=serving:serving src/serving/ ./src/serving/
COPY --chown=serving:serving src/common/ ./src/common/

# Set Python path
ENV PATH="/home/serving/.local/bin:${PATH}" \
    PYTHONPATH="/home/serving:${PYTHONPATH}"

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (vLLM server)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
