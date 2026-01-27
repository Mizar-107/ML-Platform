# Pipeline Components Docker Image
# Lightweight image for Kubeflow pipeline steps

FROM python:3.10-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 pipeline
USER pipeline
WORKDIR /home/pipeline

# Install Python packages
COPY --chown=pipeline:pipeline docker/pipeline-components/requirements.txt ./requirements.txt

RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

# Copy source code
COPY --chown=pipeline:pipeline src/ ./src/
COPY --chown=pipeline:pipeline pipelines/ ./pipelines/
COPY --chown=pipeline:pipeline configs/ ./configs/

# Set Python path
ENV PATH="/home/pipeline/.local/bin:${PATH}" \
    PYTHONPATH="/home/pipeline:${PYTHONPATH}"

# Default entrypoint
ENTRYPOINT ["python"]
