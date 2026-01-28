# LiteLLM Docker Image
# Custom image with additional configurations

FROM ghcr.io/berriai/litellm:main-v1.56.3

# Labels
LABEL maintainer="MLOps Team"
LABEL org.opencontainers.image.source="https://github.com/YOUR_ORG/llm-mlops-platform"
LABEL org.opencontainers.image.description="LiteLLM proxy gateway for LLM serving"

# Environment
ENV LITELLM_MODE=PRODUCTION \
    LITELLM_LOG=INFO

# Create config directory
USER root
RUN mkdir -p /app/config && chown -R 1000:1000 /app/config
USER 1000

# Copy default configuration (can be overridden via ConfigMap)
COPY --chown=1000:1000 config.yaml /app/config/config.yaml

# Expose ports
EXPOSE 4000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1

# Default entrypoint
ENTRYPOINT ["litellm"]
CMD ["--config", "/app/config/config.yaml", "--port", "4000"]
