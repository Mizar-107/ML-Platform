"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False

    # AWS
    aws_region: str = "us-west-2"
    aws_profile: str | None = None

    # S3 Buckets
    data_bucket: str = "llm-mlops-data-dev"
    models_bucket: str = "llm-mlops-models-dev"
    artifacts_bucket: str = "llm-mlops-artifacts-dev"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "default"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Ray
    ray_address: str = "ray://localhost:10001"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None

    # Model Defaults
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Hugging Face
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
