"""Serving configuration classes.

This module provides Pydantic-based configuration models for
vLLM server and model serving settings.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class QuantizationType(str, Enum):
    """Supported quantization methods."""

    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
    FP8 = "fp8"


class ModelSource(str, Enum):
    """Model source types."""

    HUGGINGFACE = "huggingface"
    S3 = "s3"
    LOCAL = "local"


class VLLMConfig(BaseModel):
    """Configuration for vLLM inference engine.

    Attributes:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
        max_model_len: Maximum sequence length (context window)
        max_num_batched_tokens: Maximum tokens per batch
        max_num_seqs: Maximum concurrent sequences
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type for model weights
        quantization: Quantization method
        enforce_eager: Disable CUDA graphs for debugging
        trust_remote_code: Trust remote code in model
        swap_space: CPU swap space in GB
        block_size: KV cache block size
        seed: Random seed
    """

    tensor_parallel_size: int = Field(
        default=1, ge=1, le=8, description="Tensor parallel GPUs"
    )
    pipeline_parallel_size: int = Field(
        default=1, ge=1, le=8, description="Pipeline parallel GPUs"
    )
    max_model_len: int = Field(
        default=4096, ge=256, le=131072, description="Maximum sequence length"
    )
    max_num_batched_tokens: int | None = Field(
        default=None, description="Maximum tokens per batch"
    )
    max_num_seqs: int = Field(
        default=256, ge=1, description="Maximum concurrent sequences"
    )
    gpu_memory_utilization: float = Field(
        default=0.90, ge=0.1, le=0.99, description="GPU memory fraction"
    )
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto", description="Model dtype"
    )
    quantization: QuantizationType = Field(
        default=QuantizationType.NONE, description="Quantization method"
    )
    enforce_eager: bool = Field(
        default=False, description="Disable CUDA graphs"
    )
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code"
    )
    swap_space: int = Field(
        default=4, ge=0, description="CPU swap space (GB)"
    )
    block_size: int = Field(
        default=16, description="KV cache block size"
    )
    seed: int = Field(default=42, description="Random seed")

    def to_engine_args(self) -> dict:
        """Convert to vLLM EngineArgs kwargs.

        Returns:
            Dictionary suitable for vLLM engine initialization
        """
        args = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": self.trust_remote_code,
            "swap_space": self.swap_space,
            "block_size": self.block_size,
            "seed": self.seed,
        }

        if self.max_num_batched_tokens:
            args["max_num_batched_tokens"] = self.max_num_batched_tokens

        if self.quantization != QuantizationType.NONE:
            args["quantization"] = self.quantization.value

        return args


class ModelConfig(BaseModel):
    """Configuration for model loading.

    Attributes:
        model_name_or_path: Model identifier (HuggingFace ID or path)
        source: Model source type (huggingface, s3, local)
        revision: Model revision/version
        s3_bucket: S3 bucket for S3 source
        s3_prefix: S3 prefix for model files
        cache_dir: Local cache directory
        hf_token: HuggingFace token for private models
        lora_adapters: Optional LoRA adapter paths
    """

    model_name_or_path: str = Field(
        ..., description="Model name or path"
    )
    source: ModelSource = Field(
        default=ModelSource.HUGGINGFACE, description="Model source"
    )
    revision: str | None = Field(
        default=None, description="Model revision"
    )
    s3_bucket: str | None = Field(
        default=None, description="S3 bucket for model"
    )
    s3_prefix: str | None = Field(
        default=None, description="S3 prefix for model"
    )
    cache_dir: str = Field(
        default="/mnt/models", description="Local cache directory"
    )
    hf_token: str | None = Field(
        default=None, description="HuggingFace token"
    )
    lora_adapters: list[str] | None = Field(
        default=None, description="LoRA adapter paths"
    )

    @model_validator(mode="after")
    def validate_s3_config(self) -> "ModelConfig":
        """Validate S3 configuration when source is S3."""
        if self.source == ModelSource.S3:
            if not self.s3_bucket:
                raise ValueError("s3_bucket is required when source is 's3'")
        return self

    def get_model_path(self) -> str:
        """Get the resolved model path.

        Returns:
            Model path suitable for loading
        """
        if self.source == ModelSource.LOCAL:
            return self.model_name_or_path
        elif self.source == ModelSource.S3:
            return f"{self.cache_dir}/{self.model_name_or_path}"
        else:
            return self.model_name_or_path

    def get_s3_uri(self) -> str | None:
        """Get S3 URI if source is S3.

        Returns:
            S3 URI or None
        """
        if self.source != ModelSource.S3:
            return None
        prefix = self.s3_prefix or self.model_name_or_path
        return f"s3://{self.s3_bucket}/{prefix}"


class ServerConfig(BaseModel):
    """Configuration for the inference server.

    Attributes:
        host: Server host address
        port: Server port
        api_key: Optional API key for authentication
        cors_origins: Allowed CORS origins
        log_level: Logging level
        metrics_port: Prometheus metrics port
    """

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    api_key: str | None = Field(default=None, description="API key")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS origins"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )
    metrics_port: int = Field(
        default=9090, ge=1, le=65535, description="Metrics port"
    )


class ServingConfig(BaseModel):
    """Complete serving configuration.

    This is the top-level configuration loaded from YAML files.
    """

    model: ModelConfig
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # MLflow model registry integration
    mlflow_tracking_uri: str | None = Field(
        default=None, description="MLflow tracking URI"
    )
    mlflow_model_name: str | None = Field(
        default=None, description="MLflow model name"
    )
    mlflow_model_version: str | None = Field(
        default=None, description="MLflow model version"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "ServingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            ServingConfig instance
        """
        import yaml

        config_path = Path(path)
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)

        return cls.model_validate(config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        import yaml

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Preset configurations for common models
PRESETS = {
    "mistral-7b": ServingConfig(
        model=ModelConfig(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"),
        vllm=VLLMConfig(
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
        ),
    ),
    "llama-7b": ServingConfig(
        model=ModelConfig(model_name_or_path="meta-llama/Llama-2-7b-chat-hf"),
        vllm=VLLMConfig(
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
        ),
    ),
    "llama-13b": ServingConfig(
        model=ModelConfig(model_name_or_path="meta-llama/Llama-2-13b-chat-hf"),
        vllm=VLLMConfig(
            max_model_len=4096,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
        ),
    ),
}


def get_preset(name: str) -> ServingConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (e.g., 'mistral-7b', 'llama-7b')

    Returns:
        ServingConfig instance

    Raises:
        KeyError: If preset not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Preset '{name}' not found. Available: {available}")
    return PRESETS[name].model_copy(deep=True)
