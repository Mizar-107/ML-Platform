"""Model loading utilities for serving.

This module provides utilities for downloading and loading
models from various sources (S3, HuggingFace Hub, local).
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from src.serving.config import ModelConfig, ModelSource

logger = logging.getLogger(__name__)


def download_model_from_s3(
    bucket: str,
    prefix: str,
    local_path: str,
    aws_region: str | None = None,
) -> Path:
    """Download model files from S3 to local storage.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder) containing model files
        local_path: Local directory to download to
        aws_region: AWS region (uses default if not specified)

    Returns:
        Path to downloaded model directory

    Raises:
        ImportError: If boto3 is not installed
        Exception: If download fails
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError("boto3 is required for S3 downloads: pip install boto3") from e

    logger.info(f"Downloading model from s3://{bucket}/{prefix} to {local_path}")

    # Create S3 client
    config = Config(region_name=aws_region) if aws_region else None
    s3 = boto3.client("s3", config=config)

    # Create local directory
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    # List and download all objects
    paginator = s3.get_paginator("list_objects_v2")
    total_files = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Get relative path from prefix
            rel_path = key[len(prefix):].lstrip("/")
            if not rel_path:
                continue

            local_file = local_dir / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists with same size
            if local_file.exists() and local_file.stat().st_size == obj["Size"]:
                logger.debug(f"Skipping {rel_path} (already exists)")
                continue

            logger.info(f"Downloading {rel_path} ({obj['Size']} bytes)")
            s3.download_file(bucket, key, str(local_file))
            total_files += 1

    logger.info(f"Downloaded {total_files} files to {local_path}")
    return local_dir


def load_model_from_hub(
    model_name: str,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier
        revision: Model revision/branch/tag
        cache_dir: Local cache directory
        token: HuggingFace API token

    Returns:
        Path to downloaded model directory

    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from e

    logger.info(f"Downloading model {model_name} from HuggingFace Hub")

    # Use environment token if not provided
    token = token or os.environ.get("HF_TOKEN")

    # Download model snapshot
    model_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        resume_download=True,
    )

    logger.info(f"Model downloaded to {model_path}")
    return Path(model_path)


class ModelLoader:
    """Model loader with caching and version management.

    Handles downloading and caching models from various sources.
    """

    def __init__(
        self,
        cache_dir: str = "/mnt/models",
        aws_region: str | None = None,
    ):
        """Initialize the model loader.

        Args:
            cache_dir: Local cache directory for models
            aws_region: AWS region for S3 downloads
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.aws_region = aws_region

    def _get_cache_key(self, config: ModelConfig) -> str:
        """Generate cache key for model config.

        Args:
            config: Model configuration

        Returns:
            Cache key string
        """
        # Create hash from model identifier and revision
        key_parts = [
            config.source.value,
            config.model_name_or_path,
            config.revision or "main",
        ]
        if config.source == ModelSource.S3:
            key_parts.extend([config.s3_bucket or "", config.s3_prefix or ""])

        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_cached_path(self, config: ModelConfig) -> Path:
        """Get cached model path if exists.

        Args:
            config: Model configuration

        Returns:
            Path to cached model or None
        """
        cache_key = self._get_cache_key(config)
        cached_path = self.cache_dir / cache_key

        if cached_path.exists() and any(cached_path.iterdir()):
            return cached_path
        return None

    def load(self, config: ModelConfig, force_download: bool = False) -> Path:
        """Load model from configuration.

        Downloads if not cached, uses cache otherwise.

        Args:
            config: Model configuration
            force_download: Force re-download even if cached

        Returns:
            Path to model directory

        Raises:
            ValueError: If source is invalid
        """
        # Check cache first
        if not force_download:
            cached_path = self._get_cached_path(config)
            if cached_path:
                logger.info(f"Using cached model at {cached_path}")
                return cached_path

        # Download based on source
        cache_key = self._get_cache_key(config)
        target_path = self.cache_dir / cache_key

        # Clean existing cache if force download
        if force_download and target_path.exists():
            shutil.rmtree(target_path)

        target_path.mkdir(parents=True, exist_ok=True)

        if config.source == ModelSource.HUGGINGFACE:
            model_path = load_model_from_hub(
                model_name=config.model_name_or_path,
                revision=config.revision,
                cache_dir=str(self.cache_dir / "hub"),
                token=config.hf_token,
            )
            # Symlink to cache path
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.symlink_to(model_path)

        elif config.source == ModelSource.S3:
            download_model_from_s3(
                bucket=config.s3_bucket,
                prefix=config.s3_prefix or config.model_name_or_path,
                local_path=str(target_path),
                aws_region=self.aws_region,
            )

        elif config.source == ModelSource.LOCAL:
            local_path = Path(config.model_name_or_path)
            if not local_path.exists():
                raise ValueError(f"Local model path does not exist: {local_path}")
            return local_path

        else:
            raise ValueError(f"Unsupported model source: {config.source}")

        logger.info(f"Model loaded to {target_path}")
        return target_path

    def clear_cache(self, config: ModelConfig | None = None) -> None:
        """Clear model cache.

        Args:
            config: Specific model config to clear, or None for all
        """
        if config:
            cache_key = self._get_cache_key(config)
            cached_path = self.cache_dir / cache_key
            if cached_path.exists():
                shutil.rmtree(cached_path)
                logger.info(f"Cleared cache for {config.model_name_or_path}")
        else:
            for path in self.cache_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
            logger.info("Cleared all model cache")

    def list_cached(self) -> list[str]:
        """List all cached models.

        Returns:
            List of cache keys
        """
        return [p.name for p in self.cache_dir.iterdir() if p.is_dir()]


def merge_lora_adapters(
    base_model_path: str,
    adapter_paths: list[str],
    output_path: str,
    adapter_weights: list[float] | None = None,
) -> Path:
    """Merge LoRA adapters into base model.

    Args:
        base_model_path: Path to base model
        adapter_paths: Paths to LoRA adapters
        output_path: Output path for merged model
        adapter_weights: Optional weights for each adapter

    Returns:
        Path to merged model

    Raises:
        ImportError: If PEFT is not installed
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "peft and transformers are required: pip install peft transformers"
        ) from e

    logger.info(f"Merging {len(adapter_paths)} LoRA adapters into {base_model_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load and merge adapters
    for i, adapter_path in enumerate(adapter_paths):
        logger.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

        # Merge into base model
        model = model.merge_and_unload()

    # Save merged model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
