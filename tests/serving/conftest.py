"""Pytest fixtures for serving module tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    from src.serving.config import ModelConfig, ModelSource
    
    return ModelConfig(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        source=ModelSource.HUGGINGFACE,
        revision="main",
        cache_dir="/tmp/models",
    )


@pytest.fixture
def sample_s3_model_config():
    """Sample S3 model configuration for testing."""
    from src.serving.config import ModelConfig, ModelSource
    
    return ModelConfig(
        model_name_or_path="mistral-7b-instruct",
        source=ModelSource.S3,
        s3_bucket="llm-mlops-dev-models",
        s3_prefix="mistral-7b-instruct",
        cache_dir="/tmp/models",
    )


@pytest.fixture
def sample_vllm_config():
    """Sample vLLM configuration for testing."""
    from src.serving.config import VLLMConfig, QuantizationType
    
    return VLLMConfig(
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        quantization=QuantizationType.NONE,
        trust_remote_code=True,
    )


@pytest.fixture
def sample_serving_config(sample_model_config, sample_vllm_config):
    """Sample complete serving configuration."""
    from src.serving.config import ServingConfig, ServerConfig
    
    return ServingConfig(
        model=sample_model_config,
        vllm=sample_vllm_config,
        server=ServerConfig(
            host="0.0.0.0",
            port=8000,
            log_level="INFO",
        ),
    )


@pytest.fixture
def mock_s3_client():
    """Mock boto3 S3 client."""
    with patch("boto3.client") as mock_client:
        s3_mock = MagicMock()
        mock_client.return_value = s3_mock
        
        # Mock paginator
        paginator_mock = MagicMock()
        s3_mock.get_paginator.return_value = paginator_mock
        
        # Mock pagination results
        paginator_mock.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/config.json", "Size": 1024},
                    {"Key": "models/model.safetensors", "Size": 1024 * 1024 * 100},
                ]
            }
        ]
        
        yield s3_mock


@pytest.fixture
def mock_hf_hub():
    """Mock huggingface_hub snapshot_download."""
    with patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = "/tmp/models/test-model"
        yield mock_download


@pytest.fixture
def mock_vllm_engine():
    """Mock vLLM AsyncLLMEngine."""
    with patch("vllm.AsyncLLMEngine") as mock_engine_class:
        engine_mock = MagicMock()
        mock_engine_class.from_engine_args.return_value = engine_mock
        
        # Mock generate method
        async def mock_generate(prompt, sampling_params, request_id):
            result_mock = MagicMock()
            result_mock.outputs = [
                MagicMock(
                    text="This is a test response.",
                    token_ids=[1, 2, 3, 4, 5],
                )
            ]
            result_mock.prompt_token_ids = [10, 11, 12]
            return result_mock
        
        engine_mock.generate = mock_generate
        yield engine_mock


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for KServe operations."""
    with patch("kubernetes.config.load_incluster_config"):
        with patch("kubernetes.client.CustomObjectsApi") as mock_api_class:
            api_mock = MagicMock()
            mock_api_class.return_value = api_mock
            
            # Mock InferenceService responses
            api_mock.get_namespaced_custom_object.return_value = {
                "metadata": {"name": "test-model"},
                "status": {
                    "url": "http://test-model.serving.svc.cluster.local",
                    "conditions": [
                        {"type": "Ready", "status": "True"},
                    ],
                },
            }
            
            yield api_mock


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML config file."""
    config_content = """
model:
  model_name_or_path: mistralai/Mistral-7B-Instruct-v0.2
  source: huggingface
  cache_dir: /tmp/models

vllm:
  tensor_parallel_size: 1
  max_model_len: 4096
  gpu_memory_utilization: 0.90
  dtype: bfloat16

server:
  host: 0.0.0.0
  port: 8000
  log_level: INFO
"""
    config_path = temp_dir / "serving_config.yaml"
    config_path.write_text(config_content)
    return config_path
