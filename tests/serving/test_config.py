"""Tests for serving configuration classes."""

import pytest
from pathlib import Path

from src.serving.config import (
    ModelConfig,
    ModelSource,
    QuantizationType,
    ServerConfig,
    ServingConfig,
    VLLMConfig,
    get_preset,
    PRESETS,
)


class TestVLLMConfig:
    """Tests for VLLMConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VLLMConfig()
        
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.90
        assert config.dtype == "auto"
        assert config.quantization == QuantizationType.NONE
        assert config.trust_remote_code is True
        assert config.seed == 42

    def test_to_engine_args(self, sample_vllm_config):
        """Test conversion to vLLM engine args."""
        engine_args = sample_vllm_config.to_engine_args()
        
        assert "tensor_parallel_size" in engine_args
        assert engine_args["max_model_len"] == 4096
        assert engine_args["gpu_memory_utilization"] == 0.90
        assert "quantization" not in engine_args  # NONE quantization excluded

    def test_quantization_in_engine_args(self):
        """Test quantization is included when not NONE."""
        config = VLLMConfig(quantization=QuantizationType.AWQ)
        engine_args = config.to_engine_args()
        
        assert engine_args["quantization"] == "awq"

    def test_max_num_batched_tokens(self):
        """Test max_num_batched_tokens optional field."""
        config = VLLMConfig(max_num_batched_tokens=8192)
        engine_args = config.to_engine_args()
        
        assert engine_args["max_num_batched_tokens"] == 8192

    def test_validation_constraints(self):
        """Test validation constraints."""
        with pytest.raises(ValueError):
            VLLMConfig(tensor_parallel_size=0)  # Must be >= 1
        
        with pytest.raises(ValueError):
            VLLMConfig(gpu_memory_utilization=1.5)  # Must be <= 0.99


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_huggingface_model(self):
        """Test HuggingFace model configuration."""
        config = ModelConfig(
            model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
            source=ModelSource.HUGGINGFACE,
        )
        
        assert config.get_model_path() == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.get_s3_uri() is None

    def test_s3_model(self, sample_s3_model_config):
        """Test S3 model configuration."""
        assert sample_s3_model_config.source == ModelSource.S3
        assert "s3://" in sample_s3_model_config.get_s3_uri()

    def test_local_model(self):
        """Test local model configuration."""
        config = ModelConfig(
            model_name_or_path="/path/to/model",
            source=ModelSource.LOCAL,
        )
        
        assert config.get_model_path() == "/path/to/model"

    def test_s3_validation(self):
        """Test S3 bucket required for S3 source."""
        with pytest.raises(ValueError, match="s3_bucket is required"):
            ModelConfig(
                model_name_or_path="test-model",
                source=ModelSource.S3,
                # Missing s3_bucket
            )

    def test_get_tokenizer_path(self):
        """Test tokenizer defaults to model path."""
        config = ModelConfig(model_name_or_path="test/model")
        assert config.model_name_or_path == "test/model"


class TestServerConfig:
    """Tests for ServerConfig class."""

    def test_default_values(self):
        """Test default server configuration."""
        config = ServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "INFO"
        assert config.metrics_port == 9090
        assert "*" in config.cors_origins

    def test_port_validation(self):
        """Test port validation bounds."""
        with pytest.raises(ValueError):
            ServerConfig(port=0)
        
        with pytest.raises(ValueError):
            ServerConfig(port=70000)


class TestServingConfig:
    """Tests for ServingConfig class."""

    def test_complete_config(self, sample_serving_config):
        """Test complete serving configuration."""
        config = sample_serving_config
        
        assert config.model.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.vllm.max_model_len == 4096
        assert config.server.port == 8000

    def test_from_yaml(self, sample_yaml_config):
        """Test loading from YAML file."""
        config = ServingConfig.from_yaml(str(sample_yaml_config))
        
        assert config.model.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.vllm.max_model_len == 4096
        assert config.server.log_level == "INFO"

    def test_to_yaml(self, sample_serving_config, temp_dir):
        """Test saving to YAML file."""
        output_path = temp_dir / "output_config.yaml"
        sample_serving_config.to_yaml(str(output_path))
        
        assert output_path.exists()
        
        # Reload and verify
        loaded = ServingConfig.from_yaml(str(output_path))
        assert loaded.model.model_name_or_path == sample_serving_config.model.model_name_or_path

    def test_default_subconfigs(self):
        """Test default sub-configurations."""
        config = ServingConfig(
            model=ModelConfig(model_name_or_path="test/model"),
        )
        
        assert config.vllm is not None
        assert config.server is not None


class TestPresets:
    """Tests for configuration presets."""

    def test_available_presets(self):
        """Test available preset configurations."""
        assert "mistral-7b" in PRESETS
        assert "llama-7b" in PRESETS
        assert "llama-13b" in PRESETS

    def test_get_preset(self):
        """Test getting preset configuration."""
        config = get_preset("mistral-7b")
        
        assert config.model.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2"
        assert config.vllm.max_model_len == 8192

    def test_preset_copy(self):
        """Test presets return independent copies."""
        config1 = get_preset("mistral-7b")
        config2 = get_preset("mistral-7b")
        
        config1.vllm.max_model_len = 1024
        
        assert config2.vllm.max_model_len == 8192  # Unchanged

    def test_invalid_preset(self):
        """Test error on invalid preset name."""
        with pytest.raises(KeyError, match="not found"):
            get_preset("nonexistent-model")
