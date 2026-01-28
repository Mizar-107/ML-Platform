"""Tests for model loader module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.serving.config import ModelConfig, ModelSource
from src.serving.model_loader import (
    ModelLoader,
    download_model_from_s3,
    load_model_from_hub,
)


class TestDownloadModelFromS3:
    """Tests for S3 download function."""

    def test_download_basic(self, mock_s3_client, temp_dir):
        """Test basic S3 download."""
        result = download_model_from_s3(
            bucket="test-bucket",
            prefix="models/test-model",
            local_path=str(temp_dir / "model"),
        )
        
        assert result.exists()
        mock_s3_client.get_paginator.assert_called_once_with("list_objects_v2")

    def test_skip_existing_files(self, mock_s3_client, temp_dir):
        """Test skipping already downloaded files."""
        local_dir = temp_dir / "model"
        local_dir.mkdir(parents=True)
        
        # Create existing file
        config_file = local_dir / "config.json"
        config_file.write_text("{}")
        
        # Download should skip existing
        download_model_from_s3(
            bucket="test-bucket",
            prefix="models/test-model",
            local_path=str(local_dir),
        )
        
        # Original file should still exist
        assert config_file.exists()

    def test_import_error(self):
        """Test ImportError when boto3 not installed."""
        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3"):
                # Force reimport
                import importlib
                import src.serving.model_loader as ml
                importlib.reload(ml)


class TestLoadModelFromHub:
    """Tests for HuggingFace Hub download function."""

    def test_load_from_hub(self, mock_hf_hub):
        """Test loading from HuggingFace Hub."""
        result = load_model_from_hub(
            model_name="test/model",
            revision="main",
        )
        
        assert result == Path("/tmp/models/test-model")
        mock_hf_hub.assert_called_once()

    def test_load_with_token(self, mock_hf_hub):
        """Test loading with HF token."""
        load_model_from_hub(
            model_name="test/model",
            token="test-token",
        )
        
        call_kwargs = mock_hf_hub.call_args.kwargs
        assert call_kwargs["token"] == "test-token"

    def test_load_with_cache_dir(self, mock_hf_hub, temp_dir):
        """Test loading with custom cache directory."""
        load_model_from_hub(
            model_name="test/model",
            cache_dir=str(temp_dir),
        )
        
        call_kwargs = mock_hf_hub.call_args.kwargs
        assert str(temp_dir) in call_kwargs["cache_dir"]


class TestModelLoader:
    """Tests for ModelLoader class."""

    def test_loader_initialization(self, temp_dir):
        """Test loader initialization creates cache directory."""
        cache_dir = temp_dir / "model_cache"
        loader = ModelLoader(cache_dir=str(cache_dir))
        
        assert cache_dir.exists()
        assert loader.cache_dir == cache_dir

    def test_cache_key_generation(self, temp_dir, sample_model_config):
        """Test cache key generation."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        key1 = loader._get_cache_key(sample_model_config)
        key2 = loader._get_cache_key(sample_model_config)
        
        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 chars

    def test_cache_key_different_configs(self, temp_dir):
        """Test different configs produce different keys."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        config1 = ModelConfig(model_name_or_path="model/a")
        config2 = ModelConfig(model_name_or_path="model/b")
        
        key1 = loader._get_cache_key(config1)
        key2 = loader._get_cache_key(config2)
        
        assert key1 != key2

    def test_load_from_huggingface(self, temp_dir, mock_hf_hub, sample_model_config):
        """Test loading HuggingFace model."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create mock downloaded path
        mock_path = temp_dir / "hub" / "test-model"
        mock_path.mkdir(parents=True)
        (mock_path / "config.json").write_text("{}")
        mock_hf_hub.return_value = str(mock_path)
        
        result = loader.load(sample_model_config)
        
        assert result.exists()

    def test_load_local_model(self, temp_dir):
        """Test loading local model."""
        # Create local model directory
        local_model = temp_dir / "local_model"
        local_model.mkdir()
        (local_model / "config.json").write_text("{}")
        
        config = ModelConfig(
            model_name_or_path=str(local_model),
            source=ModelSource.LOCAL,
        )
        
        loader = ModelLoader(cache_dir=str(temp_dir))
        result = loader.load(config)
        
        assert result == local_model

    def test_load_local_not_found(self, temp_dir):
        """Test error when local model not found."""
        config = ModelConfig(
            model_name_or_path="/nonexistent/path",
            source=ModelSource.LOCAL,
        )
        
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        with pytest.raises(ValueError, match="does not exist"):
            loader.load(config)

    def test_cache_hit(self, temp_dir, sample_model_config, mock_hf_hub):
        """Test cache hit returns cached path."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create cached model
        cache_key = loader._get_cache_key(sample_model_config)
        cached_path = temp_dir / cache_key
        cached_path.mkdir()
        (cached_path / "config.json").write_text("{}")
        
        result = loader.load(sample_model_config)
        
        assert result == cached_path
        mock_hf_hub.assert_not_called()  # Should not download

    def test_force_download(self, temp_dir, sample_model_config, mock_hf_hub):
        """Test force download bypasses cache."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create cached model
        cache_key = loader._get_cache_key(sample_model_config)
        cached_path = temp_dir / cache_key
        cached_path.mkdir()
        (cached_path / "old_file.txt").write_text("old")
        
        # Create mock downloaded path
        mock_path = temp_dir / "hub" / "new-model"
        mock_path.mkdir(parents=True)
        (mock_path / "config.json").write_text("{}")
        mock_hf_hub.return_value = str(mock_path)
        
        loader.load(sample_model_config, force_download=True)
        
        mock_hf_hub.assert_called_once()  # Should download

    def test_clear_cache_specific(self, temp_dir, sample_model_config):
        """Test clearing specific model cache."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create cached model
        cache_key = loader._get_cache_key(sample_model_config)
        cached_path = temp_dir / cache_key
        cached_path.mkdir()
        (cached_path / "config.json").write_text("{}")
        
        loader.clear_cache(sample_model_config)
        
        assert not cached_path.exists()

    def test_clear_cache_all(self, temp_dir):
        """Test clearing all cache."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create multiple cached models
        (temp_dir / "model1").mkdir()
        (temp_dir / "model2").mkdir()
        
        loader.clear_cache()
        
        assert len(list(temp_dir.iterdir())) == 0

    def test_list_cached(self, temp_dir):
        """Test listing cached models."""
        loader = ModelLoader(cache_dir=str(temp_dir))
        
        # Create cached models
        (temp_dir / "model1").mkdir()
        (temp_dir / "model2").mkdir()
        
        cached = loader.list_cached()
        
        assert len(cached) == 2
        assert "model1" in cached
        assert "model2" in cached


class TestMergeLoraAdapters:
    """Tests for LoRA adapter merging."""

    @pytest.mark.skip(reason="Requires PEFT and transformers")
    def test_merge_adapters(self, temp_dir):
        """Test merging LoRA adapters."""
        from src.serving.model_loader import merge_lora_adapters
        
        # This would require actual model files
        pass
