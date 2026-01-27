"""Unit tests for DataPipeline orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.pipeline.pipeline import (
    ChunkStrategy,
    DataPipeline,
    DataPipelineConfig,
    EmbeddingModelType,
    PipelineMetrics,
    create_pipeline,
)


class TestDataPipelineConfig:
    """Tests for DataPipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataPipelineConfig(source_path="s3://bucket/docs")

        assert config.chunk_strategy == ChunkStrategy.SEMANTIC
        assert config.chunk_size == 512
        assert config.embedding_model_type == EmbeddingModelType.SENTENCE_TRANSFORMER
        assert config.milvus_host == "localhost"
        assert config.dry_run is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataPipelineConfig(
            source_path="/local/path",
            chunk_strategy=ChunkStrategy.FIXED,
            chunk_size=256,
            embedding_model_name="BAAI/bge-small-en-v1.5",
            dry_run=True,
        )

        assert config.source_path == "/local/path"
        assert config.chunk_strategy == ChunkStrategy.FIXED
        assert config.chunk_size == 256
        assert config.dry_run is True

    def test_config_from_dict(self, sample_config):
        """Test creating config from dict."""
        config = DataPipelineConfig(**sample_config)
        assert config.source_path == sample_config["source_path"]


class TestDataPipeline:
    """Tests for DataPipeline class."""

    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initializes correctly."""
        config = DataPipelineConfig(**sample_config)
        pipeline = DataPipeline(config)

        assert pipeline.config == config
        assert pipeline.metrics.documents_loaded == 0
        assert pipeline._loader is None  # Lazy init

    def test_create_pipeline_from_config(self, sample_config):
        """Test create_pipeline factory function."""
        pipeline = create_pipeline(sample_config)

        assert isinstance(pipeline, DataPipeline)
        assert pipeline.config.source_path == sample_config["source_path"]

    @patch("src.data.pipeline.pipeline.get_loader")
    def test_ingest_loads_documents(self, mock_get_loader, sample_config):
        """Test ingestion loads documents."""
        from src.data.ingestion.loaders import Document

        # Mock loader
        mock_loader = MagicMock()
        mock_loader.load.return_value = iter([
            Document(
                content="Test content",
                source="test.txt",
                doc_type=".txt",
                metadata={},
            )
        ])
        mock_get_loader.return_value = mock_loader

        config = DataPipelineConfig(**sample_config)
        pipeline = DataPipeline(config)

        with patch("src.data.pipeline.pipeline.get_parser") as mock_get_parser:
            mock_parser = MagicMock()
            mock_parsed = MagicMock()
            mock_parsed.elements = [MagicMock(text="Test content")]
            mock_parsed.metadata = {}
            mock_parser.parse.return_value = mock_parsed
            mock_get_parser.return_value = mock_parser

            docs = list(pipeline.ingest())

        assert len(docs) == 1
        assert pipeline.metrics.documents_loaded == 1
        assert pipeline.metrics.documents_parsed == 1

    def test_chunk_creates_chunks(self, sample_config):
        """Test chunking creates chunks."""
        from src.data.ingestion.loaders import Document

        config = DataPipelineConfig(**sample_config)
        pipeline = DataPipeline(config)

        doc = Document(
            content="This is a test document. " * 20,
            source="test.txt",
            doc_type=".txt",
            metadata={},
        )

        chunks = list(pipeline.chunk(doc))

        assert len(chunks) >= 1
        assert pipeline.metrics.chunks_created >= 1

    @patch("src.data.pipeline.pipeline.get_embedding_model")
    def test_embed_generates_embeddings(self, mock_get_model, sample_config):
        """Test embedding generation."""
        from src.data.embedding.models import EmbeddingResult
        from src.data.ingestion.chunkers import Chunk

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.embed.return_value = [
            EmbeddingResult(
                embedding=[0.1] * 384,
                model="test-model",
                dimension=384,
            )
        ]
        mock_model.dimension = 384
        mock_get_model.return_value = mock_model

        config = DataPipelineConfig(**sample_config)
        pipeline = DataPipeline(config)

        chunks = [
            Chunk(
                text="Test chunk",
                chunk_index=0,
                start_char=0,
                end_char=10,
                metadata={},
            )
        ]

        results = pipeline.embed(chunks)

        assert len(results) == 1
        assert len(results[0][1]) == 384  # Embedding dimension

    def test_dry_run_skips_storage(self, sample_config):
        """Test dry run mode skips storage."""
        config = DataPipelineConfig(**sample_config)
        config.dry_run = True

        pipeline = DataPipeline(config)

        # Should not create Milvus client in dry run
        assert pipeline.milvus_client is None


class TestPipelineMetrics:
    """Tests for PipelineMetrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = PipelineMetrics()

        assert metrics.documents_loaded == 0
        assert metrics.chunks_created == 0
        assert metrics.embeddings_generated == 0
        assert metrics.vectors_stored == 0
        assert metrics.errors == []

    def test_metrics_tracking(self):
        """Test metrics can be updated."""
        metrics = PipelineMetrics()
        metrics.documents_loaded = 10
        metrics.chunks_created = 100
        metrics.errors.append("Test error")

        assert metrics.documents_loaded == 10
        assert len(metrics.errors) == 1
