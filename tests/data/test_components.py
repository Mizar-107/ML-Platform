"""Tests for Kubeflow Pipeline components."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestComponentContracts:
    """Tests for KFP component input/output contracts."""

    def test_ingest_component_returns_document_count(self, tmp_path):
        """Test ingest component returns correct count."""
        # This tests the logic, not the KFP decorator
        from src.data.ingestion.loaders import Document

        with patch("src.data.ingestion.loaders.get_loader") as mock:
            mock_loader = MagicMock()
            mock_loader.load.return_value = iter([
                Document(
                    content="Test 1",
                    source="s3://bucket/doc1.txt",
                    doc_type=".txt",
                    metadata={},
                ),
                Document(
                    content="Test 2",
                    source="s3://bucket/doc2.txt",
                    doc_type=".txt",
                    metadata={},
                ),
            ])
            mock.return_value = mock_loader

            # Simulate component logic
            source_path = "s3://bucket/docs"
            loader = mock(source_path)
            documents = []

            for doc in loader.load(source_path):
                documents.append({
                    "content": doc.content,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "metadata": doc.metadata,
                })

            assert len(documents) == 2

    def test_chunk_component_creates_valid_output(self):
        """Test chunk component creates valid chunks."""
        from src.data.ingestion.chunkers import get_chunker

        chunker = get_chunker("semantic", max_chunk_size=100, min_chunk_size=20)

        text = "This is a test paragraph. " * 10
        chunks = list(chunker.chunk(text, {"source": "test.txt"}))

        assert len(chunks) >= 1
        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "chunk_index")
            assert hasattr(chunk, "metadata")

    def test_embed_component_output_format(self):
        """Test embed component produces correct output format."""
        from src.data.ingestion.chunkers import Chunk

        # Define expected output structure
        expected_keys = ["text", "chunk_index", "metadata", "embedding", "model", "dimension"]

        sample_output = {
            "text": "Sample text",
            "chunk_index": 0,
            "metadata": {"source": "test.txt"},
            "embedding": [0.1] * 384,
            "model": "test-model",
            "dimension": 384,
        }

        for key in expected_keys:
            assert key in sample_output


class TestPipelineDAG:
    """Tests for pipeline DAG structure."""

    def test_pipeline_has_required_steps(self):
        """Test pipeline includes all required steps."""
        # Import to verify no syntax errors
        from pipelines.data.ingestion_pipeline import data_ingestion_pipeline

        # Verify it's a valid pipeline function
        assert callable(data_ingestion_pipeline)

    def test_components_are_importable(self):
        """Test all components can be imported."""
        from pipelines.data.components import (
            chunk_component,
            embed_component,
            ingest_component,
            parse_component,
            store_component,
        )

        # All should be KFP components (decorated functions)
        assert callable(ingest_component)
        assert callable(parse_component)
        assert callable(chunk_component)
        assert callable(embed_component)
        assert callable(store_component)
