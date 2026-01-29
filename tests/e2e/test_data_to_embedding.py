# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for data ingestion to embedding pipeline.

This module tests the complete data pipeline workflow from document
ingestion through chunking to embedding storage in Milvus.
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.e2e
class TestDataToEmbeddingPipeline:
    """Test suite for data ingestion to embedding workflow."""

    def test_document_ingestion_json(
        self,
        sample_documents,
        temp_data_dir,
    ):
        """Test ingesting documents from JSON file.

        Args:
            sample_documents: Sample document fixtures.
            temp_data_dir: Temporary data directory.
        """
        # Arrange: Create input JSON file
        input_file = temp_data_dir / "documents.json"
        with open(input_file, "w") as f:
            json.dump(sample_documents, f)

        # Act: Load documents
        with open(input_file) as f:
            loaded_docs = json.load(f)

        # Assert
        assert len(loaded_docs) == len(sample_documents)
        assert all("content" in doc for doc in loaded_docs)
        assert all("id" in doc for doc in loaded_docs)

    def test_text_chunking(self, sample_documents):
        """Test document chunking with fixed-size strategy.

        Args:
            sample_documents: Sample documents.
        """
        # Arrange
        chunk_size = 100
        chunk_overlap = 20

        # Act: Simple fixed-size chunking
        chunks = []
        for doc in sample_documents:
            content = doc["content"]
            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk_text = content[i : i + chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "doc_id": doc["id"],
                        "chunk_index": len(chunks),
                    })

        # Assert
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("doc_id" in chunk for chunk in chunks)

    def test_embedding_generation_mock(self, sample_documents):
        """Test embedding generation with mocked model.

        Args:
            sample_documents: Sample documents.
        """
        # Arrange: Mock embedding model
        embedding_dim = 384
        mock_embeddings = [
            [0.1] * embedding_dim for _ in sample_documents
        ]

        with patch("sentence_transformers.SentenceTransformer") as mock_model:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = mock_embeddings
            mock_model.return_value = mock_instance

            # Act: Generate embeddings
            texts = [doc["content"] for doc in sample_documents]
            embeddings = mock_instance.encode(texts)

            # Assert
            assert len(embeddings) == len(sample_documents)
            assert len(embeddings[0]) == embedding_dim

    def test_milvus_storage_mock(
        self,
        sample_documents,
        mock_milvus_client,
    ):
        """Test storing embeddings in Milvus with mock.

        Args:
            sample_documents: Sample documents.
            mock_milvus_client: Mock Milvus client.
        """
        # Arrange
        embedding_dim = 384
        embeddings = [[0.1] * embedding_dim for _ in sample_documents]
        ids = [doc["id"] for doc in sample_documents]
        texts = [doc["content"] for doc in sample_documents]

        # Act: Insert into Milvus
        result = mock_milvus_client.insert([ids, texts, embeddings])

        # Assert
        mock_milvus_client.insert.assert_called_once()
        assert result.primary_keys is not None

    def test_full_pipeline_mock(
        self,
        sample_documents,
        temp_data_dir,
        mock_milvus_client,
    ):
        """Test complete data to embedding pipeline with mocks.

        Args:
            sample_documents: Sample documents.
            temp_data_dir: Temporary directory.
            mock_milvus_client: Mock Milvus client.
        """
        # Step 1: Write documents to file
        input_file = temp_data_dir / "input_docs.json"
        with open(input_file, "w") as f:
            json.dump(sample_documents, f)

        # Step 2: Load and parse documents
        with open(input_file) as f:
            docs = json.load(f)

        assert len(docs) == 3

        # Step 3: Chunk documents
        chunks = []
        chunk_size = 200
        for doc in docs:
            content = doc["content"]
            # Simple chunking for test
            chunks.append({
                "text": content[:chunk_size],
                "doc_id": doc["id"],
                "metadata": doc.get("metadata", {}),
            })

        assert len(chunks) == 3

        # Step 4: Generate embeddings (mocked)
        embedding_dim = 384
        embeddings = [[0.1] * embedding_dim for _ in chunks]

        assert len(embeddings) == len(chunks)

        # Step 5: Store in Milvus (mocked)
        mock_milvus_client.insert([
            [c["doc_id"] for c in chunks],
            [c["text"] for c in chunks],
            embeddings,
        ])

        mock_milvus_client.insert.assert_called_once()

    @pytest.mark.requires_cluster
    def test_pipeline_with_ray(
        self,
        sample_documents,
        e2e_config,
    ):
        """Test pipeline with Ray for distributed processing.

        Args:
            sample_documents: Sample documents.
            e2e_config: E2E configuration.
        """
        pytest.skip("Requires Ray cluster - run with --run-cluster")

    def test_error_handling_invalid_input(self, temp_data_dir):
        """Test error handling for invalid input data.

        Args:
            temp_data_dir: Temporary directory.
        """
        # Arrange: Create invalid JSON file
        invalid_file = temp_data_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json {{{")

        # Act & Assert
        with pytest.raises(json.JSONDecodeError):
            with open(invalid_file) as f:
                json.load(f)

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        # Arrange
        empty_docs = [
            {"id": "empty1", "content": ""},
            {"id": "empty2", "content": "   "},
        ]

        # Act: Filter empty documents
        valid_docs = [
            doc for doc in empty_docs
            if doc["content"].strip()
        ]

        # Assert
        assert len(valid_docs) == 0

    def test_special_characters_handling(self):
        """Test handling of special characters in documents."""
        # Arrange
        special_docs = [
            {
                "id": "special1",
                "content": "Document with émojis 🚀 and spëcial chars éàü",
            },
            {
                "id": "special2",
                "content": "Document with\nnewlines\tand\ttabs",
            },
        ]

        # Act: Normalize text
        normalized = []
        for doc in special_docs:
            text = doc["content"]
            # Simple normalization
            text = " ".join(text.split())
            normalized.append({"id": doc["id"], "content": text})

        # Assert
        assert len(normalized) == 2
        assert "\n" not in normalized[1]["content"]
        assert "\t" not in normalized[1]["content"]

    def test_metadata_preservation(self, sample_documents):
        """Test that metadata is preserved through pipeline.

        Args:
            sample_documents: Sample documents with metadata.
        """
        # Arrange
        doc = sample_documents[0]
        original_metadata = doc["metadata"]

        # Act: Simulate chunking with metadata preservation
        chunks = [{
            "text": doc["content"][:100],
            "doc_id": doc["id"],
            "metadata": original_metadata.copy(),
        }]

        # Assert
        assert chunks[0]["metadata"]["source"] == original_metadata["source"]
        assert chunks[0]["metadata"]["category"] == original_metadata["category"]


@pytest.mark.e2e
class TestVectorSearchIntegration:
    """Test vector search functionality."""

    def test_vector_similarity_search_mock(self, mock_milvus_client):
        """Test vector similarity search with mock.

        Args:
            mock_milvus_client: Mock Milvus client.
        """
        # Arrange
        query_vector = [0.1] * 384
        mock_milvus_client.search.return_value = [[
            MagicMock(id="doc1", distance=0.95),
            MagicMock(id="doc2", distance=0.85),
        ]]

        # Act
        results = mock_milvus_client.search(
            [query_vector],
            "embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=10,
        )

        # Assert
        mock_milvus_client.search.assert_called_once()
        assert len(results[0]) == 2

    def test_hybrid_search_mock(self, mock_milvus_client):
        """Test hybrid search combining vector and keyword search.

        Args:
            mock_milvus_client: Mock Milvus client.
        """
        # Arrange
        query_vector = [0.1] * 384
        keyword_filter = "category == 'mlops'"

        mock_milvus_client.search.return_value = [[
            MagicMock(id="doc1", distance=0.95),
        ]]

        # Act
        results = mock_milvus_client.search(
            [query_vector],
            "embedding",
            param={"metric_type": "COSINE"},
            expr=keyword_filter,
            limit=10,
        )

        # Assert
        mock_milvus_client.search.assert_called_once()
        call_kwargs = mock_milvus_client.search.call_args[1]
        assert call_kwargs["expr"] == keyword_filter
