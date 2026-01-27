"""Test fixtures for data pipeline tests."""

import pytest


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "This is a sample document about machine learning. "
            "Machine learning is a subset of artificial intelligence.",
            "source": "test/doc1.txt",
            "doc_type": ".txt",
            "metadata": {"author": "test"},
        },
        {
            "content": "Natural language processing enables computers to understand text. "
            "NLP is used in many applications including chatbots and translation.",
            "source": "test/doc2.txt",
            "doc_type": ".txt",
            "metadata": {"author": "test"},
        },
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "text": "This is a sample document about machine learning.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 48,
            "metadata": {"source": "test/doc1.txt"},
        },
        {
            "text": "Machine learning is a subset of artificial intelligence.",
            "chunk_index": 1,
            "start_char": 49,
            "end_char": 104,
            "metadata": {"source": "test/doc1.txt"},
        },
        {
            "text": "Natural language processing enables computers to understand text.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 64,
            "metadata": {"source": "test/doc2.txt"},
        },
    ]


@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        "source_path": "tests/fixtures/docs",
        "chunk_strategy": "semantic",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model_type": "sentence-transformer",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_batch_size": 32,
        "milvus_host": "localhost",
        "milvus_port": 19530,
        "collection_name": "test_documents",
        "dry_run": True,
    }
