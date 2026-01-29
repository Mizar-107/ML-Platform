# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for end-to-end tests.

This module provides shared fixtures for E2E testing including
mock services, test data, and Kubernetes client setup.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Environment Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "requires_cluster: mark test as requiring Kubernetes cluster"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def e2e_config() -> dict[str, Any]:
    """Configuration for E2E tests.

    Returns:
        Dictionary with test configuration.
    """
    return {
        "namespace": os.getenv("E2E_NAMESPACE", "mlops-test"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
        "milvus_port": int(os.getenv("MILVUS_PORT", "19530")),
        "ray_address": os.getenv("RAY_ADDRESS", "ray://localhost:10001"),
        "s3_bucket": os.getenv("E2E_S3_BUCKET", "mlops-test-bucket"),
        "model_registry": os.getenv("MODEL_REGISTRY", "mlops-model-registry"),
        "kserve_domain": os.getenv("KSERVE_DOMAIN", "example.com"),
        "timeout_seconds": int(os.getenv("E2E_TIMEOUT", "300")),
        "skip_cleanup": os.getenv("E2E_SKIP_CLEANUP", "false").lower() == "true",
    }


@pytest.fixture(scope="session")
def cluster_available() -> bool:
    """Check if Kubernetes cluster is available.

    Returns:
        True if cluster is accessible.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        v1 = client.CoreV1Api()
        v1.list_namespace(limit=1)
        return True
    except Exception:
        return False


# =============================================================================
# Kubernetes Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def k8s_core_client():
    """Kubernetes CoreV1Api client.

    Returns:
        CoreV1Api client or None if not available.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        return client.CoreV1Api()
    except Exception:
        return None


@pytest.fixture(scope="session")
def k8s_custom_client():
    """Kubernetes CustomObjectsApi client.

    Returns:
        CustomObjectsApi client or None if not available.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        return client.CustomObjectsApi()
    except Exception:
        return None


@pytest.fixture(scope="session")
def k8s_apps_client():
    """Kubernetes AppsV1Api client.

    Returns:
        AppsV1Api client or None if not available.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        return client.AppsV1Api()
    except Exception:
        return None


@pytest.fixture
def test_namespace(k8s_core_client, e2e_config) -> Generator[str, None, None]:
    """Create and cleanup test namespace.

    Args:
        k8s_core_client: Kubernetes client.
        e2e_config: E2E configuration.

    Yields:
        Namespace name.
    """
    from kubernetes import client

    namespace = f"{e2e_config['namespace']}-{os.getpid()}"

    if k8s_core_client:
        try:
            # Create namespace
            ns = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=namespace,
                    labels={"app": "e2e-test", "cleanup": "true"},
                )
            )
            k8s_core_client.create_namespace(ns)
        except client.ApiException as e:
            if e.status != 409:  # Already exists
                raise

    yield namespace

    # Cleanup
    if k8s_core_client and not e2e_config["skip_cleanup"]:
        try:
            k8s_core_client.delete_namespace(namespace)
        except client.ApiException:
            pass


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample documents for data pipeline testing.

    Returns:
        List of sample documents.
    """
    return [
        {
            "id": "doc1",
            "content": "Machine Learning Operations (MLOps) is a set of practices "
                      "that combines Machine Learning, DevOps and Data Engineering.",
            "metadata": {"source": "test", "category": "mlops"},
        },
        {
            "id": "doc2",
            "content": "Large Language Models (LLMs) are neural networks trained "
                      "on massive amounts of text data to generate human-like text.",
            "metadata": {"source": "test", "category": "llm"},
        },
        {
            "id": "doc3",
            "content": "Retrieval Augmented Generation (RAG) combines retrieval "
                      "systems with generative models for improved accuracy.",
            "metadata": {"source": "test", "category": "rag"},
        },
    ]


@pytest.fixture
def sample_training_data() -> list[dict[str, Any]]:
    """Sample training data for fine-tuning tests.

    Returns:
        List of training examples.
    """
    return [
        {
            "instruction": "Explain MLOps in simple terms.",
            "input": "",
            "output": "MLOps is the practice of applying DevOps principles to "
                     "machine learning systems.",
        },
        {
            "instruction": "What is a Large Language Model?",
            "input": "",
            "output": "A Large Language Model (LLM) is an AI model trained on "
                     "vast amounts of text to understand and generate human language.",
        },
        {
            "instruction": "Describe RAG in one sentence.",
            "input": "",
            "output": "RAG enhances LLM responses by retrieving relevant "
                     "information from external sources.",
        },
    ]


@pytest.fixture
def sample_evaluation_data() -> list[dict[str, Any]]:
    """Sample evaluation data for testing.

    Returns:
        List of evaluation samples.
    """
    return [
        {
            "question": "What is MLOps?",
            "answer": "MLOps is the practice of combining ML, DevOps, and Data Engineering.",
            "contexts": [
                "MLOps (Machine Learning Operations) integrates ML, DevOps, and data engineering practices."
            ],
            "ground_truth": "MLOps combines Machine Learning, DevOps and Data Engineering.",
        },
        {
            "question": "What are LLMs?",
            "answer": "LLMs are large neural networks for text generation.",
            "contexts": [
                "Large Language Models are neural networks trained on massive text data."
            ],
            "ground_truth": "LLMs are neural networks for generating human-like text.",
        },
    ]


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory(prefix="e2e_test_") as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Mock Services
# =============================================================================


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing without AWS.

    Returns:
        Mock S3 client.
    """
    mock_client = MagicMock()

    # Mock common operations
    mock_client.list_buckets.return_value = {"Buckets": []}
    mock_client.put_object.return_value = {}
    mock_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: b'{"test": "data"}')
    }

    with patch("boto3.client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_milvus_client():
    """Mock Milvus client for testing without Milvus.

    Returns:
        Mock Milvus connections and Collection.
    """
    mock_connections = MagicMock()
    mock_collection = MagicMock()

    mock_collection.num_entities = 100
    mock_collection.insert.return_value = MagicMock(primary_keys=[1, 2, 3])
    mock_collection.search.return_value = [[]]

    with patch("pymilvus.connections", mock_connections):
        with patch("pymilvus.Collection", return_value=mock_collection):
            yield mock_collection


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing.

    Returns:
        Mock MLflow client.
    """
    mock_client = MagicMock()

    mock_client.create_experiment.return_value = "test-experiment-id"
    mock_client.create_run.return_value = MagicMock(
        info=MagicMock(run_id="test-run-id")
    )
    mock_client.get_model_version.return_value = MagicMock(
        version="1",
        current_stage="Production",
    )

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        with patch("mlflow.set_tracking_uri"):
            with patch("mlflow.set_experiment"):
                with patch("mlflow.start_run"):
                    with patch("mlflow.log_metric"):
                        yield mock_client


@pytest.fixture
def mock_kserve_client(k8s_custom_client):
    """Mock KServe client for testing.

    Args:
        k8s_custom_client: Kubernetes custom objects client.

    Returns:
        Mock client or real client if available.
    """
    if k8s_custom_client:
        return k8s_custom_client

    mock_client = MagicMock()
    mock_client.create_namespaced_custom_object.return_value = {}
    mock_client.get_namespaced_custom_object.return_value = {
        "status": {"conditions": [{"type": "Ready", "status": "True"}]}
    }

    return mock_client


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def wait_for_condition():
    """Factory fixture for waiting on conditions.

    Returns:
        Function to wait for conditions.
    """
    import time

    def _wait(condition_fn, timeout=60, interval=5, description="condition"):
        """Wait for condition to be true.

        Args:
            condition_fn: Function returning bool.
            timeout: Maximum wait time in seconds.
            interval: Check interval in seconds.
            description: Description for error message.

        Raises:
            TimeoutError: If condition not met within timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if condition_fn():
                return True
            time.sleep(interval)

        raise TimeoutError(f"Timeout waiting for {description}")

    return _wait


@pytest.fixture
def assert_eventually():
    """Factory fixture for eventual assertions.

    Returns:
        Function for eventual assertions.
    """
    import time

    def _assert(assertion_fn, timeout=30, interval=2, message="Assertion failed"):
        """Assert condition eventually becomes true.

        Args:
            assertion_fn: Function to assert.
            timeout: Maximum wait time.
            interval: Check interval.
            message: Error message on failure.

        Raises:
            AssertionError: If assertion never passes.
        """
        start = time.time()
        last_error = None

        while time.time() - start < timeout:
            try:
                assertion_fn()
                return
            except AssertionError as e:
                last_error = e
                time.sleep(interval)

        raise AssertionError(f"{message}: {last_error}")

    return _assert


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_resources(request, e2e_config):
    """Automatic cleanup of test resources.

    Args:
        request: Pytest request object.
        e2e_config: E2E configuration.
    """
    yield

    # Cleanup logic after test
    if e2e_config["skip_cleanup"]:
        return

    # Add any global cleanup here
    pass
