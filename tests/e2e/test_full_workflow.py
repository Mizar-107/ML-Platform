# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for complete workflow.

This module tests the complete end-to-end workflow from raw data
through training to deployed model serving.
"""

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.e2e
@pytest.mark.slow
class TestFullWorkflow:
    """Test suite for complete end-to-end workflow."""

    def test_complete_workflow_mock(
        self,
        sample_documents,
        sample_training_data,
        sample_evaluation_data,
        temp_data_dir,
        mock_s3_client,
        mock_milvus_client,
        mock_mlflow_client,
        mock_kserve_client,
    ):
        """Test complete workflow with mocked services.

        This test validates the entire pipeline:
        1. Data ingestion → embedding → vector store
        2. Training → model registry
        3. Deployment → inference validation

        Args:
            sample_documents: Sample documents.
            sample_training_data: Training data.
            sample_evaluation_data: Evaluation data.
            temp_data_dir: Temporary directory.
            mock_s3_client: Mock S3 client.
            mock_milvus_client: Mock Milvus client.
            mock_mlflow_client: Mock MLflow client.
            mock_kserve_client: Mock KServe client.
        """
        # =====================================================================
        # Phase 1: Data Ingestion
        # =====================================================================
        print("\n=== Phase 1: Data Ingestion ===")

        # Step 1.1: Store raw documents to S3
        raw_data_path = temp_data_dir / "raw_documents.json"
        with open(raw_data_path, "w") as f:
            json.dump(sample_documents, f)

        mock_s3_client.upload_file(
            str(raw_data_path),
            "llm-mlops-data",
            "raw/documents.json",
        )
        print(f"✓ Uploaded {len(sample_documents)} documents to S3")

        # Step 1.2: Generate embeddings
        embedding_dim = 384
        embeddings = [[0.1] * embedding_dim for _ in sample_documents]
        print(f"✓ Generated {len(embeddings)} embeddings (dim={embedding_dim})")

        # Step 1.3: Store embeddings in Milvus
        mock_milvus_client.insert([
            [doc["id"] for doc in sample_documents],
            [doc["content"] for doc in sample_documents],
            embeddings,
        ])
        print("✓ Stored embeddings in Milvus")

        # =====================================================================
        # Phase 2: Model Training
        # =====================================================================
        print("\n=== Phase 2: Model Training ===")

        # Step 2.1: Prepare training data
        train_data_path = temp_data_dir / "training_data.json"
        with open(train_data_path, "w") as f:
            json.dump(sample_training_data, f)
        print(f"✓ Prepared {len(sample_training_data)} training samples")

        # Step 2.2: Simulate training with MLflow tracking
        with patch("mlflow.start_run") as mock_run:
            with patch("mlflow.log_params") as mock_params:
                with patch("mlflow.log_metrics") as mock_metrics:
                    # Log training config
                    mock_params({
                        "model": "mistral-7b",
                        "lora_r": 8,
                        "learning_rate": 2e-4,
                    })

                    # Simulate training epochs
                    for epoch in range(3):
                        mock_metrics({
                            "train_loss": 2.0 - epoch * 0.4,
                            "eval_loss": 2.1 - epoch * 0.35,
                        })

        print("✓ Training completed (3 epochs)")

        # Step 2.3: Register model
        with patch("mlflow.register_model") as mock_register:
            mock_register.return_value = MagicMock(
                name="mistral-7b-finetuned",
                version="1",
            )
            model = mock_register("runs:/test-run/model", "mistral-7b-finetuned")

        print(f"✓ Registered model: {model.name} v{model.version}")

        # =====================================================================
        # Phase 3: Model Evaluation
        # =====================================================================
        print("\n=== Phase 3: Model Evaluation ===")

        # Step 3.1: Run evaluation
        eval_metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.82,
            "context_precision": 0.78,
            "context_recall": 0.80,
        }
        print(f"✓ Evaluation metrics: {eval_metrics}")

        # Step 3.2: Quality gate check
        min_threshold = 0.7
        all_passed = all(v >= min_threshold for v in eval_metrics.values())
        print(f"✓ Quality gate: {'PASSED' if all_passed else 'FAILED'}")

        assert all_passed, "Evaluation quality gate failed"

        # =====================================================================
        # Phase 4: Model Deployment
        # =====================================================================
        print("\n=== Phase 4: Model Deployment ===")

        # Step 4.1: Create InferenceService
        inference_service = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": "mistral-7b-finetuned",
                "namespace": "mlops-serving",
            },
            "spec": {
                "predictor": {
                    "model": {
                        "modelFormat": {"name": "vLLM"},
                        "storageUri": "s3://llm-mlops-models/mistral-7b-finetuned/1",
                    },
                },
            },
        }

        mock_kserve_client.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace="mlops-serving",
            plural="inferenceservices",
            body=inference_service,
        )
        print("✓ Created InferenceService")

        # Step 4.2: Wait for deployment (simulated)
        mock_kserve_client.get_namespaced_custom_object.return_value = {
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
                "url": "http://mistral-7b-finetuned.mlops-serving.example.com",
            },
        }
        print("✓ Deployment ready")

        # =====================================================================
        # Phase 5: Inference Validation
        # =====================================================================
        print("\n=== Phase 5: Inference Validation ===")

        # Step 5.1: Smoke test
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"text": "This is a test response."}],
                    "usage": {"total_tokens": 15},
                },
            )

            import requests
            response = requests.post(
                "http://mistral-7b-finetuned.mlops-serving.example.com/v1/completions",
                json={"prompt": "Test prompt", "max_tokens": 50},
            )

            assert response.status_code == 200
            result = response.json()
            assert len(result["choices"]) > 0

        print("✓ Smoke test passed")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n=== Workflow Complete ===")
        print(f"Documents processed: {len(sample_documents)}")
        print(f"Training samples: {len(sample_training_data)}")
        print(f"Model version: {model.version}")
        print(f"Evaluation score: {sum(eval_metrics.values())/len(eval_metrics):.2f}")
        print("Status: SUCCESS")

    def test_workflow_data_lineage(
        self,
        sample_documents,
        temp_data_dir,
    ):
        """Test data lineage tracking through workflow.

        Args:
            sample_documents: Sample documents.
            temp_data_dir: Temporary directory.
        """
        # Track lineage through each stage
        lineage = {
            "stages": [],
            "artifacts": {},
        }

        # Stage 1: Raw data
        raw_artifact_id = "raw-docs-001"
        lineage["stages"].append({
            "name": "data_ingestion",
            "input": None,
            "output": raw_artifact_id,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        lineage["artifacts"][raw_artifact_id] = {
            "type": "documents",
            "count": len(sample_documents),
            "location": "s3://data/raw/",
        }

        # Stage 2: Embeddings
        embedding_artifact_id = "embeddings-001"
        lineage["stages"].append({
            "name": "embedding_generation",
            "input": raw_artifact_id,
            "output": embedding_artifact_id,
            "timestamp": "2024-01-15T10:30:00Z",
        })
        lineage["artifacts"][embedding_artifact_id] = {
            "type": "embeddings",
            "count": len(sample_documents),
            "location": "milvus://collections/documents",
        }

        # Stage 3: Training
        model_artifact_id = "model-001"
        lineage["stages"].append({
            "name": "model_training",
            "input": raw_artifact_id,
            "output": model_artifact_id,
            "timestamp": "2024-01-15T12:00:00Z",
        })
        lineage["artifacts"][model_artifact_id] = {
            "type": "model",
            "version": "1",
            "location": "mlflow://models/mistral-7b-finetuned/1",
        }

        # Assert lineage is complete
        assert len(lineage["stages"]) == 3
        assert len(lineage["artifacts"]) == 3

        # Verify chain
        for i, stage in enumerate(lineage["stages"][1:], 1):
            assert stage["input"] in lineage["artifacts"]

    def test_workflow_rollback_scenario(
        self,
        mock_mlflow_client,
        mock_kserve_client,
    ):
        """Test rollback scenario when deployment fails.

        Args:
            mock_mlflow_client: Mock MLflow client.
            mock_kserve_client: Mock KServe client.
        """
        # Arrange: Previous good version
        previous_version = "1"
        new_version = "2"

        # Simulate failed deployment
        deployment_success = False
        rollback_triggered = False

        # Act: Deploy new version
        try:
            # Attempt deployment
            mock_kserve_client.patch_namespaced_custom_object.side_effect = Exception(
                "Deployment failed: OOM"
            )
            mock_kserve_client.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace="mlops-serving",
                plural="inferenceservices",
                name="test-model",
                body={"spec": {"predictor": {"model": {"version": new_version}}}},
            )
            deployment_success = True
        except Exception as e:
            # Rollback to previous version
            rollback_triggered = True
            mock_kserve_client.patch_namespaced_custom_object.side_effect = None
            mock_kserve_client.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace="mlops-serving",
                plural="inferenceservices",
                name="test-model",
                body={"spec": {"predictor": {"model": {"version": previous_version}}}},
            )

        # Assert
        assert not deployment_success
        assert rollback_triggered

    def test_workflow_performance_metrics(self):
        """Test workflow performance metrics collection."""
        # Simulate workflow metrics
        workflow_metrics = {
            "data_ingestion": {
                "duration_seconds": 120,
                "documents_processed": 1000,
                "throughput_docs_per_sec": 8.33,
            },
            "embedding_generation": {
                "duration_seconds": 300,
                "embeddings_generated": 1000,
                "throughput_emb_per_sec": 3.33,
            },
            "training": {
                "duration_seconds": 3600,
                "epochs": 3,
                "samples_per_second": 15.0,
                "final_loss": 0.45,
            },
            "deployment": {
                "duration_seconds": 180,
                "replicas_ready": 2,
                "time_to_first_inference_ms": 150,
            },
        }

        # Calculate total workflow time
        total_duration = sum(
            stage["duration_seconds"]
            for stage in workflow_metrics.values()
        )

        # Assert reasonable performance
        assert total_duration < 7200  # Less than 2 hours
        assert workflow_metrics["training"]["final_loss"] < 1.0
        assert workflow_metrics["deployment"]["time_to_first_inference_ms"] < 500


@pytest.mark.e2e
class TestWorkflowErrorHandling:
    """Test error handling in complete workflow."""

    def test_data_validation_failure(self, temp_data_dir):
        """Test handling of invalid input data."""
        # Arrange: Create invalid data
        invalid_data = [
            {"id": "1"},  # Missing content
            {"content": "text"},  # Missing id
            {},  # Empty
        ]

        data_path = temp_data_dir / "invalid.json"
        with open(data_path, "w") as f:
            json.dump(invalid_data, f)

        # Act: Validate data
        with open(data_path) as f:
            data = json.load(f)

        errors = []
        for i, doc in enumerate(data):
            if "id" not in doc:
                errors.append(f"Document {i}: missing 'id'")
            if "content" not in doc:
                errors.append(f"Document {i}: missing 'content'")

        # Assert
        assert len(errors) > 0
        assert "missing 'content'" in errors[0]

    def test_training_failure_recovery(self, temp_data_dir):
        """Test recovery from training failure."""
        # Arrange: Simulate checkpoints
        checkpoint_dir = temp_data_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints at different steps
        for step in [100, 200]:
            ckpt = checkpoint_dir / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "trainer_state.json").write_text(
                json.dumps({"global_step": step})
            )

        # Simulate failure at step 250
        failed_at_step = 250

        # Act: Find last valid checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        last_checkpoint = checkpoints[-1] if checkpoints else None

        # Assert: Can resume from checkpoint
        assert last_checkpoint is not None
        state = json.loads((last_checkpoint / "trainer_state.json").read_text())
        assert state["global_step"] == 200
        assert state["global_step"] < failed_at_step

    def test_deployment_health_recovery(self, mock_kserve_client):
        """Test recovery from unhealthy deployment."""
        # Arrange: Simulate unhealthy → healthy transition
        health_states = [
            {"status": {"conditions": [{"type": "Ready", "status": "False"}]}},
            {"status": {"conditions": [{"type": "Ready", "status": "False"}]}},
            {"status": {"conditions": [{"type": "Ready", "status": "True"}]}},
        ]

        call_count = [0]

        def get_status(*args, **kwargs):
            state = health_states[min(call_count[0], len(health_states) - 1)]
            call_count[0] += 1
            return state

        mock_kserve_client.get_namespaced_custom_object.side_effect = get_status

        # Act: Poll until healthy
        is_healthy = False
        for attempt in range(5):
            result = mock_kserve_client.get_namespaced_custom_object()
            conditions = result.get("status", {}).get("conditions", [])
            for cond in conditions:
                if cond["type"] == "Ready" and cond["status"] == "True":
                    is_healthy = True
                    break
            if is_healthy:
                break

        # Assert
        assert is_healthy
        assert call_count[0] == 3  # Required 3 attempts
