"""Tests for training callbacks."""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path


class TestMLflowCallback:
    """Tests for MLflow callback."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        from src.training.callbacks import MLflowCallback

        callback = MLflowCallback(
            experiment_name="test-experiment",
            run_name="test-run",
            log_model=True,
        )

        assert callback.experiment_name == "test-experiment"
        assert callback.run_name == "test-run"
        assert callback.log_model is True

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_mlflow(self, mock_set_exp, mock_set_uri):
        """Test MLflow setup."""
        from src.training.callbacks import MLflowCallback

        callback = MLflowCallback(
            experiment_name="test",
            tracking_uri="http://localhost:5000",
        )
        callback._setup_mlflow()

        mock_set_uri.assert_called_once_with("http://localhost:5000")
        mock_set_exp.assert_called_once_with("test")
        assert callback._initialized is True

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_params")
    def test_on_train_begin(self, mock_log_params, mock_start_run, mock_set_exp, mock_set_uri):
        """Test on_train_begin callback."""
        from src.training.callbacks import MLflowCallback

        # Setup mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value = mock_run

        callback = MLflowCallback(experiment_name="test")

        # Create mock training args
        args = MagicMock()
        args.learning_rate = 2e-4
        args.num_train_epochs = 3
        args.per_device_train_batch_size = 4
        args.gradient_accumulation_steps = 4
        args.warmup_ratio = 0.03
        args.weight_decay = 0.01
        args.max_grad_norm = 0.3
        args.lr_scheduler_type = "cosine"
        args.optim = "adamw"
        args.bf16 = True
        args.fp16 = False
        args.gradient_checkpointing = True
        args.seed = 42

        state = MagicMock()
        control = MagicMock()

        callback.on_train_begin(args, state, control)

        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_metrics")
    def test_on_log(self, mock_log_metrics, mock_start_run, mock_set_exp, mock_set_uri):
        """Test on_log callback."""
        from src.training.callbacks import MLflowCallback

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value = mock_run

        callback = MLflowCallback(experiment_name="test")
        callback._setup_mlflow()
        callback._run = mock_run

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()

        logs = {
            "loss": 0.5,
            "learning_rate": 1e-4,
            "epoch": 1.5,
        }

        callback.on_log(args, state, control, logs=logs)

        mock_log_metrics.assert_called_once()

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.end_run")
    def test_on_train_end(self, mock_end_run, mock_start_run, mock_set_exp, mock_set_uri):
        """Test on_train_end callback."""
        from src.training.callbacks import MLflowCallback

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value = mock_run

        callback = MLflowCallback(experiment_name="test", log_model=False)
        callback._setup_mlflow()
        callback._run = mock_run

        args = MagicMock()
        args.output_dir = "/tmp/output"
        state = MagicMock()
        state.best_metric = 0.5
        control = MagicMock()

        callback.on_train_end(args, state, control)

        mock_end_run.assert_called_once()


class TestMLflowModelRegistry:
    """Tests for MLflow model registry."""

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    def test_registry_initialization(self, mock_client, mock_set_uri):
        """Test registry initialization."""
        from src.training.callbacks import MLflowModelRegistry

        registry = MLflowModelRegistry(tracking_uri="http://localhost:5000")

        mock_set_uri.assert_called_once()
        assert registry.tracking_uri == "http://localhost:5000"

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.register_model")
    def test_register_model(self, mock_register, mock_client, mock_set_uri):
        """Test model registration."""
        from src.training.callbacks import MLflowModelRegistry

        mock_result = MagicMock()
        mock_result.version = "1"
        mock_register.return_value = mock_result

        registry = MLflowModelRegistry()
        version = registry.register_model(
            run_id="test-run",
            model_name="test-model",
        )

        assert version == "1"
        mock_register.assert_called_once()


class TestCheckpointCallback:
    """Tests for checkpoint callback."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        from src.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            checkpoint_dir="./checkpoints",
            s3_bucket="my-bucket",
            keep_n_checkpoints=3,
        )

        assert callback.checkpoint_dir == Path("./checkpoints")
        assert callback.s3_bucket == "my-bucket"
        assert callback.keep_n_checkpoints == 3

    def test_get_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint."""
        from src.training.callbacks import CheckpointCallback

        # Create fake checkpoints
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-50").mkdir()

        callback = CheckpointCallback(checkpoint_dir=tmp_path)

        latest = callback.get_latest_checkpoint()

        assert latest.name == "checkpoint-200"

    def test_cleanup_old_checkpoints(self, tmp_path):
        """Test old checkpoint cleanup."""
        from src.training.callbacks import CheckpointCallback

        callback = CheckpointCallback(
            checkpoint_dir=tmp_path,
            keep_n_checkpoints=2,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        # Add checkpoints
        callback._checkpoints = [
            {"step": 100, "path": str(tmp_path / "ckpt-100"), "metric": 0.5},
            {"step": 200, "path": str(tmp_path / "ckpt-200"), "metric": 0.3},
            {"step": 300, "path": str(tmp_path / "ckpt-300"), "metric": 0.4},
        ]

        # Create directories
        for ckpt in callback._checkpoints:
            Path(ckpt["path"]).mkdir(exist_ok=True)

        callback._cleanup_old_checkpoints()

        # Should keep 2 best (lowest loss)
        assert len(callback._checkpoints) == 2

    @patch("boto3.client")
    def test_s3_upload(self, mock_boto, tmp_path):
        """Test S3 upload functionality."""
        from src.training.callbacks import CheckpointCallback

        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        callback = CheckpointCallback(
            checkpoint_dir=tmp_path,
            s3_bucket="test-bucket",
            s3_prefix="checkpoints",
        )

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = callback._upload_to_s3(test_file, "test-key")

        assert result is True
        mock_s3.upload_file.assert_called()


class TestEarlyStoppingCallback:
    """Tests for early stopping callback."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        from src.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=5,
            threshold=0.001,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        assert callback.patience == 5
        assert callback.threshold == 0.001
        assert callback.greater_is_better is False

    def test_early_stopping_trigger(self):
        """Test early stopping triggers after patience."""
        from src.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=2,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # First eval - sets baseline
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
        assert control.should_training_stop is False

        # Second eval - no improvement
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.6})
        assert control.should_training_stop is False

        # Third eval - no improvement, should trigger
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.7})
        assert control.should_training_stop is True

    def test_early_stopping_reset_on_improvement(self):
        """Test patience resets on improvement."""
        from src.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=2,
            metric_for_best="eval_loss",
            greater_is_better=False,
        )

        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        control.should_training_stop = False

        # First eval - sets baseline
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})

        # Second eval - no improvement
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.6})
        assert callback._patience_counter == 1

        # Third eval - improvement!
        callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.3})
        assert callback._patience_counter == 0
        assert callback._best_metric == 0.3
