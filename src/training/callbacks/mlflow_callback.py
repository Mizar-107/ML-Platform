"""MLflow callback for HuggingFace Trainer.

Provides MLflow integration for experiment tracking during training.
"""

import os
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from src.common.logging import get_logger

logger = get_logger(__name__)


class MLflowCallback(TrainerCallback):
    """Callback for MLflow experiment tracking.

    Logs training metrics, hyperparameters, and model artifacts to MLflow.

    Attributes:
        experiment_name: MLflow experiment name
        run_name: MLflow run name (auto-generated if None)
        tracking_uri: MLflow tracking server URI
        log_model: Whether to log model to MLflow
        model_name: Name for registered model
        nested: Whether this is a nested run
    """

    def __init__(
        self,
        experiment_name: str = "lora-finetuning",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        log_model: bool = True,
        model_name: str | None = None,
        nested: bool = False,
    ):
        """Initialize MLflow callback.

        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name (auto-generated if None)
            tracking_uri: MLflow tracking server URI (from env if None)
            log_model: Whether to log model to MLflow
            model_name: Name for registered model
            nested: Whether this is a nested run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.log_model = log_model
        self.model_name = model_name
        self.nested = nested

        self._mlflow = None
        self._run = None
        self._initialized = False

    def _setup_mlflow(self) -> None:
        """Initialize MLflow client and run."""
        if self._initialized:
            return

        try:
            import mlflow

            self._mlflow = mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)

            # Set experiment
            mlflow.set_experiment(self.experiment_name)

            logger.info(
                "MLflow initialized",
                experiment=self.experiment_name,
                tracking_uri=self.tracking_uri,
            )
            self._initialized = True

        except ImportError:
            logger.warning("MLflow not installed, skipping tracking")
            self._initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called at the beginning of training.

        Starts MLflow run and logs hyperparameters.
        """
        self._setup_mlflow()

        if not self._mlflow:
            return

        # Start run
        self._run = self._mlflow.start_run(
            run_name=self.run_name,
            nested=self.nested,
        )

        # Log training arguments
        params = {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lr_scheduler_type": args.lr_scheduler_type,
            "optim": args.optim,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
        }

        # Log LoRA config if available
        model = kwargs.get("model")
        if model and hasattr(model, "peft_config"):
            peft_config = model.peft_config.get("default", {})
            if hasattr(peft_config, "r"):
                params["lora_r"] = peft_config.r
                params["lora_alpha"] = peft_config.lora_alpha
                params["lora_dropout"] = peft_config.lora_dropout
                params["lora_target_modules"] = str(peft_config.target_modules)

        self._mlflow.log_params(params)

        logger.info(
            "MLflow run started",
            run_id=self._run.info.run_id,
            params_logged=len(params),
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when logging metrics.

        Logs training metrics to MLflow.
        """
        if not self._mlflow or not logs:
            return

        # Filter out non-numeric values
        metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                # Prefix with train/ or eval/ for clarity
                if key.startswith("eval_"):
                    metrics[f"eval/{key[5:]}"] = value
                elif key in ("loss", "learning_rate", "epoch"):
                    metrics[f"train/{key}"] = value
                else:
                    metrics[key] = value

        if metrics:
            self._mlflow.log_metrics(metrics, step=state.global_step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called after evaluation.

        Logs evaluation metrics to MLflow.
        """
        if not self._mlflow or not metrics:
            return

        eval_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Clean up metric names
                clean_key = key.replace("eval_", "")
                eval_metrics[f"eval/{clean_key}"] = value

        if eval_metrics:
            self._mlflow.log_metrics(eval_metrics, step=state.global_step)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called when saving a checkpoint.

        Logs checkpoint path as artifact.
        """
        if not self._mlflow:
            return

        # Log the checkpoint directory
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )

        if os.path.exists(checkpoint_dir):
            self._mlflow.log_artifact(checkpoint_dir, artifact_path="checkpoints")
            logger.debug(
                "Checkpoint logged to MLflow",
                checkpoint=checkpoint_dir,
                step=state.global_step,
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called at the end of training.

        Logs final model and ends MLflow run.
        """
        if not self._mlflow:
            return

        # Log final metrics
        if state.best_metric is not None:
            self._mlflow.log_metric("best_metric", state.best_metric)

        # Log model if requested
        if self.log_model and model is not None:
            try:
                # Save model to output directory
                output_dir = os.path.join(args.output_dir, "final_model")
                model.save_pretrained(output_dir)

                # Log as artifact
                self._mlflow.log_artifact(output_dir, artifact_path="model")

                logger.info("Model logged to MLflow", output_dir=output_dir)

                # Register model if name provided
                if self.model_name:
                    model_uri = f"runs:/{self._run.info.run_id}/model"
                    self._mlflow.register_model(model_uri, self.model_name)
                    logger.info("Model registered", model_name=self.model_name)

            except Exception as e:
                logger.error("Failed to log model to MLflow", error=str(e))

        # End run
        self._mlflow.end_run()
        logger.info(
            "MLflow run ended",
            run_id=self._run.info.run_id,
            status="FINISHED",
        )

    def on_train_error(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Called when training encounters an error.

        Ends MLflow run with failed status.
        """
        if not self._mlflow or not self._run:
            return

        self._mlflow.end_run(status="FAILED")
        logger.error(
            "MLflow run ended with error",
            run_id=self._run.info.run_id,
            status="FAILED",
        )


class MLflowModelRegistry:
    """Utility class for MLflow Model Registry operations.

    Provides methods for model registration, staging, and retrieval.
    """

    def __init__(self, tracking_uri: str | None = None):
        """Initialize registry client.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        import mlflow

        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
    ) -> str:
        """Register a model from a run.

        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact in run

        Returns:
            Model version string
        """
        import mlflow

        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        return result.version

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Transition model to a new stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )

    def get_latest_version(
        self,
        model_name: str,
        stage: str | None = None,
    ) -> str | None:
        """Get the latest version of a model.

        Args:
            model_name: Registered model name
            stage: Filter by stage (optional)

        Returns:
            Latest version string or None
        """
        stages = [stage] if stage else None
        versions = self.client.get_latest_versions(model_name, stages=stages)
        if versions:
            return versions[0].version
        return None

    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any:
        """Load a model from the registry.

        Args:
            model_name: Registered model name
            stage: Model stage to load

        Returns:
            Loaded model
        """
        import mlflow

        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)
