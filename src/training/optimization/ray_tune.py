"""Ray Tune integration for hyperparameter optimization.

Provides utilities for running hyperparameter search using Ray Tune
with integration to MLflow for experiment tracking.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchSpace:
    """Hyperparameter search space definition.

    Attributes:
        lora_r: LoRA rank search space
        lora_alpha: LoRA alpha search space
        learning_rate: Learning rate search space
        batch_size: Per-device batch size options
        gradient_accumulation: Gradient accumulation options
        warmup_ratio: Warmup ratio search space
        weight_decay: Weight decay search space
    """

    lora_r: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    lora_alpha: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    learning_rate: tuple[float, float] = (1e-5, 5e-4)
    batch_size: list[int] = field(default_factory=lambda: [2, 4, 8])
    gradient_accumulation: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    warmup_ratio: tuple[float, float] = (0.01, 0.1)
    weight_decay: tuple[float, float] = (0.0, 0.1)

    def to_ray_config(self) -> dict[str, Any]:
        """Convert to Ray Tune search space config.

        Returns:
            Dictionary compatible with Ray Tune
        """
        from ray import tune

        return {
            "lora_r": tune.choice(self.lora_r),
            "lora_alpha": tune.choice(self.lora_alpha),
            "learning_rate": tune.loguniform(*self.learning_rate),
            "batch_size": tune.choice(self.batch_size),
            "gradient_accumulation": tune.choice(self.gradient_accumulation),
            "warmup_ratio": tune.uniform(*self.warmup_ratio),
            "weight_decay": tune.uniform(*self.weight_decay),
        }


@dataclass
class TuneConfig:
    """Configuration for Ray Tune hyperparameter search.

    Attributes:
        num_samples: Number of hyperparameter configurations to try
        max_concurrent_trials: Maximum concurrent trials
        metric: Metric to optimize
        mode: Optimization mode ("min" or "max")
        scheduler: Scheduler type ("asha", "pbt", "none")
        grace_period: Minimum iterations before early stopping
        reduction_factor: Reduction factor for ASHA
        resources_per_trial: Resources allocated per trial
    """

    num_samples: int = 10
    max_concurrent_trials: int = 2
    metric: str = "eval_loss"
    mode: str = "min"
    scheduler: str = "asha"
    grace_period: int = 1
    reduction_factor: int = 4
    resources_per_trial: dict = field(
        default_factory=lambda: {"cpu": 4, "gpu": 1}
    )


class RayTuneTrainer:
    """Hyperparameter optimization trainer using Ray Tune.

    Orchestrates hyperparameter search with ASHA scheduler
    and MLflow integration for tracking.

    Attributes:
        base_config: Base training configuration
        search_space: Hyperparameter search space
        tune_config: Ray Tune configuration
    """

    def __init__(
        self,
        base_config: dict[str, Any],
        search_space: SearchSpace | None = None,
        tune_config: TuneConfig | None = None,
    ):
        """Initialize Ray Tune trainer.

        Args:
            base_config: Base training configuration dictionary
            search_space: Hyperparameter search space
            tune_config: Tuning configuration
        """
        self.base_config = base_config
        self.search_space = search_space or SearchSpace()
        self.tune_config = tune_config or TuneConfig()

        self._ray = None
        self._tune = None

    def _setup_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        try:
            import ray
            from ray import tune

            self._ray = ray
            self._tune = tune

            if not ray.is_initialized():
                ray.init(
                    address=os.getenv("RAY_ADDRESS"),
                    ignore_reinit_error=True,
                )
                logger.info("Ray initialized")

        except ImportError as e:
            raise ImportError("Ray is required for HPO. Install with: pip install 'ray[tune]'") from e

    def _get_scheduler(self):
        """Get the trial scheduler based on config.

        Returns:
            Ray Tune scheduler instance
        """
        from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

        if self.tune_config.scheduler == "asha":
            return ASHAScheduler(
                metric=self.tune_config.metric,
                mode=self.tune_config.mode,
                grace_period=self.tune_config.grace_period,
                reduction_factor=self.tune_config.reduction_factor,
            )
        elif self.tune_config.scheduler == "pbt":
            return PopulationBasedTraining(
                metric=self.tune_config.metric,
                mode=self.tune_config.mode,
                perturbation_interval=1,
                hyperparam_mutations={
                    "learning_rate": lambda: self._tune.loguniform(1e-5, 5e-4).sample(),
                    "weight_decay": lambda: self._tune.uniform(0, 0.1).sample(),
                },
            )
        else:
            return None

    def _create_trainable(self) -> Callable:
        """Create the trainable function for Ray Tune.

        Returns:
            Trainable function that takes config and returns metrics
        """
        base_config = self.base_config

        def train_fn(config: dict[str, Any]) -> dict[str, float]:
            """Training function for a single trial.

            Args:
                config: Hyperparameter configuration

            Returns:
                Dictionary of metrics
            """
            from ray import train

            # Merge base config with trial config
            trial_config = base_config.copy()
            trial_config["lora"]["r"] = config["lora_r"]
            trial_config["lora"]["lora_alpha"] = config["lora_alpha"]
            trial_config["training"]["learning_rate"] = config["learning_rate"]
            trial_config["training"]["per_device_train_batch_size"] = config["batch_size"]
            trial_config["training"]["gradient_accumulation_steps"] = config["gradient_accumulation"]
            trial_config["training"]["warmup_ratio"] = config["warmup_ratio"]
            trial_config["training"]["weight_decay"] = config["weight_decay"]

            # Import here to avoid circular deps
            from src.training.configs import FullTrainingConfig
            from src.training.trainers import LoRATrainer

            # Create trainer
            full_config = FullTrainingConfig.model_validate(trial_config)
            trainer = LoRATrainer(full_config)

            # Setup
            trainer.load_tokenizer()
            trainer.load_model()
            trainer.apply_lora()

            train_dataset, eval_dataset = trainer.load_dataset()
            train_dataset = trainer.tokenize_dataset(train_dataset)
            if eval_dataset:
                eval_dataset = trainer.tokenize_dataset(eval_dataset)

            trainer.setup_trainer(train_dataset, eval_dataset)

            # Train
            result = trainer.trainer.train()

            # Get final metrics
            metrics = {
                "train_loss": result.training_loss,
                "eval_loss": trainer.trainer.state.best_metric or result.training_loss,
            }

            # Report to Ray Tune
            train.report(metrics)

            return metrics

        return train_fn

    def run(
        self,
        experiment_name: str = "lora-hpo",
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            experiment_name: Name for the experiment
            storage_path: Path for storing results

        Returns:
            Dictionary with best hyperparameters and metrics
        """
        self._setup_ray()

        from ray import tune
        from ray.tune.integration.mlflow import MLflowLoggerCallback

        logger.info(
            "Starting hyperparameter search",
            num_samples=self.tune_config.num_samples,
            metric=self.tune_config.metric,
        )

        # Setup callbacks
        callbacks = []

        # MLflow integration
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=experiment_name,
                save_artifact=True,
            )
        )

        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(
                self._create_trainable(),
                resources=self.tune_config.resources_per_trial,
            ),
            param_space=self.search_space.to_ray_config(),
            tune_config=tune.TuneConfig(
                num_samples=self.tune_config.num_samples,
                max_concurrent_trials=self.tune_config.max_concurrent_trials,
                scheduler=self._get_scheduler(),
            ),
            run_config=self._ray.train.RunConfig(
                name=experiment_name,
                storage_path=storage_path,
                callbacks=callbacks,
            ),
        )

        results = tuner.fit()

        # Get best result
        best_result = results.get_best_result(
            metric=self.tune_config.metric,
            mode=self.tune_config.mode,
        )

        best_config = best_result.config
        best_metrics = best_result.metrics

        logger.info(
            "Hyperparameter search complete",
            best_config=best_config,
            best_metrics=best_metrics,
        )

        return {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "all_results": results,
        }

    def get_best_config(self, results: dict[str, Any]) -> dict[str, Any]:
        """Extract best configuration from results.

        Args:
            results: Results from run()

        Returns:
            Best hyperparameter configuration
        """
        return results["best_config"]

    def export_best_config(
        self,
        results: dict[str, Any],
        output_path: str,
    ) -> None:
        """Export best configuration to YAML file.

        Args:
            results: Results from run()
            output_path: Output path for YAML file
        """
        import yaml
        from pathlib import Path

        best_config = results["best_config"]

        # Merge with base config
        export_config = self.base_config.copy()
        export_config["lora"]["r"] = best_config["lora_r"]
        export_config["lora"]["lora_alpha"] = best_config["lora_alpha"]
        export_config["training"]["learning_rate"] = best_config["learning_rate"]
        export_config["training"]["per_device_train_batch_size"] = best_config["batch_size"]
        export_config["training"]["gradient_accumulation_steps"] = best_config["gradient_accumulation"]
        export_config["training"]["warmup_ratio"] = best_config["warmup_ratio"]
        export_config["training"]["weight_decay"] = best_config["weight_decay"]

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(export_config, f, default_flow_style=False)

        logger.info("Best configuration exported", path=str(output_path))


def run_hpo_from_config(
    base_config_path: str,
    num_samples: int = 10,
    experiment_name: str = "lora-hpo",
) -> dict[str, Any]:
    """Run HPO from a base configuration file.

    Args:
        base_config_path: Path to base YAML configuration
        num_samples: Number of trials to run
        experiment_name: Name for the experiment

    Returns:
        HPO results with best configuration
    """
    import yaml

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    tune_config = TuneConfig(num_samples=num_samples)
    trainer = RayTuneTrainer(base_config, tune_config=tune_config)

    return trainer.run(experiment_name=experiment_name)


def main() -> None:
    """CLI entrypoint for hyperparameter optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Hyperparameter Optimization")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base training configuration YAML",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of hyperparameter configurations to try",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="lora-hpo",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_config.yaml",
        help="Output path for best configuration",
    )

    args = parser.parse_args()

    results = run_hpo_from_config(
        args.config,
        num_samples=args.num_samples,
        experiment_name=args.experiment_name,
    )

    # Export best config
    import yaml

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    trainer = RayTuneTrainer(base_config)
    trainer.export_best_config(results, args.output)

    print(f"Best configuration saved to: {args.output}")
    print(f"Best metrics: {results['best_metrics']}")


if __name__ == "__main__":
    main()
