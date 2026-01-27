"""Training module for LLM fine-tuning.

This module provides components for LoRA/QLoRA fine-tuning of large
language models with distributed training support.

Components:
- configs: Configuration classes for LoRA, QLoRA, and DeepSpeed
- callbacks: MLflow tracking and checkpoint management callbacks
- trainers: LoRA trainer and distributed training utilities
- optimization: Ray Tune hyperparameter optimization
"""

from src.training.configs import (
    LoRAConfig,
    QLoRAConfig,
    QuantizationConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    FullTrainingConfig,
    DeepSpeedConfig,
    ZeROConfig,
    create_zero2_config,
    create_zero3_config,
)
from src.training.callbacks import (
    MLflowCallback,
    MLflowModelRegistry,
    CheckpointCallback,
    EarlyStoppingCallback,
)
from src.training.trainers import (
    LoRATrainer,
    train_from_config,
    DistributedTrainer,
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
)
from src.training.optimization import (
    RayTuneTrainer,
    SearchSpace,
    TuneConfig,
    run_hpo_from_config,
)

__all__ = [
    # Configs
    "LoRAConfig",
    "QLoRAConfig",
    "QuantizationConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "FullTrainingConfig",
    "DeepSpeedConfig",
    "ZeROConfig",
    "create_zero2_config",
    "create_zero3_config",
    # Callbacks
    "MLflowCallback",
    "MLflowModelRegistry",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    # Trainers
    "LoRATrainer",
    "train_from_config",
    "DistributedTrainer",
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    # Optimization
    "RayTuneTrainer",
    "SearchSpace",
    "TuneConfig",
    "run_hpo_from_config",
]
