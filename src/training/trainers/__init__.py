"""Training trainers module.

Provides trainer classes for LoRA/QLoRA fine-tuning and distributed training.
"""

from src.training.trainers.lora_trainer import (
    LoRATrainer,
    train_from_config,
)
from src.training.trainers.distributed_trainer import (
    DistributedConfig,
    DistributedTrainer,
    barrier,
    cleanup_distributed,
    estimate_memory_requirements,
    get_gradient_accumulation_steps,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    sync_across_processes,
)

__all__ = [
    # LoRA trainer
    "LoRATrainer",
    "train_from_config",
    # Distributed utilities
    "DistributedConfig",
    "DistributedTrainer",
    "setup_distributed",
    "cleanup_distributed",
    "get_gradient_accumulation_steps",
    "estimate_memory_requirements",
    "sync_across_processes",
    "barrier",
    "is_main_process",
    "get_world_size",
    "get_rank",
]
