"""Distributed training utilities.

This module provides utilities for multi-GPU distributed training
using DeepSpeed ZeRO optimization.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

from src.common.logging import get_logger
from src.training.configs import DeepSpeedConfig, create_zero3_config

logger = get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        world_size: Total number of processes
        rank: Global rank of this process
        local_rank: Local rank on this node
        master_addr: Address of the master node
        master_port: Port of the master node
        backend: Distributed backend (nccl, gloo)
    """

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Create config from environment variables.

        Returns:
            DistributedConfig from environment
        """
        return cls(
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(os.getenv("RANK", "0")),
            local_rank=int(os.getenv("LOCAL_RANK", "0")),
            master_addr=os.getenv("MASTER_ADDR", "localhost"),
            master_port=os.getenv("MASTER_PORT", "29500"),
            backend=os.getenv("DISTRIBUTED_BACKEND", "nccl"),
        )

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1


def setup_distributed(config: DistributedConfig | None = None) -> DistributedConfig:
    """Initialize distributed training.

    Args:
        config: Distributed configuration (from env if None)

    Returns:
        Distributed configuration
    """
    if config is None:
        config = DistributedConfig.from_env()

    if not config.is_distributed:
        logger.info("Running in single-GPU mode")
        return config

    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            rank=config.rank,
            world_size=config.world_size,
        )

    # Set device
    torch.cuda.set_device(config.local_rank)

    logger.info(
        "Distributed training initialized",
        world_size=config.world_size,
        rank=config.rank,
        local_rank=config.local_rank,
    )

    return config


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleanup complete")


class DistributedTrainer:
    """Wrapper for distributed training with DeepSpeed.

    Provides utilities for setting up and running distributed
    training with HuggingFace Trainer and DeepSpeed.

    Attributes:
        deepspeed_config: DeepSpeed configuration
        dist_config: Distributed configuration
    """

    def __init__(
        self,
        deepspeed_config: DeepSpeedConfig | str | None = None,
        offload_optimizer: bool = False,
        offload_param: bool = False,
    ):
        """Initialize distributed trainer.

        Args:
            deepspeed_config: DeepSpeed config, path, or None for default
            offload_optimizer: Offload optimizer to CPU (if creating config)
            offload_param: Offload parameters to CPU (if creating config)
        """
        # Setup distributed
        self.dist_config = setup_distributed()

        # Setup DeepSpeed config
        if deepspeed_config is None:
            self.deepspeed_config = create_zero3_config(
                offload_optimizer=offload_optimizer,
                offload_param=offload_param,
            )
        elif isinstance(deepspeed_config, str):
            self.deepspeed_config = DeepSpeedConfig.from_json(deepspeed_config)
        else:
            self.deepspeed_config = deepspeed_config

        self._temp_config_path: Path | None = None

    def get_training_args_kwargs(self) -> dict[str, Any]:
        """Get kwargs to add to TrainingArguments.

        Returns:
            Dictionary of training argument kwargs
        """
        kwargs = {
            "local_rank": self.dist_config.local_rank,
        }

        # Write DeepSpeed config to temp file
        if self.deepspeed_config:
            self._temp_config_path = Path("/tmp/deepspeed_config.json")
            self.deepspeed_config.save(self._temp_config_path)
            kwargs["deepspeed"] = str(self._temp_config_path)

        return kwargs

    def modify_training_args(
        self,
        args: TrainingArguments,
    ) -> TrainingArguments:
        """Modify training arguments for distributed training.

        Args:
            args: Original training arguments

        Returns:
            Modified training arguments
        """
        # Get new kwargs
        kwargs = self.get_training_args_kwargs()

        # Create new args with modifications
        args_dict = args.to_dict()
        args_dict.update(kwargs)

        return TrainingArguments(**args_dict)

    def wrap_trainer(self, trainer: Trainer) -> Trainer:
        """Wrap a trainer for distributed training.

        This modifies the trainer's arguments for distributed
        training with DeepSpeed.

        Args:
            trainer: HuggingFace Trainer instance

        Returns:
            Modified trainer
        """
        # Modify training arguments
        trainer.args = self.modify_training_args(trainer.args)

        logger.info(
            "Trainer wrapped for distributed training",
            deepspeed=trainer.args.deepspeed is not None,
            local_rank=self.dist_config.local_rank,
        )

        return trainer

    def cleanup(self) -> None:
        """Cleanup distributed training resources."""
        # Remove temp config
        if self._temp_config_path and self._temp_config_path.exists():
            self._temp_config_path.unlink()

        # Cleanup distributed
        cleanup_distributed()


def get_gradient_accumulation_steps(
    target_batch_size: int,
    per_device_batch_size: int,
    world_size: int = 1,
) -> int:
    """Calculate gradient accumulation steps for target batch size.

    Args:
        target_batch_size: Desired effective batch size
        per_device_batch_size: Batch size per GPU
        world_size: Number of GPUs

    Returns:
        Number of gradient accumulation steps
    """
    effective_batch = per_device_batch_size * world_size
    accumulation_steps = target_batch_size // effective_batch

    if accumulation_steps < 1:
        logger.warning(
            "Effective batch size larger than target",
            effective=effective_batch,
            target=target_batch_size,
        )
        accumulation_steps = 1

    logger.info(
        "Gradient accumulation calculated",
        target_batch_size=target_batch_size,
        per_device_batch_size=per_device_batch_size,
        world_size=world_size,
        accumulation_steps=accumulation_steps,
    )

    return accumulation_steps


def estimate_memory_requirements(
    model_size_b: float,
    sequence_length: int,
    batch_size: int,
    precision: str = "bf16",
    use_lora: bool = True,
    lora_rank: int = 64,
    zero_stage: int = 3,
) -> dict[str, float]:
    """Estimate GPU memory requirements for training.

    Args:
        model_size_b: Model size in billions of parameters
        sequence_length: Maximum sequence length
        batch_size: Per-device batch size
        precision: Training precision (bf16, fp16, fp32)
        use_lora: Whether using LoRA (reduces requirements)
        lora_rank: LoRA rank if using LoRA
        zero_stage: DeepSpeed ZeRO stage

    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per parameter
    bytes_per_param = {
        "bf16": 2,
        "fp16": 2,
        "fp32": 4,
    }[precision]

    # Model parameters in billions
    params = model_size_b * 1e9

    # Base model memory
    model_memory_gb = (params * bytes_per_param) / (1024**3)

    # Optimizer states (AdamW: 2 states per param)
    # ZeRO-3 partitions these across GPUs
    optimizer_memory_gb = (params * 8) / (1024**3)  # 4 bytes per state

    # Gradient memory
    gradient_memory_gb = (params * bytes_per_param) / (1024**3)

    # Activation memory (rough estimate)
    activation_memory_gb = (
        batch_size * sequence_length * 4096 * 32 * bytes_per_param
    ) / (1024**3)

    # Adjust for ZeRO
    if zero_stage == 3:
        model_memory_gb /= 8  # Assume 8 GPUs
        optimizer_memory_gb /= 8
        gradient_memory_gb /= 8

    # Adjust for LoRA
    if use_lora:
        # Only train ~1% of parameters
        lora_factor = (lora_rank * 2 * 4096 * 32) / params
        gradient_memory_gb *= lora_factor
        optimizer_memory_gb *= lora_factor

    total_gb = (
        model_memory_gb
        + optimizer_memory_gb
        + gradient_memory_gb
        + activation_memory_gb
    )

    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "optimizer_memory_gb": round(optimizer_memory_gb, 2),
        "gradient_memory_gb": round(gradient_memory_gb, 2),
        "activation_memory_gb": round(activation_memory_gb, 2),
        "total_memory_gb": round(total_gb, 2),
    }


def sync_across_processes(data: Any) -> Any:
    """Synchronize data across all processes.

    Only works in distributed mode. Returns data as-is in single GPU mode.

    Args:
        data: Data to synchronize (must be a tensor)

    Returns:
        Synchronized data
    """
    if not dist.is_initialized():
        return data

    if isinstance(data, torch.Tensor):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        data = data / dist.get_world_size()
    else:
        # Convert to tensor, sync, convert back
        tensor = torch.tensor(data, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        data = (tensor / dist.get_world_size()).item()

    return data


def barrier() -> None:
    """Synchronization barrier across processes."""
    if dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """Check if this is the main process.

    Returns:
        True if main process or not distributed
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes.

    Returns:
        World size (1 if not distributed)
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of this process.

    Returns:
        Rank (0 if not distributed)
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
