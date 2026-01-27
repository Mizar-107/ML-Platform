"""DeepSpeed ZeRO-3 configuration utilities.

This module provides configuration builders and utilities for
DeepSpeed ZeRO Stage 3 distributed training.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class ZeROStage(int, Enum):
    """DeepSpeed ZeRO optimization stages."""

    DISABLED = 0
    OPTIMIZER_STATES = 1  # Partition optimizer states
    GRADIENTS = 2  # + Partition gradients
    PARAMETERS = 3  # + Partition parameters


@dataclass
class ZeROConfig:
    """ZeRO optimization configuration.

    Attributes:
        stage: ZeRO stage (0-3)
        offload_optimizer: Offload optimizer to CPU
        offload_param: Offload parameters to CPU
        overlap_comm: Overlap communication with computation
        contiguous_gradients: Use contiguous gradients
        reduce_bucket_size: Bucket size for all-reduce (bytes)
        stage3_prefetch_bucket_size: Prefetch bucket size (stage 3)
        stage3_param_persistence_threshold: Param persistence threshold
        stage3_max_live_parameters: Max live parameters
        stage3_max_reuse_distance: Max reuse distance
        stage3_gather_16bit_weights_on_model_save: Gather weights on save
    """

    stage: int = 3
    offload_optimizer: bool = False
    offload_param: bool = False
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 5e8
    stage3_prefetch_bucket_size: int = 5e8
    stage3_param_persistence_threshold: int = 1e6
    stage3_max_live_parameters: int = 1e9
    stage3_max_reuse_distance: int = 1e9
    stage3_gather_16bit_weights_on_model_save: bool = True


@dataclass
class OptimizerConfig:
    """DeepSpeed optimizer configuration.

    Attributes:
        type: Optimizer type
        params: Optimizer parameters
    """

    type: str = "AdamW"
    params: dict = field(
        default_factory=lambda: {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        }
    )


@dataclass
class SchedulerConfig:
    """DeepSpeed scheduler configuration.

    Attributes:
        type: Scheduler type
        params: Scheduler parameters
    """

    type: str = "WarmupDecayLR"
    params: dict = field(
        default_factory=lambda: {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto",
        }
    )


@dataclass
class FP16Config:
    """FP16 mixed precision configuration.

    Attributes:
        enabled: Enable FP16
        loss_scale: Loss scale (0 for dynamic)
        loss_scale_window: Window for dynamic loss scaling
        initial_scale_power: Initial scale as 2^power
        hysteresis: Hysteresis for dynamic loss scaling
        min_loss_scale: Minimum loss scale
    """

    enabled: bool = False
    loss_scale: float = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: float = 1


@dataclass
class BF16Config:
    """BF16 mixed precision configuration.

    Attributes:
        enabled: Enable BF16
    """

    enabled: bool = True


@dataclass
class GradientClippingConfig:
    """Gradient clipping configuration.

    Attributes:
        enabled: Enable gradient clipping
        value: Clipping value (max norm)
    """

    enabled: bool = True
    value: float = 1.0


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing configuration.

    Attributes:
        partition_activations: Partition activations across GPUs
        cpu_checkpointing: Checkpoint to CPU
        contiguous_memory_optimization: Use contiguous memory
        number_checkpoints: Number of checkpoints
        synchronize_checkpoint_boundary: Synchronize at boundary
        profile: Profile checkpointing
    """

    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: int | None = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


@dataclass
class DeepSpeedConfig:
    """Complete DeepSpeed configuration.

    Attributes:
        train_batch_size: Global training batch size
        train_micro_batch_size_per_gpu: Micro batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        gradient_clipping: Gradient clipping setting
        zero_optimization: ZeRO configuration
        fp16: FP16 configuration
        bf16: BF16 configuration
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        activation_checkpointing: Activation checkpointing config
        wall_clock_breakdown: Enable timing breakdown
        steps_per_print: Steps between status prints
    """

    train_batch_size: int | str = "auto"
    train_micro_batch_size_per_gpu: int | str = "auto"
    gradient_accumulation_steps: int | str = "auto"
    gradient_clipping: float = 1.0
    zero_optimization: ZeROConfig = field(default_factory=ZeROConfig)
    fp16: FP16Config = field(default_factory=FP16Config)
    bf16: BF16Config = field(default_factory=BF16Config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    activation_checkpointing: ActivationCheckpointingConfig | None = None
    wall_clock_breakdown: bool = False
    steps_per_print: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_optimization.stage,
                "offload_optimizer": {
                    "device": "cpu" if self.zero_optimization.offload_optimizer else "none",
                    "pin_memory": True,
                }
                if self.zero_optimization.offload_optimizer
                else {"device": "none"},
                "offload_param": {
                    "device": "cpu" if self.zero_optimization.offload_param else "none",
                    "pin_memory": True,
                }
                if self.zero_optimization.offload_param
                else {"device": "none"},
                "overlap_comm": self.zero_optimization.overlap_comm,
                "contiguous_gradients": self.zero_optimization.contiguous_gradients,
                "reduce_bucket_size": int(self.zero_optimization.reduce_bucket_size),
                "stage3_prefetch_bucket_size": int(
                    self.zero_optimization.stage3_prefetch_bucket_size
                ),
                "stage3_param_persistence_threshold": int(
                    self.zero_optimization.stage3_param_persistence_threshold
                ),
                "stage3_max_live_parameters": int(
                    self.zero_optimization.stage3_max_live_parameters
                ),
                "stage3_max_reuse_distance": int(
                    self.zero_optimization.stage3_max_reuse_distance
                ),
                "stage3_gather_16bit_weights_on_model_save": (
                    self.zero_optimization.stage3_gather_16bit_weights_on_model_save
                ),
            },
            "fp16": {
                "enabled": self.fp16.enabled,
                "loss_scale": self.fp16.loss_scale,
                "loss_scale_window": self.fp16.loss_scale_window,
                "initial_scale_power": self.fp16.initial_scale_power,
                "hysteresis": self.fp16.hysteresis,
                "min_loss_scale": self.fp16.min_loss_scale,
            },
            "bf16": {"enabled": self.bf16.enabled},
            "optimizer": {
                "type": self.optimizer.type,
                "params": self.optimizer.params,
            },
            "scheduler": {
                "type": self.scheduler.type,
                "params": self.scheduler.params,
            },
            "wall_clock_breakdown": self.wall_clock_breakdown,
            "steps_per_print": self.steps_per_print,
        }

        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": self.activation_checkpointing.partition_activations,
                "cpu_checkpointing": self.activation_checkpointing.cpu_checkpointing,
                "contiguous_memory_optimization": (
                    self.activation_checkpointing.contiguous_memory_optimization
                ),
                "synchronize_checkpoint_boundary": (
                    self.activation_checkpointing.synchronize_checkpoint_boundary
                ),
                "profile": self.activation_checkpointing.profile,
            }
            if self.activation_checkpointing.number_checkpoints:
                config["activation_checkpointing"]["number_checkpoints"] = (
                    self.activation_checkpointing.number_checkpoints
                )

        return config

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSpeedConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            DeepSpeedConfig instance
        """
        path = Path(path)
        config_dict = json.loads(path.read_text())
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DeepSpeedConfig":
        """Create configuration from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            DeepSpeedConfig instance
        """
        zero_config = config.get("zero_optimization", {})
        zero = ZeROConfig(
            stage=zero_config.get("stage", 3),
            offload_optimizer=zero_config.get("offload_optimizer", {}).get("device", "none")
            == "cpu",
            offload_param=zero_config.get("offload_param", {}).get("device", "none") == "cpu",
            overlap_comm=zero_config.get("overlap_comm", True),
            contiguous_gradients=zero_config.get("contiguous_gradients", True),
            reduce_bucket_size=zero_config.get("reduce_bucket_size", 5e8),
            stage3_prefetch_bucket_size=zero_config.get("stage3_prefetch_bucket_size", 5e8),
            stage3_param_persistence_threshold=zero_config.get(
                "stage3_param_persistence_threshold", 1e6
            ),
            stage3_max_live_parameters=zero_config.get("stage3_max_live_parameters", 1e9),
            stage3_max_reuse_distance=zero_config.get("stage3_max_reuse_distance", 1e9),
            stage3_gather_16bit_weights_on_model_save=zero_config.get(
                "stage3_gather_16bit_weights_on_model_save", True
            ),
        )

        fp16_config = config.get("fp16", {})
        fp16 = FP16Config(
            enabled=fp16_config.get("enabled", False),
            loss_scale=fp16_config.get("loss_scale", 0),
            loss_scale_window=fp16_config.get("loss_scale_window", 1000),
            initial_scale_power=fp16_config.get("initial_scale_power", 16),
            hysteresis=fp16_config.get("hysteresis", 2),
            min_loss_scale=fp16_config.get("min_loss_scale", 1),
        )

        bf16_config = config.get("bf16", {})
        bf16 = BF16Config(enabled=bf16_config.get("enabled", True))

        opt_config = config.get("optimizer", {})
        optimizer = OptimizerConfig(
            type=opt_config.get("type", "AdamW"),
            params=opt_config.get("params", {}),
        )

        sched_config = config.get("scheduler", {})
        scheduler = SchedulerConfig(
            type=sched_config.get("type", "WarmupDecayLR"),
            params=sched_config.get("params", {}),
        )

        activation_checkpointing = None
        if "activation_checkpointing" in config:
            ac_config = config["activation_checkpointing"]
            activation_checkpointing = ActivationCheckpointingConfig(
                partition_activations=ac_config.get("partition_activations", True),
                cpu_checkpointing=ac_config.get("cpu_checkpointing", False),
                contiguous_memory_optimization=ac_config.get(
                    "contiguous_memory_optimization", True
                ),
                number_checkpoints=ac_config.get("number_checkpoints"),
                synchronize_checkpoint_boundary=ac_config.get(
                    "synchronize_checkpoint_boundary", False
                ),
                profile=ac_config.get("profile", False),
            )

        return cls(
            train_batch_size=config.get("train_batch_size", "auto"),
            train_micro_batch_size_per_gpu=config.get("train_micro_batch_size_per_gpu", "auto"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", "auto"),
            gradient_clipping=config.get("gradient_clipping", 1.0),
            zero_optimization=zero,
            fp16=fp16,
            bf16=bf16,
            optimizer=optimizer,
            scheduler=scheduler,
            activation_checkpointing=activation_checkpointing,
            wall_clock_breakdown=config.get("wall_clock_breakdown", False),
            steps_per_print=config.get("steps_per_print", 100),
        )


def create_zero3_config(
    offload_optimizer: bool = False,
    offload_param: bool = False,
    use_bf16: bool = True,
    gradient_clipping: float = 1.0,
) -> DeepSpeedConfig:
    """Create a standard ZeRO Stage 3 configuration.

    Args:
        offload_optimizer: Offload optimizer states to CPU
        offload_param: Offload parameters to CPU
        use_bf16: Use BF16 mixed precision (else FP16)
        gradient_clipping: Maximum gradient norm

    Returns:
        DeepSpeedConfig with ZeRO Stage 3 settings
    """
    return DeepSpeedConfig(
        zero_optimization=ZeROConfig(
            stage=3,
            offload_optimizer=offload_optimizer,
            offload_param=offload_param,
        ),
        fp16=FP16Config(enabled=not use_bf16),
        bf16=BF16Config(enabled=use_bf16),
        gradient_clipping=gradient_clipping,
    )


def create_zero2_config(
    offload_optimizer: bool = False,
    use_bf16: bool = True,
    gradient_clipping: float = 1.0,
) -> DeepSpeedConfig:
    """Create a ZeRO Stage 2 configuration.

    Args:
        offload_optimizer: Offload optimizer states to CPU
        use_bf16: Use BF16 mixed precision (else FP16)
        gradient_clipping: Maximum gradient norm

    Returns:
        DeepSpeedConfig with ZeRO Stage 2 settings
    """
    return DeepSpeedConfig(
        zero_optimization=ZeROConfig(
            stage=2,
            offload_optimizer=offload_optimizer,
            offload_param=False,  # Not supported in stage 2
        ),
        fp16=FP16Config(enabled=not use_bf16),
        bf16=BF16Config(enabled=use_bf16),
        gradient_clipping=gradient_clipping,
    )
