"""LoRA and QLoRA configuration classes.

This module provides Pydantic-based configuration models for
LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) fine-tuning.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TargetModules(str, Enum):
    """Common target module patterns for LoRA."""

    ATTENTION_ONLY = "attention"
    ATTENTION_MLP = "attention_mlp"
    ALL_LINEAR = "all_linear"


# Default target modules for common architectures
TARGET_MODULE_PATTERNS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation).

    LoRA reduces the number of trainable parameters by decomposing
    weight updates into low-rank matrices.

    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        lora_alpha: LoRA scaling factor (effective learning rate = alpha/r)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA
        bias: Bias training mode ("none", "all", "lora_only")
        task_type: Task type for PEFT ("CAUSAL_LM", "SEQ_2_SEQ_LM")
        fan_in_fan_out: Set True for GPT-2 style layers
        modules_to_save: Additional modules to make trainable
    """

    r: int = Field(default=64, ge=1, le=512, description="LoRA rank")
    lora_alpha: int = Field(default=128, ge=1, description="LoRA alpha scaling")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: list[str] = Field(
        default_factory=lambda: TARGET_MODULE_PATTERNS["attention_mlp"],
        description="Target modules for LoRA",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none", description="Bias training mode"
    )
    task_type: Literal["CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS", "SEQ_CLS"] = Field(
        default="CAUSAL_LM", description="Task type"
    )
    fan_in_fan_out: bool = Field(default=False, description="GPT-2 style layers")
    modules_to_save: list[str] | None = Field(
        default=None, description="Additional trainable modules"
    )

    @field_validator("target_modules", mode="before")
    @classmethod
    def resolve_target_modules(cls, v: str | list[str]) -> list[str]:
        """Resolve target module patterns to actual module names."""
        if isinstance(v, str):
            if v in TARGET_MODULE_PATTERNS:
                return TARGET_MODULE_PATTERNS[v]
            return [v]
        return v

    def to_peft_config(self) -> dict:
        """Convert to PEFT LoraConfig kwargs.

        Returns:
            Dictionary suitable for peft.LoraConfig initialization
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "fan_in_fan_out": self.fan_in_fan_out,
            "modules_to_save": self.modules_to_save,
        }


class QuantizationConfig(BaseModel):
    """Configuration for model quantization (used with QLoRA).

    Attributes:
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_use_double_quant: Use double quantization
    """

    load_in_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    load_in_8bit: bool = Field(default=False, description="Use 8-bit quantization")
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(
        default="nf4", description="4-bit quantization type"
    )
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16", description="Compute dtype for 4-bit"
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True, description="Use double quantization"
    )

    @field_validator("load_in_8bit", mode="after")
    @classmethod
    def validate_quantization_mode(cls, v: bool, info) -> bool:
        """Ensure only one quantization mode is enabled."""
        if v and info.data.get("load_in_4bit", False):
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        return v

    def to_bnb_config(self) -> dict:
        """Convert to BitsAndBytesConfig kwargs.

        Returns:
            Dictionary suitable for BitsAndBytesConfig initialization
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": dtype_map[self.bnb_4bit_compute_dtype],
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA).

    Extends LoRA with quantization settings for memory-efficient
    fine-tuning of large models.
    """

    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="Quantization configuration",
    )


class ModelConfig(BaseModel):
    """Configuration for the base model.

    Attributes:
        model_name_or_path: HuggingFace model name or local path
        tokenizer_name: Tokenizer name (defaults to model_name_or_path)
        max_length: Maximum sequence length
        dtype: Model dtype for loading
        trust_remote_code: Trust remote code in model
        use_flash_attention_2: Use Flash Attention 2
        attn_implementation: Attention implementation to use
    """

    model_name_or_path: str = Field(
        ..., description="Model name or path"
    )
    tokenizer_name: str | None = Field(
        default=None, description="Tokenizer name (defaults to model)"
    )
    max_length: int = Field(
        default=2048, ge=128, le=32768, description="Maximum sequence length"
    )
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto", description="Model dtype"
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code"
    )
    use_flash_attention_2: bool = Field(
        default=True, description="Use Flash Attention 2"
    )
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = Field(
        default=None, description="Attention implementation"
    )

    def get_tokenizer_name(self) -> str:
        """Get the tokenizer name, defaulting to model name."""
        return self.tokenizer_name or self.model_name_or_path

    def get_torch_dtype(self):
        """Get the torch dtype for model loading."""
        import torch

        if self.dtype == "auto":
            return "auto"

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]


class DataConfig(BaseModel):
    """Configuration for training data.

    Attributes:
        dataset_name: HuggingFace dataset name or S3 path
        dataset_config: Dataset configuration name
        train_split: Training split name
        eval_split: Evaluation split name
        text_column: Column containing text data
        max_samples: Maximum samples to use (for debugging)
        preprocessing_num_workers: Number of preprocessing workers
    """

    dataset_name: str = Field(..., description="Dataset name or path")
    dataset_config: str | None = Field(default=None, description="Dataset config")
    train_split: str = Field(default="train", description="Training split")
    eval_split: str | None = Field(default="validation", description="Eval split")
    text_column: str = Field(default="text", description="Text column name")
    max_samples: int | None = Field(default=None, description="Max samples")
    preprocessing_num_workers: int = Field(
        default=4, ge=1, description="Preprocessing workers"
    )


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters.

    Attributes:
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        per_device_eval_batch_size: Eval batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        warmup_ratio: Warmup ratio of total steps
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum checkpoints to keep
        bf16: Use bfloat16 training
        fp16: Use float16 training
        gradient_checkpointing: Enable gradient checkpointing
        optim: Optimizer to use
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        deepspeed: DeepSpeed config file path
        ddp_find_unused_parameters: DDP unused parameters
        report_to: Reporting integrations
    """

    output_dir: str = Field(default="./outputs", description="Output directory")
    num_train_epochs: float = Field(default=3.0, ge=0.1, description="Training epochs")
    per_device_train_batch_size: int = Field(
        default=4, ge=1, description="Train batch size"
    )
    per_device_eval_batch_size: int = Field(
        default=4, ge=1, description="Eval batch size"
    )
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Gradient accumulation"
    )
    learning_rate: float = Field(
        default=2e-4, gt=0, le=1.0, description="Learning rate"
    )
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    warmup_ratio: float = Field(
        default=0.03, ge=0, le=1.0, description="Warmup ratio"
    )
    lr_scheduler_type: str = Field(
        default="cosine", description="LR scheduler type"
    )
    logging_steps: int = Field(default=10, ge=1, description="Logging steps")
    eval_steps: int | None = Field(default=100, description="Eval steps")
    save_steps: int = Field(default=100, ge=1, description="Save steps")
    save_total_limit: int = Field(default=3, ge=1, description="Max checkpoints")
    bf16: bool = Field(default=True, description="Use bfloat16")
    fp16: bool = Field(default=False, description="Use float16")
    gradient_checkpointing: bool = Field(
        default=True, description="Gradient checkpointing"
    )
    optim: str = Field(default="paged_adamw_32bit", description="Optimizer")
    max_grad_norm: float = Field(default=0.3, ge=0, description="Max grad norm")
    seed: int = Field(default=42, description="Random seed")
    deepspeed: str | None = Field(default=None, description="DeepSpeed config path")
    ddp_find_unused_parameters: bool = Field(
        default=False, description="DDP unused params"
    )
    report_to: list[str] = Field(
        default_factory=lambda: ["mlflow"], description="Report integrations"
    )

    def to_training_arguments(self) -> dict:
        """Convert to TrainingArguments kwargs.

        Returns:
            Dictionary suitable for TrainingArguments initialization
        """
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "deepspeed": self.deepspeed,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "report_to": self.report_to,
        }


class FullTrainingConfig(BaseModel):
    """Complete training configuration combining all components.

    This is the top-level configuration loaded from YAML files.
    """

    model: ModelConfig
    lora: LoRAConfig | QLoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # MLflow settings
    mlflow_experiment_name: str = Field(
        default="lora-finetuning", description="MLflow experiment name"
    )
    mlflow_run_name: str | None = Field(
        default=None, description="MLflow run name"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "FullTrainingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            FullTrainingConfig instance
        """
        import yaml
        from pathlib import Path

        config_path = Path(path)
        with config_path.open() as f:
            config_dict = yaml.safe_load(f)

        return cls.model_validate(config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        import yaml
        from pathlib import Path

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
