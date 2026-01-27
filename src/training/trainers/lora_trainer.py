"""LoRA/QLoRA fine-tuning trainer.

This module provides the main trainer class for LoRA and QLoRA
fine-tuning of large language models.
"""

import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.common.logging import get_logger, setup_logging
from src.training.callbacks import CheckpointCallback, EarlyStoppingCallback, MLflowCallback
from src.training.configs import (
    FullTrainingConfig,
    LoRAConfig as LoRAConfigModel,
    QLoRAConfig,
)

logger = get_logger(__name__)


class LoRATrainer:
    """Trainer for LoRA/QLoRA fine-tuning.

    Handles model loading, LoRA application, and training orchestration.

    Attributes:
        config: Full training configuration
        model: The loaded and prepared model
        tokenizer: The model tokenizer
        trainer: HuggingFace Trainer instance
    """

    def __init__(self, config: FullTrainingConfig):
        """Initialize LoRA trainer.

        Args:
            config: Full training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Setup logging
        setup_logging(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format_type=os.getenv("LOG_FORMAT", "console"),
            service_name="lora-trainer",
        )

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer.

        Returns:
            Configured tokenizer
        """
        logger.info(
            "Loading tokenizer",
            model=self.config.model.get_tokenizer_name(),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.get_tokenizer_name(),
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right",
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        return tokenizer

    def load_model(self) -> AutoModelForCausalLM:
        """Load base model with optional quantization.

        Returns:
            Loaded model (possibly quantized)
        """
        logger.info(
            "Loading model",
            model=self.config.model.model_name_or_path,
        )

        # Build model kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": self.config.model.get_torch_dtype(),
        }

        # Configure attention implementation
        if self.config.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation
        elif self.config.model.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Configure quantization for QLoRA
        if isinstance(self.config.lora, QLoRAConfig):
            logger.info("Using QLoRA quantization")
            bnb_config = BitsAndBytesConfig(**self.config.lora.quantization.to_bnb_config())
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            **model_kwargs,
        )

        # Prepare model for k-bit training if quantized
        if isinstance(self.config.lora, QLoRAConfig):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing,
            )

        logger.info(
            "Model loaded",
            num_params=sum(p.numel() for p in model.parameters()),
            dtype=str(model.dtype),
        )

        self.model = model
        return model

    def apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        if self.model is None:
            raise ValueError("Model must be loaded before applying LoRA")

        logger.info(
            "Applying LoRA",
            r=self.config.lora.r,
            alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
        )

        lora_config = LoraConfig(**self.config.lora.to_peft_config())
        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            "LoRA applied",
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_pct=f"{100 * trainable_params / total_params:.2f}%",
        )

    def load_dataset(self) -> tuple[Dataset, Dataset | None]:
        """Load and prepare training dataset.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info(
            "Loading dataset",
            dataset=self.config.data.dataset_name,
        )

        # Load dataset
        if self.config.data.dataset_name.startswith("s3://"):
            # Load from S3 (assumes JSON format)
            dataset = load_dataset(
                "json",
                data_files=self.config.data.dataset_name,
                split=self.config.data.train_split,
            )
        else:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_config,
                split=self.config.data.train_split,
            )

        # Limit samples if specified
        if self.config.data.max_samples:
            dataset = dataset.select(range(min(self.config.data.max_samples, len(dataset))))

        # Load eval dataset
        eval_dataset = None
        if self.config.data.eval_split:
            try:
                if self.config.data.dataset_name.startswith("s3://"):
                    eval_dataset = None  # TODO: Handle S3 eval split
                else:
                    eval_dataset = load_dataset(
                        self.config.data.dataset_name,
                        self.config.data.dataset_config,
                        split=self.config.data.eval_split,
                    )
                    if self.config.data.max_samples:
                        eval_dataset = eval_dataset.select(
                            range(min(self.config.data.max_samples // 10, len(eval_dataset)))
                        )
            except Exception as e:
                logger.warning(f"Could not load eval split: {e}")

        logger.info(
            "Dataset loaded",
            train_size=len(dataset),
            eval_size=len(eval_dataset) if eval_dataset else 0,
        )

        return dataset, eval_dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """Tokenize dataset for training.

        Args:
            dataset: Raw dataset

        Returns:
            Tokenized dataset
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before tokenizing")

        def tokenize_function(examples: dict[str, list]) -> dict[str, list]:
            """Tokenize a batch of examples."""
            texts = examples[self.config.data.text_column]

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.model.max_length,
                padding=False,
            )

            return tokenized

        logger.info("Tokenizing dataset")

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized_dataset

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ) -> Trainer:
        """Setup HuggingFace Trainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)

        Returns:
            Configured Trainer instance
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before setup")

        logger.info("Setting up trainer")

        # Training arguments
        training_args = TrainingArguments(
            **self.config.training.to_training_arguments(),
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Setup callbacks
        callbacks = []

        # MLflow callback
        callbacks.append(
            MLflowCallback(
                experiment_name=self.config.mlflow_experiment_name,
                run_name=self.config.mlflow_run_name,
                log_model=True,
            )
        )

        # Checkpoint callback
        callbacks.append(
            CheckpointCallback(
                checkpoint_dir=Path(self.config.training.output_dir) / "checkpoints",
                keep_n_checkpoints=self.config.training.save_total_limit,
                metric_for_best="eval_loss",
                greater_is_better=False,
            )
        )

        # Early stopping
        callbacks.append(
            EarlyStoppingCallback(
                patience=3,
                metric_for_best="eval_loss",
                greater_is_better=False,
            )
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        return self.trainer

    def train(self, resume_from_checkpoint: str | bool | None = None) -> dict[str, float]:
        """Run training.

        Args:
            resume_from_checkpoint: Checkpoint path or True to auto-detect

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer must be setup before training")

        logger.info(
            "Starting training",
            epochs=self.config.training.num_train_epochs,
            batch_size=self.config.training.per_device_train_batch_size,
            grad_accum=self.config.training.gradient_accumulation_steps,
        )

        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info(
            "Training complete",
            train_loss=result.training_loss,
            global_step=result.global_step,
        )

        return result.metrics

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the trained model.

        Args:
            output_dir: Output directory (defaults to config output_dir)
        """
        output_dir = output_dir or self.config.training.output_dir

        logger.info("Saving model", output_dir=output_dir)

        # Save adapter only (PEFT)
        self.model.save_pretrained(output_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        logger.info("Model saved", output_dir=output_dir)

    def run(self) -> dict[str, float]:
        """Run the complete training pipeline.

        Returns:
            Training metrics
        """
        # Load components
        self.load_tokenizer()
        self.load_model()
        self.apply_lora()

        # Load and prepare data
        train_dataset, eval_dataset = self.load_dataset()
        train_dataset = self.tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.tokenize_dataset(eval_dataset)

        # Setup and run training
        self.setup_trainer(train_dataset, eval_dataset)
        metrics = self.train()

        # Save model
        self.save_model()

        return metrics


def train_from_config(config_path: str) -> dict[str, float]:
    """Train from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration

    Returns:
        Training metrics
    """
    config = FullTrainingConfig.from_yaml(config_path)
    trainer = LoRATrainer(config)
    return trainer.run()


def main() -> None:
    """CLI entrypoint for training."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA/QLoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    args = parser.parse_args()

    config = FullTrainingConfig.from_yaml(args.config)
    trainer = LoRATrainer(config)

    # Load and prepare
    trainer.load_tokenizer()
    trainer.load_model()
    trainer.apply_lora()

    train_dataset, eval_dataset = trainer.load_dataset()
    train_dataset = trainer.tokenize_dataset(train_dataset)
    if eval_dataset:
        eval_dataset = trainer.tokenize_dataset(eval_dataset)

    trainer.setup_trainer(train_dataset, eval_dataset)

    # Train with optional resume
    metrics = trainer.train(resume_from_checkpoint=args.resume)

    # Save
    trainer.save_model()

    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
