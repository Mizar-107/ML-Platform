"""Kubeflow Pipeline components for training.

Provides lightweight component wrappers for training operations.
"""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

# Training image with all dependencies
TRAINING_IMAGE = "your-registry.com/training:latest"


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["boto3", "pyyaml"],
)
def prepare_dataset_component(
    dataset_name: str,
    output_dataset: Output[Dataset],
    dataset_config: str = "",
    max_samples: int = 0,
    s3_bucket: str = "",
    text_column: str = "text",
) -> int:
    """Prepare training dataset.

    Downloads dataset from HuggingFace Hub or S3 and prepares it for training.

    Args:
        dataset_name: HuggingFace dataset name or S3 path
        output_dataset: Output dataset artifact
        dataset_config: Optional dataset configuration
        max_samples: Maximum samples to use (0 for all)
        s3_bucket: S3 bucket for output
        text_column: Name of text column

    Returns:
        Number of samples in dataset
    """
    import json
    from pathlib import Path

    from datasets import load_dataset

    # Load dataset
    if dataset_name.startswith("s3://"):
        import boto3

        # Parse S3 path
        parts = dataset_name[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client("s3")
        local_path = "/tmp/dataset.json"
        s3.download_file(bucket, key, local_path)

        dataset = load_dataset("json", data_files=local_path, split="train")
    else:
        config = dataset_config if dataset_config else None
        dataset = load_dataset(dataset_name, config, split="train")

    # Limit samples if specified
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Validate text column exists
    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found. Available: {dataset.column_names}")

    # Save dataset
    output_path = Path(output_dataset.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    dataset.to_json(str(output_path))

    # Set metadata
    output_dataset.metadata["num_samples"] = len(dataset)
    output_dataset.metadata["columns"] = dataset.column_names

    return len(dataset)


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["torch", "transformers", "peft", "bitsandbytes", "accelerate"],
)
def train_lora_component(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    model_name: str = "meta-llama/Llama-2-7b-hf",
    lora_r: int = 64,
    lora_alpha: int = 128,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    max_length: int = 2048,
    use_qlora: bool = False,
    text_column: str = "text",
) -> str:
    """Train LoRA/QLoRA model.

    Fine-tunes a language model using LoRA or QLoRA.

    Args:
        input_dataset: Training dataset
        output_model: Output model artifact
        output_metrics: Training metrics
        model_name: Base model name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Learning rate
        num_epochs: Number of epochs
        batch_size: Batch size per device
        gradient_accumulation: Gradient accumulation steps
        max_length: Maximum sequence length
        use_qlora: Use QLoRA (4-bit quantization)
        text_column: Text column in dataset

    Returns:
        Path to saved model
    """
    import json
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset("json", data_files=input_dataset.path, split="train")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Training arguments
    output_dir = "/tmp/training_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_32bit",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    train_result = trainer.train()

    # Save model
    model_path = Path(output_model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    # Save metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        "train_samples": len(tokenized_dataset),
    }

    for key, value in metrics.items():
        output_metrics.log_metric(key, value)

    # Set model metadata
    output_model.metadata["base_model"] = model_name
    output_model.metadata["lora_r"] = lora_r
    output_model.metadata["lora_alpha"] = lora_alpha
    output_model.metadata["use_qlora"] = use_qlora

    return str(model_path)


@dsl.component(
    base_image=TRAINING_IMAGE,
    packages_to_install=["torch", "transformers", "peft", "evaluate"],
)
def evaluate_model_component(
    input_model: Input[Model],
    eval_dataset: Input[Dataset],
    output_metrics: Output[Metrics],
    text_column: str = "text",
    max_samples: int = 1000,
) -> float:
    """Evaluate trained model.

    Calculates perplexity and other metrics on evaluation dataset.

    Args:
        input_model: Trained model artifact
        eval_dataset: Evaluation dataset
        output_metrics: Output metrics
        text_column: Text column in dataset
        max_samples: Maximum samples to evaluate

    Returns:
        Perplexity score
    """
    import math
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    model_path = Path(input_model.path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Check if this is a PEFT model
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        import json

        with open(adapter_config) as f:
            config = json.load(f)

        base_model_name = config.get("base_model_name_or_path")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()

    # Load eval dataset
    dataset = load_dataset("json", data_files=eval_dataset.path, split="train")
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Calculate perplexity
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(
                example[text_column],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Log metrics
    output_metrics.log_metric("perplexity", perplexity)
    output_metrics.log_metric("avg_loss", avg_loss)
    output_metrics.log_metric("eval_samples", len(dataset))

    return perplexity


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["mlflow", "boto3"],
)
def register_model_component(
    input_model: Input[Model],
    input_metrics: Input[Metrics],
    model_name: str,
    tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    perplexity_threshold: float = 50.0,
) -> str:
    """Register model in MLflow if metrics pass threshold.

    Args:
        input_model: Trained model artifact
        input_metrics: Training metrics
        model_name: Name for registered model
        tracking_uri: MLflow tracking server URI
        perplexity_threshold: Maximum perplexity to register

    Returns:
        Model version or empty string if not registered
    """
    import os
    import shutil

    import mlflow

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Check perplexity threshold
    perplexity = input_metrics.metadata.get("perplexity", 0)
    if perplexity > perplexity_threshold:
        print(f"Perplexity {perplexity} exceeds threshold {perplexity_threshold}")
        return ""

    # Start run
    with mlflow.start_run() as run:
        # Log metrics
        for key, value in input_metrics.metadata.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log model params
        for key, value in input_model.metadata.items():
            mlflow.log_param(key, value)

        # Log model artifact
        mlflow.log_artifacts(input_model.path, artifact_path="model")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri, model_name)

        print(f"Model registered: {model_name} version {result.version}")
        return result.version

    return ""
