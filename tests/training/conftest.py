"""Test fixtures for training module tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.model_max_length = 2048

    def tokenize_fn(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[1, 2, 3, 4, 5] for _ in texts],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in texts],
        }

    tokenizer.return_value = tokenize_fn(["test"])
    tokenizer.__call__ = MagicMock(side_effect=tokenize_fn)
    tokenizer.decode = MagicMock(return_value="decoded text")

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    import torch

    model = MagicMock()
    model.config = MagicMock()
    model.config.hidden_size = 4096
    model.config.num_hidden_layers = 32
    model.dtype = torch.bfloat16

    # Mock parameters
    param = torch.nn.Parameter(torch.randn(10, 10))
    model.parameters = MagicMock(return_value=[param])
    model.named_parameters = MagicMock(return_value=[("layer.weight", param)])

    # Mock PEFT methods
    model.peft_config = {}
    model.save_pretrained = MagicMock()

    return model


@pytest.fixture
def sample_dataset():
    """Create a sample dataset."""
    from datasets import Dataset

    data = {
        "text": [
            "This is a sample text for training.",
            "Another example text for the model.",
            "Third sample in the dataset.",
        ],
        "id": [1, 2, 3],
    }

    return Dataset.from_dict(data)


@pytest.fixture
def lora_config():
    """Create a sample LoRA config."""
    from src.training.configs import LoRAConfig

    return LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


@pytest.fixture
def qlora_config():
    """Create a sample QLoRA config."""
    from src.training.configs import QLoRAConfig, QuantizationConfig

    return QLoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        quantization=QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        ),
    )


@pytest.fixture
def training_config():
    """Create a sample training config."""
    from src.training.configs import TrainingConfig

    return TrainingConfig(
        output_dir="/tmp/test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
    )


@pytest.fixture
def model_config():
    """Create a sample model config."""
    from src.training.configs import ModelConfig

    return ModelConfig(
        model_name_or_path="gpt2",  # Small model for testing
        max_length=128,
        dtype="float32",
        trust_remote_code=False,
        use_flash_attention_2=False,
    )


@pytest.fixture
def data_config():
    """Create a sample data config."""
    from src.training.configs import DataConfig

    return DataConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        train_split="train[:100]",
        eval_split="validation[:10]",
        text_column="text",
        max_samples=10,
    )


@pytest.fixture
def full_training_config(model_config, lora_config, data_config, training_config):
    """Create a full training config."""
    from src.training.configs import FullTrainingConfig

    return FullTrainingConfig(
        model=model_config,
        lora=lora_config,
        data=data_config,
        training=training_config,
        mlflow_experiment_name="test-experiment",
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    test_dir = tmp_path / "training_test"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing without actual tracking server."""
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run") as mock_start, \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.end_run"), \
         patch("mlflow.register_model"):

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start.return_value.__exit__ = MagicMock(return_value=False)

        yield mock_start


@pytest.fixture
def mock_s3():
    """Mock boto3 S3 client."""
    with patch("boto3.client") as mock_client:
        s3_mock = MagicMock()
        mock_client.return_value = s3_mock

        # Mock paginator
        paginator_mock = MagicMock()
        s3_mock.get_paginator.return_value = paginator_mock
        paginator_mock.paginate.return_value = iter([
            {"Contents": [{"Key": "test/checkpoint-100"}]}
        ])

        yield s3_mock
