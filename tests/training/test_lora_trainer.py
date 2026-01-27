"""Tests for LoRA trainer module."""

import pytest
from unittest.mock import MagicMock, patch


class TestLoRAConfig:
    """Tests for LoRA configuration classes."""

    def test_lora_config_defaults(self):
        """Test LoRA config default values."""
        from src.training.configs import LoRAConfig

        config = LoRAConfig()

        assert config.r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

    def test_lora_config_custom_values(self):
        """Test LoRA config with custom values."""
        from src.training.configs import LoRAConfig

        config = LoRAConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj"],
        )

        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.target_modules == ["q_proj", "k_proj"]

    def test_lora_config_target_module_resolution(self):
        """Test target module pattern resolution."""
        from src.training.configs import LoRAConfig

        # Test string pattern
        config = LoRAConfig(target_modules="attention")
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

        # Test llama pattern
        config = LoRAConfig(target_modules="llama")
        assert "gate_proj" in config.target_modules

    def test_lora_config_to_peft(self):
        """Test conversion to PEFT config kwargs."""
        from src.training.configs import LoRAConfig

        config = LoRAConfig(r=16, lora_alpha=32)
        peft_kwargs = config.to_peft_config()

        assert peft_kwargs["r"] == 16
        assert peft_kwargs["lora_alpha"] == 32
        assert peft_kwargs["task_type"] == "CAUSAL_LM"

    def test_lora_config_validation(self):
        """Test config validation."""
        from src.training.configs import LoRAConfig
        from pydantic import ValidationError

        # Invalid rank
        with pytest.raises(ValidationError):
            LoRAConfig(r=0)

        # Invalid dropout
        with pytest.raises(ValidationError):
            LoRAConfig(lora_dropout=1.5)


class TestQLoRAConfig:
    """Tests for QLoRA configuration."""

    def test_qlora_config_defaults(self):
        """Test QLoRA config default values."""
        from src.training.configs import QLoRAConfig

        config = QLoRAConfig()

        assert config.quantization.load_in_4bit is True
        assert config.quantization.bnb_4bit_quant_type == "nf4"
        assert config.quantization.bnb_4bit_use_double_quant is True

    def test_qlora_quantization_config(self):
        """Test quantization config conversion."""
        from src.training.configs import QLoRAConfig

        config = QLoRAConfig()
        bnb_kwargs = config.quantization.to_bnb_config()

        assert bnb_kwargs["load_in_4bit"] is True
        assert bnb_kwargs["bnb_4bit_quant_type"] == "nf4"

    def test_qlora_mutex_quantization(self):
        """Test that 4-bit and 8-bit are mutually exclusive."""
        from src.training.configs import QuantizationConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QuantizationConfig(load_in_4bit=True, load_in_8bit=True)


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_training_config_defaults(self):
        """Test training config defaults."""
        from src.training.configs import TrainingConfig

        config = TrainingConfig()

        assert config.num_train_epochs == 3.0
        assert config.learning_rate == 2e-4
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_training_config_to_args(self):
        """Test conversion to training arguments."""
        from src.training.configs import TrainingConfig

        config = TrainingConfig(
            output_dir="/tmp/test",
            num_train_epochs=2,
            learning_rate=1e-4,
        )

        args_kwargs = config.to_training_arguments()

        assert args_kwargs["output_dir"] == "/tmp/test"
        assert args_kwargs["num_train_epochs"] == 2
        assert args_kwargs["learning_rate"] == 1e-4


class TestFullTrainingConfig:
    """Tests for full training configuration."""

    def test_full_config_creation(self, model_config, data_config):
        """Test creating full config."""
        from src.training.configs import FullTrainingConfig, LoRAConfig, TrainingConfig

        config = FullTrainingConfig(
            model=model_config,
            lora=LoRAConfig(),
            data=data_config,
            training=TrainingConfig(),
        )

        assert config.model.model_name_or_path == "gpt2"
        assert config.lora.r == 64

    def test_full_config_from_yaml(self, tmp_path):
        """Test loading config from YAML."""
        from src.training.configs import FullTrainingConfig

        # Create test YAML
        yaml_content = """
model:
  model_name_or_path: "gpt2"
  max_length: 128
  dtype: "float32"
  trust_remote_code: false
  use_flash_attention_2: false

lora:
  r: 16
  lora_alpha: 32
  target_modules:
    - q_proj
    - v_proj

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
  train_split: "train"
  text_column: "text"

training:
  output_dir: "./outputs"
  num_train_epochs: 1
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = FullTrainingConfig.from_yaml(str(config_path))

        assert config.model.model_name_or_path == "gpt2"
        assert config.lora.r == 16
        assert config.lora.lora_alpha == 32

    def test_full_config_to_yaml(self, full_training_config, tmp_path):
        """Test saving config to YAML."""
        output_path = tmp_path / "exported.yaml"

        full_training_config.to_yaml(str(output_path))

        assert output_path.exists()

        # Reload and verify
        from src.training.configs import FullTrainingConfig

        reloaded = FullTrainingConfig.from_yaml(str(output_path))
        assert reloaded.model.model_name_or_path == full_training_config.model.model_name_or_path


class TestLoRATrainer:
    """Tests for LoRA trainer class."""

    def test_trainer_initialization(self, full_training_config):
        """Test trainer initialization."""
        from src.training.trainers import LoRATrainer

        trainer = LoRATrainer(full_training_config)

        assert trainer.config == full_training_config
        assert trainer.model is None
        assert trainer.tokenizer is None

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_tokenizer(self, mock_from_pretrained, full_training_config, mock_tokenizer):
        """Test tokenizer loading."""
        from src.training.trainers import LoRATrainer

        mock_from_pretrained.return_value = mock_tokenizer

        trainer = LoRATrainer(full_training_config)
        tokenizer = trainer.load_tokenizer()

        assert tokenizer is not None
        mock_from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_model(self, mock_tokenizer, mock_model, full_training_config):
        """Test model loading."""
        from src.training.trainers import LoRATrainer

        mock_model.return_value = MagicMock()

        trainer = LoRATrainer(full_training_config)
        model = trainer.load_model()

        assert model is not None
        mock_model.assert_called_once()

    @patch("peft.get_peft_model")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_apply_lora(self, mock_load_model, mock_get_peft, full_training_config):
        """Test LoRA application."""
        from src.training.trainers import LoRATrainer

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        peft_model = MagicMock()
        peft_model.parameters.return_value = iter([MagicMock(numel=lambda: 100, requires_grad=True)])
        mock_get_peft.return_value = peft_model

        trainer = LoRATrainer(full_training_config)
        trainer.load_model()
        trainer.apply_lora()

        mock_get_peft.assert_called_once()


class TestDeepSpeedConfig:
    """Tests for DeepSpeed configuration."""

    def test_create_zero3_config(self):
        """Test creating ZeRO-3 config."""
        from src.training.configs import create_zero3_config

        config = create_zero3_config()

        assert config.zero_optimization.stage == 3
        assert config.bf16.enabled is True

    def test_create_zero3_with_offload(self):
        """Test ZeRO-3 with CPU offloading."""
        from src.training.configs import create_zero3_config

        config = create_zero3_config(offload_optimizer=True, offload_param=True)

        assert config.zero_optimization.offload_optimizer is True
        assert config.zero_optimization.offload_param is True

    def test_deepspeed_to_dict(self):
        """Test DeepSpeed config to dictionary."""
        from src.training.configs import create_zero3_config

        config = create_zero3_config()
        config_dict = config.to_dict()

        assert "zero_optimization" in config_dict
        assert config_dict["zero_optimization"]["stage"] == 3
        assert config_dict["bf16"]["enabled"] is True

    def test_deepspeed_save_load(self, tmp_path):
        """Test saving and loading DeepSpeed config."""
        from src.training.configs import DeepSpeedConfig, create_zero3_config

        config = create_zero3_config()
        config_path = tmp_path / "ds_config.json"

        config.save(config_path)
        assert config_path.exists()

        loaded = DeepSpeedConfig.from_json(config_path)
        assert loaded.zero_optimization.stage == 3


class TestDistributedTrainer:
    """Tests for distributed training utilities."""

    def test_distributed_config_from_env(self, monkeypatch):
        """Test creating config from environment."""
        from src.training.trainers import DistributedConfig

        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")

        config = DistributedConfig.from_env()

        assert config.world_size == 4
        assert config.rank == 1
        assert config.local_rank == 1

    def test_is_main_process(self):
        """Test main process detection."""
        from src.training.trainers import DistributedConfig

        config = DistributedConfig(rank=0, world_size=4)
        assert config.is_main_process is True

        config = DistributedConfig(rank=1, world_size=4)
        assert config.is_main_process is False

    def test_is_distributed(self):
        """Test distributed detection."""
        from src.training.trainers import DistributedConfig

        config = DistributedConfig(world_size=1)
        assert config.is_distributed is False

        config = DistributedConfig(world_size=4)
        assert config.is_distributed is True

    def test_gradient_accumulation_calculation(self):
        """Test gradient accumulation steps calculation."""
        from src.training.trainers import get_gradient_accumulation_steps

        steps = get_gradient_accumulation_steps(
            target_batch_size=64,
            per_device_batch_size=4,
            world_size=4,
        )

        assert steps == 4  # 64 / (4 * 4) = 4

    def test_memory_estimation(self):
        """Test memory requirement estimation."""
        from src.training.trainers import estimate_memory_requirements

        estimate = estimate_memory_requirements(
            model_size_b=7.0,
            sequence_length=2048,
            batch_size=4,
            precision="bf16",
            use_lora=True,
            zero_stage=3,
        )

        assert "total_memory_gb" in estimate
        assert "model_memory_gb" in estimate
        assert estimate["total_memory_gb"] > 0
