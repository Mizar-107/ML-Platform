"""LoRA Fine-tuning Kubeflow Pipeline.

End-to-end pipeline for LoRA/QLoRA fine-tuning of language models.
"""

from kfp import dsl
from kfp.dsl import If, PipelineTask

from pipelines.training.components import (
    evaluate_model_component,
    prepare_dataset_component,
    register_model_component,
    train_lora_component,
)


@dsl.pipeline(
    name="lora-finetuning-pipeline",
    description="End-to-end LoRA fine-tuning pipeline for LLMs",
)
def lora_finetuning_pipeline(
    # Dataset parameters
    dataset_name: str = "tatsu-lab/alpaca",
    dataset_config: str = "",
    max_samples: int = 0,
    text_column: str = "text",
    # Model parameters
    model_name: str = "meta-llama/Llama-2-7b-hf",
    use_qlora: bool = False,
    # LoRA parameters
    lora_r: int = 64,
    lora_alpha: int = 128,
    # Training parameters
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    max_length: int = 2048,
    # Evaluation parameters
    eval_max_samples: int = 1000,
    perplexity_threshold: float = 50.0,
    # MLflow parameters
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    registered_model_name: str = "lora-finetuned-llm",
) -> None:
    """LoRA fine-tuning pipeline.

    This pipeline:
    1. Prepares the training dataset
    2. Fine-tunes the model using LoRA/QLoRA
    3. Evaluates the trained model
    4. Registers the model if evaluation passes

    Args:
        dataset_name: HuggingFace dataset name or S3 path
        dataset_config: Dataset configuration
        max_samples: Max training samples (0 for all)
        text_column: Name of text column
        model_name: Base model name
        use_qlora: Use QLoRA (4-bit quantization)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Learning rate
        num_epochs: Training epochs
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        max_length: Maximum sequence length
        eval_max_samples: Max evaluation samples
        perplexity_threshold: Max perplexity to register
        mlflow_tracking_uri: MLflow tracking server
        registered_model_name: Name for registered model
    """
    # Step 1: Prepare dataset
    prepare_task = prepare_dataset_component(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_samples=max_samples,
        text_column=text_column,
    )
    prepare_task.set_display_name("Prepare Dataset")
    prepare_task.set_caching_options(enable_caching=True)

    # Step 2: Train model
    train_task = train_lora_component(
        input_dataset=prepare_task.outputs["output_dataset"],
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        max_length=max_length,
        use_qlora=use_qlora,
        text_column=text_column,
    )
    train_task.set_display_name("Train LoRA Model")
    train_task.after(prepare_task)

    # Request GPU
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)

    # Set resource limits
    train_task.set_cpu_limit("8")
    train_task.set_memory_limit("64Gi")

    # Step 3: Evaluate model
    eval_task = evaluate_model_component(
        input_model=train_task.outputs["output_model"],
        eval_dataset=prepare_task.outputs["output_dataset"],
        text_column=text_column,
        max_samples=eval_max_samples,
    )
    eval_task.set_display_name("Evaluate Model")
    eval_task.after(train_task)

    # Request GPU for evaluation
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)

    # Step 4: Register model
    register_task = register_model_component(
        input_model=train_task.outputs["output_model"],
        input_metrics=eval_task.outputs["output_metrics"],
        model_name=registered_model_name,
        tracking_uri=mlflow_tracking_uri,
        perplexity_threshold=perplexity_threshold,
    )
    register_task.set_display_name("Register Model")
    register_task.after(eval_task)


@dsl.pipeline(
    name="qlora-finetuning-pipeline",
    description="Memory-efficient QLoRA fine-tuning pipeline",
)
def qlora_finetuning_pipeline(
    dataset_name: str = "tatsu-lab/alpaca",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    lora_r: int = 64,
    lora_alpha: int = 16,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 8,  # Higher batch size possible with QLoRA
    gradient_accumulation: int = 2,
    max_length: int = 2048,
    registered_model_name: str = "qlora-finetuned-llm",
) -> None:
    """QLoRA fine-tuning pipeline.

    Optimized for memory-efficient training with 4-bit quantization.

    Args:
        dataset_name: Training dataset
        model_name: Base model name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (lower for QLoRA)
        learning_rate: Learning rate
        num_epochs: Training epochs
        batch_size: Per-device batch size (can be higher)
        gradient_accumulation: Gradient accumulation steps
        max_length: Maximum sequence length
        registered_model_name: Name for registered model
    """
    # Use the main pipeline with QLoRA enabled
    prepare_task = prepare_dataset_component(
        dataset_name=dataset_name,
    )
    prepare_task.set_display_name("Prepare Dataset")

    train_task = train_lora_component(
        input_dataset=prepare_task.outputs["output_dataset"],
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        max_length=max_length,
        use_qlora=True,
    )
    train_task.set_display_name("Train QLoRA Model")
    train_task.after(prepare_task)
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)
    # Lower memory requirements with QLoRA
    train_task.set_memory_limit("32Gi")

    eval_task = evaluate_model_component(
        input_model=train_task.outputs["output_model"],
        eval_dataset=prepare_task.outputs["output_dataset"],
    )
    eval_task.set_display_name("Evaluate Model")
    eval_task.after(train_task)
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(1)

    register_task = register_model_component(
        input_model=train_task.outputs["output_model"],
        input_metrics=eval_task.outputs["output_metrics"],
        model_name=registered_model_name,
    )
    register_task.set_display_name("Register Model")
    register_task.after(eval_task)


def compile_pipeline(
    output_path: str = "lora_finetuning_pipeline.yaml",
    pipeline_type: str = "lora",
) -> None:
    """Compile the pipeline to YAML.

    Args:
        output_path: Output path for compiled pipeline
        pipeline_type: Type of pipeline ("lora" or "qlora")
    """
    from kfp import compiler

    if pipeline_type == "qlora":
        pipeline_func = qlora_finetuning_pipeline
        if output_path == "lora_finetuning_pipeline.yaml":
            output_path = "qlora_finetuning_pipeline.yaml"
    else:
        pipeline_func = lora_finetuning_pipeline

    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile training pipelines")
    parser.add_argument(
        "--type",
        choices=["lora", "qlora", "all"],
        default="all",
        help="Pipeline type to compile",
    )
    parser.add_argument(
        "--output-dir",
        default="./compiled",
        help="Output directory for compiled pipelines",
    )

    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    if args.type in ["lora", "all"]:
        compile_pipeline(
            f"{args.output_dir}/lora_finetuning_pipeline.yaml",
            "lora",
        )

    if args.type in ["qlora", "all"]:
        compile_pipeline(
            f"{args.output_dir}/qlora_finetuning_pipeline.yaml",
            "qlora",
        )
