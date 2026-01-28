"""Kubeflow Pipeline definitions for model deployment.

This module provides end-to-end pipelines for deploying
and managing LLM models with KServe.
"""

import argparse
from pathlib import Path

from kfp import dsl
from kfp.compiler import Compiler

from pipelines.serving.components import (
    deploy_model_component,
    validate_deployment_component,
    promote_model_component,
    rollback_deployment_component,
)


@dsl.pipeline(
    name="model-deployment-pipeline",
    description="Deploy and validate an LLM model with KServe",
)
def model_deployment_pipeline(
    model_name: str = "mistral-7b-instruct",
    model_uri: str = "hf://mistralai/Mistral-7B-Instruct-v0.2",
    namespace: str = "serving",
    runtime: str = "kserve-vllm",
    min_replicas: int = 1,
    max_replicas: int = 3,
    gpu_count: int = 1,
    memory_limit: str = "32Gi",
    cpu_limit: str = "8",
    max_model_len: int = 8192,
    test_prompt: str = "Write a short poem about machine learning.",
    expected_min_tokens: int = 20,
    validation_timeout: int = 120,
):
    """End-to-end model deployment pipeline.

    Deploys a model, validates it with a smoke test, and reports metrics.

    Args:
        model_name: Name for the InferenceService
        model_uri: Model storage URI (s3:// or hf://)
        namespace: Kubernetes namespace
        runtime: Serving runtime (e.g., kserve-vllm)
        min_replicas: Minimum replicas
        max_replicas: Maximum replicas
        gpu_count: GPUs per replica
        memory_limit: Memory limit
        cpu_limit: CPU limit
        max_model_len: Maximum model sequence length
        test_prompt: Test prompt for validation
        expected_min_tokens: Minimum tokens expected in response
        validation_timeout: Timeout for validation request
    """
    # Deploy model
    deploy_task = deploy_model_component(
        model_name=model_name,
        model_uri=model_uri,
        namespace=namespace,
        runtime=runtime,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        gpu_count=gpu_count,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        max_model_len=max_model_len,
    )
    deploy_task.set_display_name("Deploy Model")
    
    # GPU resource request for deploy task
    deploy_task.set_cpu_request("500m")
    deploy_task.set_memory_request("512Mi")

    # Validate deployment
    validate_task = validate_deployment_component(
        service_url=deploy_task.output,
        test_prompt=test_prompt,
        expected_min_tokens=expected_min_tokens,
        timeout_seconds=validation_timeout,
    )
    validate_task.set_display_name("Validate Deployment")
    validate_task.set_cpu_request("200m")
    validate_task.set_memory_request("256Mi")


@dsl.pipeline(
    name="canary-deployment-pipeline",
    description="Canary deployment with gradual traffic shift",
)
def canary_deployment_pipeline(
    model_name: str = "mistral-7b-instruct",
    model_uri: str = "hf://mistralai/Mistral-7B-Instruct-v0.3",
    previous_model_uri: str = "hf://mistralai/Mistral-7B-Instruct-v0.2",
    namespace: str = "serving",
    runtime: str = "kserve-vllm",
    min_replicas: int = 1,
    max_replicas: int = 3,
    gpu_count: int = 1,
    memory_limit: str = "32Gi",
    cpu_limit: str = "8",
    max_model_len: int = 8192,
    test_prompt: str = "Explain quantum computing in simple terms.",
    expected_min_tokens: int = 30,
    validation_timeout: int = 120,
    canary_traffic_percent: int = 20,
    promote_after_validation: bool = True,
):
    """Canary deployment pipeline with gradual rollout.

    Deploys a new model version as canary, validates, then optionally
    promotes to full traffic or rolls back on failure.

    Args:
        model_name: Name for the InferenceService
        model_uri: New model storage URI
        previous_model_uri: Previous model URI for rollback
        namespace: Kubernetes namespace
        runtime: Serving runtime
        min_replicas: Minimum replicas
        max_replicas: Maximum replicas
        gpu_count: GPUs per replica
        memory_limit: Memory limit
        cpu_limit: CPU limit
        max_model_len: Maximum model sequence length
        test_prompt: Test prompt for validation
        expected_min_tokens: Minimum tokens expected
        validation_timeout: Timeout for validation
        canary_traffic_percent: Initial canary traffic percentage
        promote_after_validation: Auto-promote if validation passes
    """
    # Deploy canary
    deploy_task = deploy_model_component(
        model_name=model_name,
        model_uri=model_uri,
        namespace=namespace,
        runtime=runtime,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        gpu_count=gpu_count,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
        max_model_len=max_model_len,
    )
    deploy_task.set_display_name("Deploy Canary")

    # Set initial canary traffic
    canary_task = promote_model_component(
        model_name=model_name,
        namespace=namespace,
        traffic_percent=canary_traffic_percent,
        deployment_info=deploy_task.outputs["deployment_info"],
    )
    canary_task.set_display_name(f"Set Canary Traffic ({canary_traffic_percent}%)")

    # Validate canary
    validate_task = validate_deployment_component(
        service_url=canary_task.output,
        test_prompt=test_prompt,
        expected_min_tokens=expected_min_tokens,
        timeout_seconds=validation_timeout,
    )
    validate_task.set_display_name("Validate Canary")

    # Conditional promotion or rollback
    with dsl.If(validate_task.output == True):
        with dsl.If(promote_after_validation == True):
            # Promote to 100%
            promote_task = promote_model_component(
                model_name=model_name,
                namespace=namespace,
                traffic_percent=100,
                deployment_info=deploy_task.outputs["deployment_info"],
            )
            promote_task.set_display_name("Promote to Production")

    with dsl.If(validate_task.output == False):
        # Rollback
        rollback_task = rollback_deployment_component(
            model_name=model_name,
            namespace=namespace,
            previous_model_uri=previous_model_uri,
        )
        rollback_task.set_display_name("Rollback")


@dsl.pipeline(
    name="multi-model-deployment-pipeline",
    description="Deploy multiple models in parallel",
)
def multi_model_deployment_pipeline(
    models: list = [
        {"name": "mistral-7b", "uri": "hf://mistralai/Mistral-7B-Instruct-v0.2"},
        {"name": "llama-7b", "uri": "hf://meta-llama/Llama-2-7b-chat-hf"},
    ],
    namespace: str = "serving",
    runtime: str = "kserve-vllm",
):
    """Deploy multiple models in parallel.

    Args:
        models: List of model configs with name and uri
        namespace: Kubernetes namespace
        runtime: Serving runtime
    """
    with dsl.ParallelFor(models) as model:
        deploy_task = deploy_model_component(
            model_name=model.name,
            model_uri=model.uri,
            namespace=namespace,
            runtime=runtime,
            min_replicas=1,
            max_replicas=3,
            gpu_count=1,
            memory_limit="32Gi",
            cpu_limit="8",
            max_model_len=4096,
        )


def compile_pipelines(output_dir: str) -> dict[str, str]:
    """Compile all pipelines to YAML.

    Args:
        output_dir: Output directory for compiled pipelines

    Returns:
        Dictionary mapping pipeline names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipelines = {
        "model_deployment": model_deployment_pipeline,
        "canary_deployment": canary_deployment_pipeline,
        "multi_model_deployment": multi_model_deployment_pipeline,
    }

    compiled = {}
    for name, pipeline in pipelines.items():
        output_file = output_path / f"{name}_pipeline.yaml"
        Compiler().compile(
            pipeline_func=pipeline,
            package_path=str(output_file),
        )
        compiled[name] = str(output_file)
        print(f"Compiled {name} -> {output_file}")

    return compiled


def main():
    """CLI entrypoint for pipeline compilation."""
    parser = argparse.ArgumentParser(description="Compile serving pipelines")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./compiled_pipelines",
        help="Output directory for compiled pipelines",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["deployment", "canary", "multi", "all"],
        default="all",
        help="Pipeline type to compile",
    )

    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.type == "all":
        compile_pipelines(args.output_dir)
    elif args.type == "deployment":
        output_file = output_path / "model_deployment_pipeline.yaml"
        Compiler().compile(model_deployment_pipeline, str(output_file))
        print(f"Compiled -> {output_file}")
    elif args.type == "canary":
        output_file = output_path / "canary_deployment_pipeline.yaml"
        Compiler().compile(canary_deployment_pipeline, str(output_file))
        print(f"Compiled -> {output_file}")
    elif args.type == "multi":
        output_file = output_path / "multi_model_deployment_pipeline.yaml"
        Compiler().compile(multi_model_deployment_pipeline, str(output_file))
        print(f"Compiled -> {output_file}")


if __name__ == "__main__":
    main()
