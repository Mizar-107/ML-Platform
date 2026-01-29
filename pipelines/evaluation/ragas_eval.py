# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""RAGAS evaluation pipeline for RAG system assessment.

This module defines a Kubeflow Pipeline for running RAGAS evaluations
on RAG systems to assess faithfulness, answer relevancy, context
precision, and context recall.
"""

import argparse
from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp.compiler import Compiler

from pipelines.evaluation.components import (
    gate_deployment_component,
    load_evaluation_data_component,
    run_ragas_evaluation_component,
)

# Pipeline constants
PIPELINE_NAME = "ragas-evaluation-pipeline"
PIPELINE_DESCRIPTION = "RAGAS evaluation pipeline for RAG system assessment"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION,
)
def ragas_evaluation_pipeline(
    data_path: str,
    data_format: str = "json",
    metrics: str = '["faithfulness", "answer_relevancy", "context_precision", "context_recall"]',
    llm_model: str = "gpt-4",
    num_samples: int = 100,
    mlflow_experiment: str = "ragas-evaluation",
    mlflow_tracking_uri: str = "",
    min_score_threshold: float = 0.7,
    enable_deployment_gate: bool = True,
):
    """RAGAS evaluation pipeline.

    This pipeline loads evaluation data, runs RAGAS metrics, and optionally
    gates deployment based on evaluation scores.

    Args:
        data_path: Path to evaluation data (S3, local, or HuggingFace).
        data_format: Format of data ("json", "csv", "parquet", "huggingface").
        metrics: JSON list of RAGAS metrics to compute.
        llm_model: LLM model for evaluation (e.g., "gpt-4", "gpt-3.5-turbo").
        num_samples: Number of samples to evaluate (0 for all).
        mlflow_experiment: MLflow experiment name for tracking.
        mlflow_tracking_uri: MLflow tracking server URI.
        min_score_threshold: Minimum score for deployment approval.
        enable_deployment_gate: Whether to run deployment gate.
    """
    # Step 1: Load evaluation data
    load_data_task = load_evaluation_data_component(
        data_path=data_path,
        data_format=data_format,
        num_samples=num_samples,
    )
    load_data_task.set_display_name("Load Evaluation Data")

    # Step 2: Run RAGAS evaluation
    ragas_task = run_ragas_evaluation_component(
        input_dataset=load_data_task.outputs["output_dataset"],
        metrics=metrics,
        llm_model=llm_model,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    ragas_task.set_display_name("Run RAGAS Evaluation")
    ragas_task.after(load_data_task)

    # Step 3: Deployment gate (optional)
    with dsl.Condition(enable_deployment_gate == True, name="deployment-gate"):
        gate_task = gate_deployment_component(
            evaluation_results=ragas_task.outputs["results_artifact"],
            min_score_threshold=min_score_threshold,
            require_all_passed=True,
        )
        gate_task.set_display_name("Deployment Gate")


@dsl.pipeline(
    name="ragas-batch-evaluation-pipeline",
    description="Batch RAGAS evaluation pipeline for multiple datasets",
)
def ragas_batch_evaluation_pipeline(
    data_paths: list,
    data_format: str = "json",
    metrics: str = '["faithfulness", "answer_relevancy", "context_precision"]',
    llm_model: str = "gpt-4",
    mlflow_experiment: str = "ragas-batch-evaluation",
    mlflow_tracking_uri: str = "",
):
    """Batch RAGAS evaluation pipeline for multiple datasets.

    Args:
        data_paths: List of paths to evaluation datasets.
        data_format: Format of data.
        metrics: JSON list of RAGAS metrics.
        llm_model: LLM model for evaluation.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking URI.
    """
    with dsl.ParallelFor(data_paths, name="evaluate-datasets") as data_path:
        # Load data
        load_task = load_evaluation_data_component(
            data_path=data_path,
            data_format=data_format,
            num_samples=0,  # All samples
        )

        # Run evaluation
        eval_task = run_ragas_evaluation_component(
            input_dataset=load_task.outputs["output_dataset"],
            metrics=metrics,
            llm_model=llm_model,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        eval_task.after(load_task)


@dsl.pipeline(
    name="ragas-ab-test-pipeline",
    description="RAGAS A/B test pipeline comparing two models",
)
def ragas_ab_test_pipeline(
    data_path: str,
    data_format: str = "json",
    model_a_answers_path: str = "",
    model_b_answers_path: str = "",
    metrics: str = '["faithfulness", "answer_relevancy", "context_precision", "context_recall"]',
    llm_model: str = "gpt-4",
    mlflow_experiment: str = "ragas-ab-test",
    mlflow_tracking_uri: str = "",
):
    """A/B test pipeline comparing two models using RAGAS metrics.

    Args:
        data_path: Path to base evaluation data (questions, contexts).
        data_format: Format of data.
        model_a_answers_path: Path to Model A's answers.
        model_b_answers_path: Path to Model B's answers.
        metrics: JSON list of RAGAS metrics.
        llm_model: LLM model for evaluation.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking URI.
    """
    # Evaluate Model A
    load_a_task = load_evaluation_data_component(
        data_path=model_a_answers_path,
        data_format=data_format,
        num_samples=0,
    )
    load_a_task.set_display_name("Load Model A Answers")

    eval_a_task = run_ragas_evaluation_component(
        input_dataset=load_a_task.outputs["output_dataset"],
        metrics=metrics,
        llm_model=llm_model,
        mlflow_experiment=f"{mlflow_experiment}-model-a",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    eval_a_task.set_display_name("Evaluate Model A")
    eval_a_task.after(load_a_task)

    # Evaluate Model B
    load_b_task = load_evaluation_data_component(
        data_path=model_b_answers_path,
        data_format=data_format,
        num_samples=0,
    )
    load_b_task.set_display_name("Load Model B Answers")

    eval_b_task = run_ragas_evaluation_component(
        input_dataset=load_b_task.outputs["output_dataset"],
        metrics=metrics,
        llm_model=llm_model,
        mlflow_experiment=f"{mlflow_experiment}-model-b",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    eval_b_task.set_display_name("Evaluate Model B")
    eval_b_task.after(load_b_task)


def compile_pipeline(
    pipeline_type: str = "standard",
    output_dir: str = "./compiled",
) -> str:
    """Compile the RAGAS pipeline to YAML.

    Args:
        pipeline_type: Type of pipeline ("standard", "batch", "ab_test").
        output_dir: Directory to save compiled pipeline.

    Returns:
        Path to compiled pipeline YAML.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_map = {
        "standard": (ragas_evaluation_pipeline, "ragas_evaluation_pipeline.yaml"),
        "batch": (ragas_batch_evaluation_pipeline, "ragas_batch_evaluation_pipeline.yaml"),
        "ab_test": (ragas_ab_test_pipeline, "ragas_ab_test_pipeline.yaml"),
    }

    if pipeline_type not in pipeline_map:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    pipeline_func, filename = pipeline_map[pipeline_type]
    output_file = output_path / filename

    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=str(output_file),
    )

    print(f"Compiled {pipeline_type} pipeline to {output_file}")
    return str(output_file)


def main():
    """CLI entrypoint for compiling RAGAS pipelines."""
    parser = argparse.ArgumentParser(
        description="Compile RAGAS evaluation pipelines"
    )
    parser.add_argument(
        "--type",
        choices=["standard", "batch", "ab_test", "all"],
        default="standard",
        help="Pipeline type to compile",
    )
    parser.add_argument(
        "--output-dir",
        default="./compiled",
        help="Output directory for compiled pipelines",
    )

    args = parser.parse_args()

    if args.type == "all":
        for ptype in ["standard", "batch", "ab_test"]:
            compile_pipeline(ptype, args.output_dir)
    else:
        compile_pipeline(args.type, args.output_dir)


if __name__ == "__main__":
    main()
