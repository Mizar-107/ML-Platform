# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""DeepEval evaluation pipeline for LLM quality assessment.

This module defines a Kubeflow Pipeline for running DeepEval evaluations
to assess LLM outputs for hallucination, toxicity, bias, and quality.
"""

import argparse
from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp.compiler import Compiler

from pipelines.evaluation.components import (
    aggregate_results_component,
    gate_deployment_component,
    load_evaluation_data_component,
    run_deepeval_evaluation_component,
    run_ragas_evaluation_component,
)

# Pipeline constants
PIPELINE_NAME = "deepeval-evaluation-pipeline"
PIPELINE_DESCRIPTION = "DeepEval evaluation pipeline for LLM quality assessment"


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION,
)
def deepeval_evaluation_pipeline(
    data_path: str,
    data_format: str = "json",
    metrics: str = '["hallucination", "toxicity", "coherence", "relevance"]',
    model: str = "gpt-4",
    threshold: float = 0.5,
    num_samples: int = 100,
    mlflow_experiment: str = "deepeval-evaluation",
    mlflow_tracking_uri: str = "",
    min_pass_rate: float = 0.8,
    enable_deployment_gate: bool = True,
):
    """DeepEval evaluation pipeline.

    This pipeline loads test cases, runs DeepEval metrics, and optionally
    gates deployment based on pass rate.

    Args:
        data_path: Path to test case data (S3, local).
        data_format: Format of data ("json", "csv", "parquet").
        metrics: JSON list of DeepEval metrics to compute.
        model: LLM model for evaluation.
        threshold: Pass/fail threshold for metrics.
        num_samples: Number of samples to evaluate (0 for all).
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking server URI.
        min_pass_rate: Minimum pass rate for deployment approval.
        enable_deployment_gate: Whether to run deployment gate.
    """
    # Step 1: Load test case data
    load_data_task = load_evaluation_data_component(
        data_path=data_path,
        data_format=data_format,
        num_samples=num_samples,
    )
    load_data_task.set_display_name("Load Test Cases")

    # Step 2: Run DeepEval evaluation
    deepeval_task = run_deepeval_evaluation_component(
        input_dataset=load_data_task.outputs["output_dataset"],
        metrics=metrics,
        model=model,
        threshold=threshold,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    deepeval_task.set_display_name("Run DeepEval Evaluation")
    deepeval_task.after(load_data_task)

    # Step 3: Deployment gate (optional)
    with dsl.Condition(enable_deployment_gate == True, name="deployment-gate"):
        gate_task = gate_deployment_component(
            evaluation_results=deepeval_task.outputs["results_artifact"],
            min_score_threshold=min_pass_rate,
            require_all_passed=False,  # Use pass rate instead
        )
        gate_task.set_display_name("Deployment Gate")


@dsl.pipeline(
    name="deepeval-safety-pipeline",
    description="DeepEval safety evaluation pipeline focusing on toxicity and bias",
)
def deepeval_safety_pipeline(
    data_path: str,
    data_format: str = "json",
    model: str = "gpt-4",
    toxicity_threshold: float = 0.1,  # Stricter threshold for safety
    bias_threshold: float = 0.2,
    num_samples: int = 0,
    mlflow_experiment: str = "deepeval-safety",
    mlflow_tracking_uri: str = "",
):
    """Safety-focused DeepEval pipeline.

    This pipeline specifically focuses on safety metrics like
    toxicity and bias with stricter thresholds.

    Args:
        data_path: Path to test case data.
        data_format: Format of data.
        model: LLM model for evaluation.
        toxicity_threshold: Threshold for toxicity (lower is stricter).
        bias_threshold: Threshold for bias (lower is stricter).
        num_samples: Number of samples (0 for all).
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking URI.
    """
    # Load data
    load_task = load_evaluation_data_component(
        data_path=data_path,
        data_format=data_format,
        num_samples=num_samples,
    )
    load_task.set_display_name("Load Safety Test Cases")

    # Run toxicity evaluation
    toxicity_task = run_deepeval_evaluation_component(
        input_dataset=load_task.outputs["output_dataset"],
        metrics='["toxicity"]',
        model=model,
        threshold=toxicity_threshold,
        mlflow_experiment=f"{mlflow_experiment}-toxicity",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    toxicity_task.set_display_name("Evaluate Toxicity")
    toxicity_task.after(load_task)

    # Run bias evaluation
    bias_task = run_deepeval_evaluation_component(
        input_dataset=load_task.outputs["output_dataset"],
        metrics='["bias"]',
        model=model,
        threshold=bias_threshold,
        mlflow_experiment=f"{mlflow_experiment}-bias",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    bias_task.set_display_name("Evaluate Bias")
    bias_task.after(load_task)


@dsl.pipeline(
    name="comprehensive-evaluation-pipeline",
    description="Comprehensive evaluation pipeline combining RAGAS and DeepEval",
)
def comprehensive_evaluation_pipeline(
    data_path: str,
    data_format: str = "json",
    ragas_metrics: str = '["faithfulness", "answer_relevancy", "context_precision"]',
    deepeval_metrics: str = '["hallucination", "toxicity", "coherence"]',
    llm_model: str = "gpt-4",
    num_samples: int = 100,
    mlflow_experiment: str = "comprehensive-evaluation",
    mlflow_tracking_uri: str = "",
    min_overall_score: float = 0.7,
    require_all_passed: bool = True,
):
    """Comprehensive evaluation pipeline combining RAGAS and DeepEval.

    This pipeline runs both RAGAS (for RAG-specific metrics) and DeepEval
    (for general LLM quality metrics) and aggregates the results.

    Args:
        data_path: Path to evaluation data.
        data_format: Format of data.
        ragas_metrics: JSON list of RAGAS metrics.
        deepeval_metrics: JSON list of DeepEval metrics.
        llm_model: LLM model for evaluation.
        num_samples: Number of samples to evaluate.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking URI.
        min_overall_score: Minimum overall score for approval.
        require_all_passed: Whether all metrics must pass.
    """
    # Step 1: Load evaluation data
    load_task = load_evaluation_data_component(
        data_path=data_path,
        data_format=data_format,
        num_samples=num_samples,
    )
    load_task.set_display_name("Load Evaluation Data")

    # Step 2a: Run RAGAS evaluation
    ragas_task = run_ragas_evaluation_component(
        input_dataset=load_task.outputs["output_dataset"],
        metrics=ragas_metrics,
        llm_model=llm_model,
        mlflow_experiment=f"{mlflow_experiment}-ragas",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    ragas_task.set_display_name("RAGAS Evaluation")
    ragas_task.after(load_task)

    # Step 2b: Run DeepEval evaluation (parallel with RAGAS)
    deepeval_task = run_deepeval_evaluation_component(
        input_dataset=load_task.outputs["output_dataset"],
        metrics=deepeval_metrics,
        model=llm_model,
        threshold=0.5,
        mlflow_experiment=f"{mlflow_experiment}-deepeval",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    deepeval_task.set_display_name("DeepEval Evaluation")
    deepeval_task.after(load_task)

    # Step 3: Aggregate results
    aggregate_task = aggregate_results_component(
        ragas_results=ragas_task.outputs["results_artifact"],
        deepeval_results=deepeval_task.outputs["results_artifact"],
    )
    aggregate_task.set_display_name("Aggregate Results")
    aggregate_task.after(ragas_task)
    aggregate_task.after(deepeval_task)

    # Step 4: Deployment gate
    gate_task = gate_deployment_component(
        evaluation_results=aggregate_task.outputs["combined_output"],
        min_score_threshold=min_overall_score,
        require_all_passed=require_all_passed,
    )
    gate_task.set_display_name("Deployment Gate")
    gate_task.after(aggregate_task)


def compile_pipeline(
    pipeline_type: str = "standard",
    output_dir: str = "./compiled",
) -> str:
    """Compile the DeepEval pipeline to YAML.

    Args:
        pipeline_type: Type of pipeline ("standard", "safety", "comprehensive").
        output_dir: Directory to save compiled pipeline.

    Returns:
        Path to compiled pipeline YAML.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_map = {
        "standard": (deepeval_evaluation_pipeline, "deepeval_evaluation_pipeline.yaml"),
        "safety": (deepeval_safety_pipeline, "deepeval_safety_pipeline.yaml"),
        "comprehensive": (comprehensive_evaluation_pipeline, "comprehensive_evaluation_pipeline.yaml"),
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
    """CLI entrypoint for compiling DeepEval pipelines."""
    parser = argparse.ArgumentParser(
        description="Compile DeepEval evaluation pipelines"
    )
    parser.add_argument(
        "--type",
        choices=["standard", "safety", "comprehensive", "all"],
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
        for ptype in ["standard", "safety", "comprehensive"]:
            compile_pipeline(ptype, args.output_dir)
    else:
        compile_pipeline(args.type, args.output_dir)


if __name__ == "__main__":
    main()
