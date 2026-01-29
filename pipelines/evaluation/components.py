# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""Kubeflow Pipeline components for LLM evaluation.

This module provides KFP v2 components for running RAGAS and DeepEval
evaluations as part of ML pipelines.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output

# Base image for evaluation components
EVAL_IMAGE = "llm-mlops/evaluation:latest"


@dsl.component(
    base_image=EVAL_IMAGE,
    packages_to_install=["datasets", "ragas", "deepeval", "mlflow", "pydantic"],
)
def load_evaluation_data_component(
    data_path: str,
    data_format: str,
    output_dataset: Output[Dataset],
    num_samples: int = 100,
    random_seed: int = 42,
) -> NamedTuple("Outputs", [("num_loaded", int), ("columns", str)]):
    """Load evaluation data from various sources.

    Args:
        data_path: Path to evaluation data (S3, local, or HuggingFace dataset).
        data_format: Format of data ("json", "csv", "parquet", "huggingface").
        output_dataset: Output artifact for loaded dataset.
        num_samples: Number of samples to load (0 for all).
        random_seed: Random seed for sampling.

    Returns:
        Tuple of (num_loaded, columns).
    """
    import json
    import logging

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading evaluation data from {data_path}")

    # Load data based on format
    if data_format == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(data_path, split="test")
        df = dataset.to_pandas()
    elif data_format == "json":
        if data_path.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            bucket, key = data_path.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_json(obj["Body"])
        else:
            df = pd.read_json(data_path)
    elif data_format == "csv":
        df = pd.read_csv(data_path)
    elif data_format == "parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    # Sample if requested
    if num_samples > 0 and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=random_seed)
        logger.info(f"Sampled {num_samples} rows from {len(df)} total")

    # Save to output artifact
    df.to_json(output_dataset.path, orient="records", indent=2)

    logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    outputs = NamedTuple("Outputs", [("num_loaded", int), ("columns", str)])
    return outputs(len(df), json.dumps(list(df.columns)))


@dsl.component(
    base_image=EVAL_IMAGE,
    packages_to_install=["datasets", "ragas", "mlflow", "pydantic", "openai"],
)
def run_ragas_evaluation_component(
    input_dataset: Input[Dataset],
    metrics_output: Output[Metrics],
    results_artifact: Output[Artifact],
    metrics: str = '["faithfulness", "answer_relevancy", "context_precision", "context_recall"]',
    llm_model: str = "gpt-4",
    mlflow_experiment: str = "ragas-evaluation",
    mlflow_tracking_uri: str = "",
) -> NamedTuple("Outputs", [("average_score", float), ("passed", bool)]):
    """Run RAGAS evaluation on input dataset.

    Args:
        input_dataset: Input dataset artifact with evaluation samples.
        metrics_output: Output metrics artifact.
        results_artifact: Output artifact for detailed results.
        metrics: JSON list of RAGAS metrics to compute.
        llm_model: LLM model for evaluation.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Tuple of (average_score, passed).
    """
    import json
    import logging
    import os

    import mlflow
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set MLflow tracking URI
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Load input data
    df = pd.read_json(input_dataset.path)
    logger.info(f"Loaded {len(df)} samples for RAGAS evaluation")

    # Parse metrics
    metric_list = json.loads(metrics)
    logger.info(f"Running RAGAS with metrics: {metric_list}")

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Map metric names to objects
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

        ragas_metrics = [metric_map[m] for m in metric_list if m in metric_map]

        # Prepare dataset
        eval_data = {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": df["contexts"].tolist(),
            "ground_truth": df.get("ground_truth", df["answer"]).tolist(),
        }
        dataset = Dataset.from_dict(eval_data)

        # Run evaluation
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run():
            result = ragas_evaluate(dataset, metrics=ragas_metrics)

            # Extract scores
            scores = {}
            for metric_name in metric_list:
                if metric_name in result:
                    scores[metric_name] = float(result[metric_name])
                    metrics_output.log_metric(f"ragas_{metric_name}", scores[metric_name])
                    mlflow.log_metric(f"ragas_{metric_name}", scores[metric_name])

            average_score = sum(scores.values()) / len(scores) if scores else 0.0
            passed = all(s > 0.5 for s in scores.values())

            metrics_output.log_metric("ragas_average", average_score)
            metrics_output.log_metric("ragas_passed", int(passed))
            mlflow.log_metric("ragas_average", average_score)

            # Save detailed results
            results_data = {
                "scores": scores,
                "average_score": average_score,
                "passed": passed,
                "num_samples": len(df),
            }
            with open(results_artifact.path, "w") as f:
                json.dump(results_data, f, indent=2)

            logger.info(f"RAGAS evaluation complete: avg={average_score:.4f}, passed={passed}")

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        average_score = 0.0
        passed = False
        with open(results_artifact.path, "w") as f:
            json.dump({"error": str(e)}, f)

    outputs = NamedTuple("Outputs", [("average_score", float), ("passed", bool)])
    return outputs(average_score, passed)


@dsl.component(
    base_image=EVAL_IMAGE,
    packages_to_install=["deepeval", "mlflow", "pydantic", "openai"],
)
def run_deepeval_evaluation_component(
    input_dataset: Input[Dataset],
    metrics_output: Output[Metrics],
    results_artifact: Output[Artifact],
    metrics: str = '["hallucination", "toxicity", "coherence", "relevance"]',
    model: str = "gpt-4",
    threshold: float = 0.5,
    mlflow_experiment: str = "deepeval-evaluation",
    mlflow_tracking_uri: str = "",
) -> NamedTuple("Outputs", [("pass_rate", float), ("all_passed", bool)]):
    """Run DeepEval evaluation on input dataset.

    Args:
        input_dataset: Input dataset artifact with test cases.
        metrics_output: Output metrics artifact.
        results_artifact: Output artifact for detailed results.
        metrics: JSON list of DeepEval metrics to compute.
        model: LLM model for evaluation.
        threshold: Pass/fail threshold.
        mlflow_experiment: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Tuple of (pass_rate, all_passed).
    """
    import json
    import logging

    import mlflow
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set MLflow tracking URI
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Load input data
    df = pd.read_json(input_dataset.path)
    logger.info(f"Loaded {len(df)} samples for DeepEval evaluation")

    # Parse metrics
    metric_list = json.loads(metrics)
    logger.info(f"Running DeepEval with metrics: {metric_list}")

    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import (
            BiasMetric,
            CoherenceMetric,
            HallucinationMetric,
            RelevanceMetric,
            ToxicityMetric,
        )
        from deepeval.test_case import LLMTestCase

        # Create metrics
        metric_map = {
            "hallucination": HallucinationMetric(threshold=threshold, model=model),
            "toxicity": ToxicityMetric(threshold=threshold, model=model),
            "coherence": CoherenceMetric(threshold=threshold, model=model),
            "relevance": RelevanceMetric(threshold=threshold, model=model),
            "bias": BiasMetric(threshold=threshold, model=model),
        }

        deepeval_metrics = [metric_map[m] for m in metric_list if m in metric_map]

        # Create test cases
        test_cases = [
            LLMTestCase(
                input=row["input"],
                actual_output=row["actual_output"],
                expected_output=row.get("expected_output"),
                context=row.get("context"),
                retrieval_context=row.get("retrieval_context"),
            )
            for _, row in df.iterrows()
        ]

        # Run evaluation
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run():
            results = deepeval_evaluate(test_cases=test_cases, metrics=deepeval_metrics)

            # Process results
            scores = {}
            passed_counts = {}

            for metric in deepeval_metrics:
                metric_name = metric.__class__.__name__.lower().replace("metric", "")
                if hasattr(metric, "score"):
                    scores[metric_name] = float(metric.score)
                    passed_counts[metric_name] = metric.is_successful()

            # Calculate pass rate
            total_passed = sum(1 for p in passed_counts.values() if p)
            pass_rate = total_passed / len(passed_counts) if passed_counts else 0.0
            all_passed = all(passed_counts.values()) if passed_counts else False

            # Log metrics
            for name, score in scores.items():
                metrics_output.log_metric(f"deepeval_{name}", score)
                mlflow.log_metric(f"deepeval_{name}", score)

            metrics_output.log_metric("deepeval_pass_rate", pass_rate)
            mlflow.log_metric("deepeval_pass_rate", pass_rate)

            # Save detailed results
            results_data = {
                "scores": scores,
                "passed": passed_counts,
                "pass_rate": pass_rate,
                "all_passed": all_passed,
                "num_samples": len(df),
            }
            with open(results_artifact.path, "w") as f:
                json.dump(results_data, f, indent=2)

            logger.info(f"DeepEval evaluation complete: pass_rate={pass_rate:.2%}")

    except Exception as e:
        logger.error(f"DeepEval evaluation failed: {e}")
        pass_rate = 0.0
        all_passed = False
        with open(results_artifact.path, "w") as f:
            json.dump({"error": str(e)}, f)

    outputs = NamedTuple("Outputs", [("pass_rate", float), ("all_passed", bool)])
    return outputs(pass_rate, all_passed)


@dsl.component(
    base_image=EVAL_IMAGE,
    packages_to_install=["pydantic"],
)
def aggregate_results_component(
    ragas_results: Input[Artifact],
    deepeval_results: Input[Artifact],
    combined_output: Output[Artifact],
    metrics_output: Output[Metrics],
) -> NamedTuple("Outputs", [("overall_score", float), ("overall_passed", bool)]):
    """Aggregate results from RAGAS and DeepEval evaluations.

    Args:
        ragas_results: RAGAS evaluation results artifact.
        deepeval_results: DeepEval evaluation results artifact.
        combined_output: Combined results output artifact.
        metrics_output: Output metrics artifact.

    Returns:
        Tuple of (overall_score, overall_passed).
    """
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load results
    with open(ragas_results.path) as f:
        ragas = json.load(f)
    with open(deepeval_results.path) as f:
        deepeval = json.load(f)

    logger.info("Aggregating evaluation results")

    # Combine scores
    combined_scores = {}
    if "scores" in ragas:
        for name, score in ragas["scores"].items():
            combined_scores[f"ragas_{name}"] = score
    if "scores" in deepeval:
        for name, score in deepeval["scores"].items():
            combined_scores[f"deepeval_{name}"] = score

    # Calculate overall metrics
    overall_score = sum(combined_scores.values()) / len(combined_scores) if combined_scores else 0.0
    ragas_passed = ragas.get("passed", True)
    deepeval_passed = deepeval.get("all_passed", True)
    overall_passed = ragas_passed and deepeval_passed

    # Log metrics
    metrics_output.log_metric("overall_score", overall_score)
    metrics_output.log_metric("overall_passed", int(overall_passed))

    # Save combined results
    combined = {
        "ragas": ragas,
        "deepeval": deepeval,
        "combined_scores": combined_scores,
        "overall_score": overall_score,
        "overall_passed": overall_passed,
    }
    with open(combined_output.path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Aggregation complete: overall_score={overall_score:.4f}, passed={overall_passed}")

    outputs = NamedTuple("Outputs", [("overall_score", float), ("overall_passed", bool)])
    return outputs(overall_score, overall_passed)


@dsl.component(
    base_image=EVAL_IMAGE,
)
def gate_deployment_component(
    evaluation_results: Input[Artifact],
    min_score_threshold: float = 0.7,
    require_all_passed: bool = True,
) -> NamedTuple("Outputs", [("approved", bool), ("reason", str)]):
    """Quality gate for model deployment based on evaluation results.

    Args:
        evaluation_results: Combined evaluation results artifact.
        min_score_threshold: Minimum overall score required for approval.
        require_all_passed: Whether all individual metrics must pass.

    Returns:
        Tuple of (approved, reason).
    """
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    with open(evaluation_results.path) as f:
        results = json.load(f)

    overall_score = results.get("overall_score", 0.0)
    overall_passed = results.get("overall_passed", False)

    # Evaluate gate conditions
    score_ok = overall_score >= min_score_threshold
    passed_ok = overall_passed if require_all_passed else True

    approved = score_ok and passed_ok

    if approved:
        reason = f"Approved: score={overall_score:.4f} >= {min_score_threshold}"
    else:
        reasons = []
        if not score_ok:
            reasons.append(f"score {overall_score:.4f} < {min_score_threshold}")
        if not passed_ok:
            reasons.append("not all metrics passed")
        reason = f"Rejected: {', '.join(reasons)}"

    logger.info(f"Deployment gate: {reason}")

    outputs = NamedTuple("Outputs", [("approved", bool), ("reason", str)])
    return outputs(approved, reason)
