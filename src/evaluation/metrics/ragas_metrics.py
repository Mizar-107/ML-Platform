# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""RAGAS evaluation metrics for RAG system assessment.

This module provides RAGAS-based evaluation metrics for assessing
RAG (Retrieval Augmented Generation) system quality including
faithfulness, answer relevancy, context precision, and context recall.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import mlflow
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RAGASMetricType(str, Enum):
    """Available RAGAS metrics."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_RELEVANCY = "context_relevancy"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


class RAGASConfig(BaseModel):
    """Configuration for RAGAS evaluation."""

    metrics: list[RAGASMetricType] = Field(
        default=[
            RAGASMetricType.FAITHFULNESS,
            RAGASMetricType.ANSWER_RELEVANCY,
            RAGASMetricType.CONTEXT_PRECISION,
            RAGASMetricType.CONTEXT_RECALL,
        ],
        description="Metrics to evaluate",
    )
    llm_model: str = Field(
        default="gpt-4",
        description="LLM model for evaluation",
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="Embedding model for similarity metrics",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for evaluation",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Timeout for evaluation",
    )
    log_to_mlflow: bool = Field(
        default=True,
        description="Log results to MLflow",
    )
    raise_on_error: bool = Field(
        default=False,
        description="Raise exception on evaluation error",
    )

    class Config:
        use_enum_values = True


@dataclass
class RAGASResult:
    """Result from RAGAS evaluation."""

    scores: dict[str, float] = field(default_factory=dict)
    detailed_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def average_score(self) -> float:
        """Calculate average score across all metrics."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def passed(self) -> bool:
        """Check if evaluation passed (all scores > 0.5)."""
        return all(score > 0.5 for score in self.scores.values())

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scores": self.scores,
            "average_score": self.average_score,
            "passed": self.passed,
            "detailed_results": self.detailed_results,
            "metadata": self.metadata,
            "errors": self.errors,
        }


@dataclass
class RAGSample:
    """A single sample for RAGAS evaluation."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


class RAGASEvaluator:
    """Evaluator using RAGAS metrics for RAG systems.

    RAGAS (Retrieval Augmented Generation Assessment) provides metrics
    specifically designed for evaluating RAG pipelines including:
    - Faithfulness: How factually accurate is the answer given the context
    - Answer Relevancy: How relevant is the answer to the question
    - Context Precision: How precise is the retrieved context
    - Context Recall: How much of the ground truth is captured

    Example:
        >>> config = RAGASConfig(metrics=[RAGASMetricType.FAITHFULNESS])
        >>> evaluator = RAGASEvaluator(config)
        >>> sample = RAGSample(
        ...     question="What is MLOps?",
        ...     answer="MLOps is ML + DevOps practices.",
        ...     contexts=["MLOps combines ML and DevOps..."],
        ... )
        >>> result = await evaluator.evaluate([sample])
        >>> print(result.scores)
    """

    def __init__(self, config: Optional[RAGASConfig] = None):
        """Initialize RAGAS evaluator.

        Args:
            config: RAGAS configuration. Uses defaults if not provided.
        """
        self.config = config or RAGASConfig()
        self._metrics = None
        self._llm = None
        self._embeddings = None

    def _initialize_metrics(self) -> None:
        """Initialize RAGAS metrics lazily."""
        if self._metrics is not None:
            return

        try:
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                answer_similarity,
                context_precision,
                context_recall,
                context_relevancy,
                faithfulness,
            )

            metric_map = {
                RAGASMetricType.FAITHFULNESS: faithfulness,
                RAGASMetricType.ANSWER_RELEVANCY: answer_relevancy,
                RAGASMetricType.CONTEXT_PRECISION: context_precision,
                RAGASMetricType.CONTEXT_RECALL: context_recall,
                RAGASMetricType.CONTEXT_RELEVANCY: context_relevancy,
                RAGASMetricType.ANSWER_SIMILARITY: answer_similarity,
                RAGASMetricType.ANSWER_CORRECTNESS: answer_correctness,
            }

            self._metrics = [
                metric_map[m]
                for m in self.config.metrics
                if m in metric_map
            ]
            logger.info(f"Initialized {len(self._metrics)} RAGAS metrics")

        except ImportError as e:
            logger.error(f"Failed to import RAGAS: {e}")
            raise ImportError(
                "RAGAS is required for RAGASEvaluator. "
                "Install with: pip install ragas"
            ) from e

    def _prepare_dataset(self, samples: list[RAGSample]) -> Any:
        """Prepare samples as RAGAS dataset.

        Args:
            samples: List of RAG samples to evaluate.

        Returns:
            RAGAS-compatible dataset.
        """
        from datasets import Dataset

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        }
        return Dataset.from_dict(data)

    async def evaluate(
        self,
        samples: list[RAGSample],
        experiment_name: Optional[str] = None,
    ) -> RAGASResult:
        """Evaluate samples using RAGAS metrics.

        Args:
            samples: List of RAG samples to evaluate.
            experiment_name: Optional MLflow experiment name.

        Returns:
            RAGASResult with scores and detailed results.
        """
        self._initialize_metrics()

        result = RAGASResult(
            metadata={
                "num_samples": len(samples),
                "metrics": [m.value for m in self.config.metrics],
                "llm_model": self.config.llm_model,
            }
        )

        if not samples:
            logger.warning("No samples provided for evaluation")
            return result

        try:
            from ragas import evaluate as ragas_evaluate

            dataset = self._prepare_dataset(samples)

            logger.info(
                f"Running RAGAS evaluation on {len(samples)} samples "
                f"with {len(self._metrics)} metrics"
            )

            # Run evaluation
            eval_result = ragas_evaluate(
                dataset,
                metrics=self._metrics,
            )

            # Extract scores
            for metric in self.config.metrics:
                metric_name = metric.value
                if metric_name in eval_result:
                    result.scores[metric_name] = float(eval_result[metric_name])

            # Store detailed results
            if hasattr(eval_result, "to_pandas"):
                df = eval_result.to_pandas()
                result.detailed_results = df.to_dict(orient="records")

            logger.info(f"RAGAS evaluation complete: {result.scores}")

            # Log to MLflow if enabled
            if self.config.log_to_mlflow:
                self._log_to_mlflow(result, experiment_name)

        except Exception as e:
            error_msg = f"RAGAS evaluation failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            if self.config.raise_on_error:
                raise

        return result

    def evaluate_sync(
        self,
        samples: list[RAGSample],
        experiment_name: Optional[str] = None,
    ) -> RAGASResult:
        """Synchronous version of evaluate.

        Args:
            samples: List of RAG samples to evaluate.
            experiment_name: Optional MLflow experiment name.

        Returns:
            RAGASResult with scores and detailed results.
        """
        import asyncio

        return asyncio.run(self.evaluate(samples, experiment_name))

    def _log_to_mlflow(
        self,
        result: RAGASResult,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Log evaluation results to MLflow.

        Args:
            result: Evaluation result to log.
            experiment_name: Optional experiment name.
        """
        try:
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow.start_run(nested=True):
                # Log metrics
                for metric_name, score in result.scores.items():
                    mlflow.log_metric(f"ragas_{metric_name}", score)

                mlflow.log_metric("ragas_average_score", result.average_score)

                # Log params
                mlflow.log_params({
                    "ragas_llm_model": self.config.llm_model,
                    "ragas_embedding_model": self.config.embedding_model,
                    "ragas_num_samples": result.metadata.get("num_samples", 0),
                })

                logger.info("Logged RAGAS results to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


def create_ragas_evaluator(
    metrics: Optional[list[str]] = None,
    llm_model: str = "gpt-4",
    log_to_mlflow: bool = True,
) -> RAGASEvaluator:
    """Factory function to create RAGAS evaluator.

    Args:
        metrics: List of metric names to use. Uses defaults if not provided.
        llm_model: LLM model for evaluation.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Configured RAGASEvaluator instance.
    """
    metric_types = None
    if metrics:
        metric_types = [RAGASMetricType(m) for m in metrics]

    config = RAGASConfig(
        metrics=metric_types or RAGASConfig().metrics,
        llm_model=llm_model,
        log_to_mlflow=log_to_mlflow,
    )
    return RAGASEvaluator(config)
