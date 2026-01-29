# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""DeepEval evaluation metrics for LLM assessment.

This module provides DeepEval-based evaluation metrics for comprehensive
LLM quality assessment including hallucination detection, toxicity,
bias, coherence, and other safety/quality metrics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import mlflow
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeepEvalMetricType(str, Enum):
    """Available DeepEval metrics."""

    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    SUMMARIZATION = "summarization"
    G_EVAL = "g_eval"


class DeepEvalConfig(BaseModel):
    """Configuration for DeepEval evaluation."""

    metrics: list[DeepEvalMetricType] = Field(
        default=[
            DeepEvalMetricType.HALLUCINATION,
            DeepEvalMetricType.TOXICITY,
            DeepEvalMetricType.COHERENCE,
            DeepEvalMetricType.RELEVANCE,
        ],
        description="Metrics to evaluate",
    )
    model: str = Field(
        default="gpt-4",
        description="LLM model for evaluation",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for pass/fail",
    )
    include_reason: bool = Field(
        default=True,
        description="Include reasoning in results",
    )
    strict_mode: bool = Field(
        default=False,
        description="Use strict evaluation mode",
    )
    async_mode: bool = Field(
        default=True,
        description="Use async evaluation",
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
class DeepEvalResult:
    """Result from DeepEval evaluation."""

    scores: dict[str, float] = field(default_factory=dict)
    passed: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
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
    def all_passed(self) -> bool:
        """Check if all metrics passed."""
        return all(self.passed.values()) if self.passed else False

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if not self.passed:
            return 0.0
        return sum(self.passed.values()) / len(self.passed)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scores": self.scores,
            "passed": self.passed,
            "reasons": self.reasons,
            "average_score": self.average_score,
            "all_passed": self.all_passed,
            "pass_rate": self.pass_rate,
            "detailed_results": self.detailed_results,
            "metadata": self.metadata,
            "errors": self.errors,
        }


@dataclass
class LLMTestCase:
    """A single test case for DeepEval evaluation."""

    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[list[str]] = None
    retrieval_context: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
        }


class DeepEvalEvaluator:
    """Evaluator using DeepEval metrics for LLM assessment.

    DeepEval provides comprehensive metrics for evaluating LLM outputs
    including safety, quality, and factuality metrics:
    - Hallucination: Detect factual inconsistencies
    - Toxicity: Detect harmful or offensive content
    - Bias: Detect unfair bias in responses
    - Coherence: Evaluate response coherence
    - Relevance: Evaluate answer relevance

    Example:
        >>> config = DeepEvalConfig(metrics=[DeepEvalMetricType.HALLUCINATION])
        >>> evaluator = DeepEvalEvaluator(config)
        >>> test_case = LLMTestCase(
        ...     input="What is MLOps?",
        ...     actual_output="MLOps is ML + DevOps practices.",
        ...     context=["MLOps combines ML and DevOps..."],
        ... )
        >>> result = await evaluator.evaluate([test_case])
        >>> print(result.scores)
    """

    def __init__(self, config: Optional[DeepEvalConfig] = None):
        """Initialize DeepEval evaluator.

        Args:
            config: DeepEval configuration. Uses defaults if not provided.
        """
        self.config = config or DeepEvalConfig()
        self._metrics = None

    def _initialize_metrics(self) -> list:
        """Initialize DeepEval metrics lazily.

        Returns:
            List of initialized metric objects.
        """
        if self._metrics is not None:
            return self._metrics

        try:
            from deepeval.metrics import (
                BiasMetric,
                CoherenceMetric,
                FluencyMetric,
                GEval,
                GroundednessMetric,
                HallucinationMetric,
                RelevanceMetric,
                SummarizationMetric,
                ToxicityMetric,
            )

            self._metrics = []

            for metric_type in self.config.metrics:
                metric = self._create_metric(
                    metric_type,
                    HallucinationMetric,
                    ToxicityMetric,
                    BiasMetric,
                    CoherenceMetric,
                    FluencyMetric,
                    RelevanceMetric,
                    GroundednessMetric,
                    SummarizationMetric,
                    GEval,
                )
                if metric:
                    self._metrics.append(metric)

            logger.info(f"Initialized {len(self._metrics)} DeepEval metrics")
            return self._metrics

        except ImportError as e:
            logger.error(f"Failed to import DeepEval: {e}")
            raise ImportError(
                "DeepEval is required for DeepEvalEvaluator. "
                "Install with: pip install deepeval"
            ) from e

    def _create_metric(
        self,
        metric_type: DeepEvalMetricType,
        HallucinationMetric,
        ToxicityMetric,
        BiasMetric,
        CoherenceMetric,
        FluencyMetric,
        RelevanceMetric,
        GroundednessMetric,
        SummarizationMetric,
        GEval,
    ):
        """Create a metric instance based on type.

        Args:
            metric_type: Type of metric to create.
            *Metric classes: DeepEval metric classes.

        Returns:
            Initialized metric instance or None.
        """
        threshold = self.config.threshold
        include_reason = self.config.include_reason
        strict_mode = self.config.strict_mode
        model = self.config.model

        try:
            if metric_type == DeepEvalMetricType.HALLUCINATION:
                return HallucinationMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.TOXICITY:
                return ToxicityMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.BIAS:
                return BiasMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.COHERENCE:
                return CoherenceMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.FLUENCY:
                return FluencyMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.RELEVANCE:
                return RelevanceMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.GROUNDEDNESS:
                return GroundednessMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.SUMMARIZATION:
                return SummarizationMetric(
                    threshold=threshold,
                    model=model,
                    include_reason=include_reason,
                )
            elif metric_type == DeepEvalMetricType.G_EVAL:
                return GEval(
                    name="quality",
                    criteria="Evaluate the overall quality of the response",
                    threshold=threshold,
                    model=model,
                    strict_mode=strict_mode,
                )
            else:
                logger.warning(f"Unknown metric type: {metric_type}")
                return None
        except Exception as e:
            logger.warning(f"Failed to create metric {metric_type}: {e}")
            return None

    def _convert_test_case(self, test_case: LLMTestCase):
        """Convert LLMTestCase to DeepEval test case.

        Args:
            test_case: Test case to convert.

        Returns:
            DeepEval LLMTestCase instance.
        """
        from deepeval.test_case import LLMTestCase as DeepEvalTestCase

        return DeepEvalTestCase(
            input=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            context=test_case.context,
            retrieval_context=test_case.retrieval_context,
        )

    async def evaluate(
        self,
        test_cases: list[LLMTestCase],
        experiment_name: Optional[str] = None,
    ) -> DeepEvalResult:
        """Evaluate test cases using DeepEval metrics.

        Args:
            test_cases: List of test cases to evaluate.
            experiment_name: Optional MLflow experiment name.

        Returns:
            DeepEvalResult with scores and detailed results.
        """
        metrics = self._initialize_metrics()

        result = DeepEvalResult(
            metadata={
                "num_test_cases": len(test_cases),
                "metrics": [m.value for m in self.config.metrics],
                "model": self.config.model,
                "threshold": self.config.threshold,
            }
        )

        if not test_cases:
            logger.warning("No test cases provided for evaluation")
            return result

        if not metrics:
            logger.warning("No metrics initialized")
            return result

        try:
            from deepeval import evaluate as deepeval_evaluate

            # Convert test cases
            deep_eval_cases = [
                self._convert_test_case(tc) for tc in test_cases
            ]

            logger.info(
                f"Running DeepEval evaluation on {len(test_cases)} test cases "
                f"with {len(metrics)} metrics"
            )

            # Run evaluation
            eval_results = deepeval_evaluate(
                test_cases=deep_eval_cases,
                metrics=metrics,
            )

            # Process results
            for metric in metrics:
                metric_name = metric.__class__.__name__.lower().replace("metric", "")

                # Aggregate scores across test cases
                scores = []
                passed_count = 0

                for case_result in eval_results:
                    if hasattr(case_result, metric_name):
                        metric_result = getattr(case_result, metric_name)
                        if hasattr(metric_result, "score"):
                            scores.append(metric_result.score)
                        if hasattr(metric_result, "success"):
                            passed_count += int(metric_result.success)

                if scores:
                    result.scores[metric_name] = sum(scores) / len(scores)
                    result.passed[metric_name] = passed_count == len(scores)

                # Get reason from first test case
                if hasattr(metric, "reason"):
                    result.reasons[metric_name] = metric.reason or ""

            # Store detailed results
            result.detailed_results = [
                tc.to_dict() for tc in test_cases
            ]

            logger.info(
                f"DeepEval evaluation complete: {result.scores}, "
                f"pass_rate: {result.pass_rate:.2%}"
            )

            # Log to MLflow if enabled
            if self.config.log_to_mlflow:
                self._log_to_mlflow(result, experiment_name)

        except Exception as e:
            error_msg = f"DeepEval evaluation failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            if self.config.raise_on_error:
                raise

        return result

    def evaluate_sync(
        self,
        test_cases: list[LLMTestCase],
        experiment_name: Optional[str] = None,
    ) -> DeepEvalResult:
        """Synchronous version of evaluate.

        Args:
            test_cases: List of test cases to evaluate.
            experiment_name: Optional MLflow experiment name.

        Returns:
            DeepEvalResult with scores and detailed results.
        """
        import asyncio

        return asyncio.run(self.evaluate(test_cases, experiment_name))

    def _log_to_mlflow(
        self,
        result: DeepEvalResult,
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
                    mlflow.log_metric(f"deepeval_{metric_name}", score)

                mlflow.log_metric("deepeval_average_score", result.average_score)
                mlflow.log_metric("deepeval_pass_rate", result.pass_rate)

                # Log params
                mlflow.log_params({
                    "deepeval_model": self.config.model,
                    "deepeval_threshold": self.config.threshold,
                    "deepeval_num_cases": result.metadata.get("num_test_cases", 0),
                })

                logger.info("Logged DeepEval results to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


def create_deepeval_evaluator(
    metrics: Optional[list[str]] = None,
    model: str = "gpt-4",
    threshold: float = 0.5,
    log_to_mlflow: bool = True,
) -> DeepEvalEvaluator:
    """Factory function to create DeepEval evaluator.

    Args:
        metrics: List of metric names to use. Uses defaults if not provided.
        model: LLM model for evaluation.
        threshold: Pass/fail threshold.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Configured DeepEvalEvaluator instance.
    """
    metric_types = None
    if metrics:
        metric_types = [DeepEvalMetricType(m) for m in metrics]

    config = DeepEvalConfig(
        metrics=metric_types or DeepEvalConfig().metrics,
        model=model,
        threshold=threshold,
        log_to_mlflow=log_to_mlflow,
    )
    return DeepEvalEvaluator(config)
