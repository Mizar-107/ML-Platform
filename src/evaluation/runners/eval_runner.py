# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""Unified evaluation runner for LLM MLOps Platform.

This module provides a unified evaluation runner that supports both
RAGAS and DeepEval frameworks with parallel execution using Ray.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
from pydantic import BaseModel, Field

from src.evaluation.metrics.deepeval_metrics import (
    DeepEvalConfig,
    DeepEvalEvaluator,
    DeepEvalResult,
    LLMTestCase,
)
from src.evaluation.metrics.ragas_metrics import (
    RAGASConfig,
    RAGASEvaluator,
    RAGASResult,
    RAGSample,
)

logger = logging.getLogger(__name__)


class EvaluationFramework(str, Enum):
    """Supported evaluation frameworks."""

    RAGAS = "ragas"
    DEEPEVAL = "deepeval"
    BOTH = "both"


class EvaluationConfig(BaseModel):
    """Configuration for unified evaluation runner."""

    framework: EvaluationFramework = Field(
        default=EvaluationFramework.BOTH,
        description="Evaluation framework(s) to use",
    )
    ragas_config: Optional[RAGASConfig] = Field(
        default=None,
        description="RAGAS configuration",
    )
    deepeval_config: Optional[DeepEvalConfig] = Field(
        default=None,
        description="DeepEval configuration",
    )
    parallel: bool = Field(
        default=True,
        description="Run evaluations in parallel using Ray",
    )
    ray_address: Optional[str] = Field(
        default=None,
        description="Ray cluster address",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save results",
    )
    experiment_name: Optional[str] = Field(
        default="llm-evaluation",
        description="MLflow experiment name",
    )
    save_detailed_results: bool = Field(
        default=True,
        description="Save detailed results to file",
    )

    class Config:
        use_enum_values = True


@dataclass
class EvaluationResult:
    """Combined result from all evaluation frameworks."""

    ragas_result: Optional[RAGASResult] = None
    deepeval_result: Optional[DeepEvalResult] = None
    combined_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall score across all metrics."""
        if not self.combined_scores:
            return 0.0
        return sum(self.combined_scores.values()) / len(self.combined_scores)

    @property
    def passed(self) -> bool:
        """Check if overall evaluation passed."""
        ragas_passed = self.ragas_result.passed if self.ragas_result else True
        deepeval_passed = (
            self.deepeval_result.all_passed if self.deepeval_result else True
        )
        return ragas_passed and deepeval_passed

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "ragas_result": (
                self.ragas_result.to_dict() if self.ragas_result else None
            ),
            "deepeval_result": (
                self.deepeval_result.to_dict() if self.deepeval_result else None
            ),
            "combined_scores": self.combined_scores,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "metadata": self.metadata,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
        ]

        if self.ragas_result:
            lines.append("\nRAGAS Metrics:")
            for name, score in self.ragas_result.scores.items():
                lines.append(f"  {name}: {score:.4f}")
            lines.append(f"  Average: {self.ragas_result.average_score:.4f}")

        if self.deepeval_result:
            lines.append("\nDeepEval Metrics:")
            for name, score in self.deepeval_result.scores.items():
                passed = self.deepeval_result.passed.get(name, False)
                status = "✓" if passed else "✗"
                lines.append(f"  {name}: {score:.4f} {status}")
            lines.append(f"  Pass Rate: {self.deepeval_result.pass_rate:.2%}")

        lines.extend([
            "",
            "-" * 60,
            f"Overall Score: {self.overall_score:.4f}",
            f"Passed: {'Yes' if self.passed else 'No'}",
            "=" * 60,
        ])

        return "\n".join(lines)


class EvaluationRunner:
    """Unified evaluation runner supporting RAGAS and DeepEval.

    This runner provides a unified interface for running evaluations
    using both RAGAS and DeepEval frameworks, with optional parallel
    execution using Ray.

    Example:
        >>> config = EvaluationConfig(framework=EvaluationFramework.BOTH)
        >>> runner = EvaluationRunner(config)
        >>> # For RAGAS
        >>> samples = [RAGSample(question="...", answer="...", contexts=[...])]
        >>> # For DeepEval
        >>> test_cases = [LLMTestCase(input="...", actual_output="...")]
        >>> result = await runner.run(
        ...     ragas_samples=samples,
        ...     deepeval_cases=test_cases,
        ... )
        >>> print(result.summary())
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluation runner.

        Args:
            config: Evaluation configuration. Uses defaults if not provided.
        """
        self.config = config or EvaluationConfig()
        self._ragas_evaluator: Optional[RAGASEvaluator] = None
        self._deepeval_evaluator: Optional[DeepEvalEvaluator] = None
        self._ray_initialized = False

    def _init_ray(self) -> None:
        """Initialize Ray for parallel evaluation."""
        if self._ray_initialized or not self.config.parallel:
            return

        try:
            import ray

            if not ray.is_initialized():
                if self.config.ray_address:
                    ray.init(address=self.config.ray_address)
                else:
                    ray.init(ignore_reinit_error=True)
            self._ray_initialized = True
            logger.info("Ray initialized for parallel evaluation")
        except ImportError:
            logger.warning("Ray not available, falling back to sequential execution")
            self.config.parallel = False

    def _get_ragas_evaluator(self) -> RAGASEvaluator:
        """Get or create RAGAS evaluator."""
        if self._ragas_evaluator is None:
            config = self.config.ragas_config or RAGASConfig()
            self._ragas_evaluator = RAGASEvaluator(config)
        return self._ragas_evaluator

    def _get_deepeval_evaluator(self) -> DeepEvalEvaluator:
        """Get or create DeepEval evaluator."""
        if self._deepeval_evaluator is None:
            config = self.config.deepeval_config or DeepEvalConfig()
            self._deepeval_evaluator = DeepEvalEvaluator(config)
        return self._deepeval_evaluator

    async def run(
        self,
        ragas_samples: Optional[list[RAGSample]] = None,
        deepeval_cases: Optional[list[LLMTestCase]] = None,
    ) -> EvaluationResult:
        """Run evaluation using configured frameworks.

        Args:
            ragas_samples: Samples for RAGAS evaluation.
            deepeval_cases: Test cases for DeepEval evaluation.

        Returns:
            EvaluationResult with results from all frameworks.
        """
        result = EvaluationResult(
            metadata={
                "framework": self.config.framework,
                "parallel": self.config.parallel,
                "experiment_name": self.config.experiment_name,
            }
        )

        use_ragas = self.config.framework in [
            EvaluationFramework.RAGAS,
            EvaluationFramework.BOTH,
        ]
        use_deepeval = self.config.framework in [
            EvaluationFramework.DEEPEVAL,
            EvaluationFramework.BOTH,
        ]

        # Initialize Ray for parallel execution
        if self.config.parallel and use_ragas and use_deepeval:
            self._init_ray()

        try:
            if self.config.parallel and self._ray_initialized:
                result = await self._run_parallel(
                    ragas_samples if use_ragas else None,
                    deepeval_cases if use_deepeval else None,
                    result,
                )
            else:
                result = await self._run_sequential(
                    ragas_samples if use_ragas else None,
                    deepeval_cases if use_deepeval else None,
                    result,
                )

            # Combine scores
            if result.ragas_result:
                for name, score in result.ragas_result.scores.items():
                    result.combined_scores[f"ragas_{name}"] = score

            if result.deepeval_result:
                for name, score in result.deepeval_result.scores.items():
                    result.combined_scores[f"deepeval_{name}"] = score

            # Save results if configured
            if self.config.save_detailed_results and self.config.output_dir:
                self._save_results(result)

            # Log summary to MLflow
            self._log_summary_to_mlflow(result)

        except Exception as e:
            error_msg = f"Evaluation runner failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    async def _run_sequential(
        self,
        ragas_samples: Optional[list[RAGSample]],
        deepeval_cases: Optional[list[LLMTestCase]],
        result: EvaluationResult,
    ) -> EvaluationResult:
        """Run evaluations sequentially.

        Args:
            ragas_samples: Samples for RAGAS evaluation.
            deepeval_cases: Test cases for DeepEval evaluation.
            result: Result object to populate.

        Returns:
            Updated EvaluationResult.
        """
        if ragas_samples:
            logger.info("Running RAGAS evaluation...")
            evaluator = self._get_ragas_evaluator()
            result.ragas_result = await evaluator.evaluate(
                ragas_samples,
                self.config.experiment_name,
            )

        if deepeval_cases:
            logger.info("Running DeepEval evaluation...")
            evaluator = self._get_deepeval_evaluator()
            result.deepeval_result = await evaluator.evaluate(
                deepeval_cases,
                self.config.experiment_name,
            )

        return result

    async def _run_parallel(
        self,
        ragas_samples: Optional[list[RAGSample]],
        deepeval_cases: Optional[list[LLMTestCase]],
        result: EvaluationResult,
    ) -> EvaluationResult:
        """Run evaluations in parallel using Ray.

        Args:
            ragas_samples: Samples for RAGAS evaluation.
            deepeval_cases: Test cases for DeepEval evaluation.
            result: Result object to populate.

        Returns:
            Updated EvaluationResult.
        """
        import asyncio

        import ray

        @ray.remote
        def run_ragas(samples, config, experiment_name):
            """Ray remote function for RAGAS evaluation."""
            import asyncio

            evaluator = RAGASEvaluator(config)
            return asyncio.run(evaluator.evaluate(samples, experiment_name))

        @ray.remote
        def run_deepeval(cases, config, experiment_name):
            """Ray remote function for DeepEval evaluation."""
            import asyncio

            evaluator = DeepEvalEvaluator(config)
            return asyncio.run(evaluator.evaluate(cases, experiment_name))

        futures = []

        if ragas_samples:
            config = self.config.ragas_config or RAGASConfig()
            futures.append(
                ("ragas", run_ragas.remote(
                    ragas_samples, config, self.config.experiment_name
                ))
            )

        if deepeval_cases:
            config = self.config.deepeval_config or DeepEvalConfig()
            futures.append(
                ("deepeval", run_deepeval.remote(
                    deepeval_cases, config, self.config.experiment_name
                ))
            )

        logger.info(f"Running {len(futures)} evaluations in parallel")

        # Wait for all futures
        for name, future in futures:
            try:
                eval_result = ray.get(future)
                if name == "ragas":
                    result.ragas_result = eval_result
                else:
                    result.deepeval_result = eval_result
            except Exception as e:
                error_msg = f"{name} evaluation failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        return result

    def run_sync(
        self,
        ragas_samples: Optional[list[RAGSample]] = None,
        deepeval_cases: Optional[list[LLMTestCase]] = None,
    ) -> EvaluationResult:
        """Synchronous version of run.

        Args:
            ragas_samples: Samples for RAGAS evaluation.
            deepeval_cases: Test cases for DeepEval evaluation.

        Returns:
            EvaluationResult with results from all frameworks.
        """
        import asyncio

        return asyncio.run(self.run(ragas_samples, deepeval_cases))

    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to file.

        Args:
            result: Evaluation result to save.
        """
        import json
        from datetime import datetime

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"evaluation_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved evaluation results to {filename}")

    def _log_summary_to_mlflow(self, result: EvaluationResult) -> None:
        """Log evaluation summary to MLflow.

        Args:
            result: Evaluation result to log.
        """
        try:
            if self.config.experiment_name:
                mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name="evaluation_summary", nested=True):
                mlflow.log_metric("overall_score", result.overall_score)
                mlflow.log_metric("passed", int(result.passed))

                for name, score in result.combined_scores.items():
                    mlflow.log_metric(name, score)

                mlflow.log_params({
                    "framework": self.config.framework,
                    "parallel": self.config.parallel,
                })

                logger.info("Logged evaluation summary to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log summary to MLflow: {e}")


def create_evaluation_runner(
    framework: str = "both",
    parallel: bool = True,
    experiment_name: str = "llm-evaluation",
    output_dir: Optional[str] = None,
) -> EvaluationRunner:
    """Factory function to create evaluation runner.

    Args:
        framework: Evaluation framework ("ragas", "deepeval", or "both").
        parallel: Whether to run evaluations in parallel.
        experiment_name: MLflow experiment name.
        output_dir: Directory to save results.

    Returns:
        Configured EvaluationRunner instance.
    """
    config = EvaluationConfig(
        framework=EvaluationFramework(framework),
        parallel=parallel,
        experiment_name=experiment_name,
        output_dir=output_dir,
    )
    return EvaluationRunner(config)
