# Copyright 2024 LLM MLOps Platform
# SPDX-License-Identifier: Apache-2.0

"""Evaluation module for LLM MLOps Platform.

This module provides evaluation capabilities using RAGAS and DeepEval
frameworks for comprehensive LLM and RAG system assessment.
"""

from src.evaluation.metrics.ragas_metrics import (
    RAGASEvaluator,
    RAGASConfig,
    RAGASResult,
)
from src.evaluation.metrics.deepeval_metrics import (
    DeepEvalEvaluator,
    DeepEvalConfig,
    DeepEvalResult,
)
from src.evaluation.runners.eval_runner import (
    EvaluationRunner,
    EvaluationConfig,
    EvaluationResult,
)

__all__ = [
    # RAGAS
    "RAGASEvaluator",
    "RAGASConfig",
    "RAGASResult",
    # DeepEval
    "DeepEvalEvaluator",
    "DeepEvalConfig",
    "DeepEvalResult",
    # Runners
    "EvaluationRunner",
    "EvaluationConfig",
    "EvaluationResult",
]
