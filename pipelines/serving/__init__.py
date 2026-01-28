"""Serving pipelines package.

This package provides Kubeflow Pipeline definitions for
model deployment and inference service management.
"""

from pipelines.serving.components import (
    deploy_model_component,
    validate_deployment_component,
    promote_model_component,
)
from pipelines.serving.deploy_pipeline import (
    model_deployment_pipeline,
    canary_deployment_pipeline,
)

__all__ = [
    # Components
    "deploy_model_component",
    "validate_deployment_component",
    "promote_model_component",
    # Pipelines
    "model_deployment_pipeline",
    "canary_deployment_pipeline",
]
