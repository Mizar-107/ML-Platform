"""Kubeflow Pipeline components for model serving.

This module provides KFP components for deploying, validating,
and managing InferenceServices with KServe.
"""

from kfp import dsl
from kfp.dsl import Artifact, Input, Output, Metrics


# Base image for serving components
SERVING_IMAGE = "python:3.10-slim"


@dsl.component(
    base_image=SERVING_IMAGE,
    packages_to_install=["kubernetes>=28.0.0", "pydantic>=2.0.0"],
)
def deploy_model_component(
    model_name: str,
    model_uri: str,
    namespace: str,
    runtime: str,
    min_replicas: int,
    max_replicas: int,
    gpu_count: int,
    memory_limit: str,
    cpu_limit: str,
    max_model_len: int,
    deployment_info: Output[Artifact],
) -> str:
    """Deploy a model as KServe InferenceService.

    Args:
        model_name: Name for the InferenceService
        model_uri: Model storage URI (s3:// or hf://)
        namespace: Kubernetes namespace
        runtime: Serving runtime (e.g., kserve-vllm)
        min_replicas: Minimum replicas
        max_replicas: Maximum replicas
        gpu_count: GPUs per replica
        memory_limit: Memory limit (e.g., "32Gi")
        cpu_limit: CPU limit (e.g., "8")
        max_model_len: Maximum model sequence length
        deployment_info: Output artifact with deployment details

    Returns:
        InferenceService URL
    """
    import json
    import time
    from kubernetes import client, config

    # Load in-cluster config
    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    # InferenceService specification
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": model_name,
                "app.kubernetes.io/part-of": "mlops-platform",
            },
            "annotations": {
                "prometheus.io/scrape": "true",
            },
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "huggingface"},
                    "runtime": runtime,
                    "storageUri": model_uri,
                    "resources": {
                        "requests": {
                            "cpu": cpu_limit,
                            "memory": memory_limit,
                            "nvidia.com/gpu": str(gpu_count),
                        },
                        "limits": {
                            "cpu": cpu_limit,
                            "memory": memory_limit,
                            "nvidia.com/gpu": str(gpu_count),
                        },
                    },
                },
                "containers": [
                    {
                        "name": "kserve-container",
                        "args": [
                            "--host=0.0.0.0",
                            "--port=8000",
                            "--model=$(MODEL_NAME)",
                            f"--max-model-len={max_model_len}",
                            "--gpu-memory-utilization=0.90",
                            "--dtype=bfloat16",
                            "--trust-remote-code",
                        ],
                    }
                ],
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
            },
        },
    }

    # Create or update InferenceService
    try:
        # Try to get existing
        existing = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
        )
        # Update existing
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service,
        )
        print(f"Updated InferenceService {model_name}")
    except client.ApiException as e:
        if e.status == 404:
            # Create new
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service,
            )
            print(f"Created InferenceService {model_name}")
        else:
            raise

    # Wait for ready status
    print("Waiting for InferenceService to be ready...")
    max_wait = 600  # 10 minutes
    start_time = time.time()
    service_url = None

    while time.time() - start_time < max_wait:
        isvc = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
        )

        status = isvc.get("status", {})
        conditions = {c["type"]: c for c in status.get("conditions", [])}

        if conditions.get("Ready", {}).get("status") == "True":
            service_url = status.get("url", "")
            print(f"InferenceService ready: {service_url}")
            break

        inference_status = conditions.get("IngressReady", {}).get("status")
        print(f"Status: IngressReady={inference_status}")
        time.sleep(10)
    else:
        raise TimeoutError(f"InferenceService {model_name} not ready after {max_wait}s")

    # Save deployment info
    deployment_data = {
        "name": model_name,
        "namespace": namespace,
        "url": service_url,
        "model_uri": model_uri,
        "runtime": runtime,
        "replicas": {"min": min_replicas, "max": max_replicas},
        "resources": {
            "gpu": gpu_count,
            "memory": memory_limit,
            "cpu": cpu_limit,
        },
    }

    with open(deployment_info.path, "w") as f:
        json.dump(deployment_data, f, indent=2)

    return service_url


@dsl.component(
    base_image=SERVING_IMAGE,
    packages_to_install=["requests>=2.31.0", "tenacity>=8.2.0"],
)
def validate_deployment_component(
    service_url: str,
    test_prompt: str,
    expected_min_tokens: int,
    timeout_seconds: int,
    metrics: Output[Metrics],
) -> bool:
    """Validate deployment with smoke test.

    Args:
        service_url: InferenceService URL
        test_prompt: Test prompt to send
        expected_min_tokens: Minimum expected tokens in response
        timeout_seconds: Request timeout
        metrics: Output metrics artifact

    Returns:
        True if validation passed
    """
    import json
    import time
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def call_inference(url: str, prompt: str, timeout: int) -> dict:
        """Make inference request with retries."""
        response = requests.post(
            f"{url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    print(f"Validating deployment at {service_url}")
    start_time = time.time()

    try:
        # Make test request
        result = call_inference(service_url, test_prompt, timeout_seconds)
        latency = time.time() - start_time

        # Validate response
        response_text = result.get("text", result.get("choices", [{}])[0].get("text", ""))
        token_count = len(response_text.split())

        print(f"Response: {response_text[:200]}...")
        print(f"Tokens: {token_count}, Latency: {latency:.2f}s")

        # Check minimum tokens
        passed = token_count >= expected_min_tokens

        # Log metrics
        metrics.log_metric("latency_seconds", latency)
        metrics.log_metric("token_count", token_count)
        metrics.log_metric("passed", 1 if passed else 0)

        if not passed:
            print(f"FAILED: Expected at least {expected_min_tokens} tokens, got {token_count}")

        return passed

    except Exception as e:
        print(f"Validation failed with error: {e}")
        metrics.log_metric("passed", 0)
        metrics.log_metric("error", str(e))
        return False


@dsl.component(
    base_image=SERVING_IMAGE,
    packages_to_install=["kubernetes>=28.0.0"],
)
def promote_model_component(
    model_name: str,
    namespace: str,
    traffic_percent: int,
    deployment_info: Input[Artifact],
) -> str:
    """Promote model to production traffic.

    Used for canary deployments to gradually shift traffic.

    Args:
        model_name: InferenceService name
        namespace: Kubernetes namespace
        traffic_percent: Percentage of traffic to route (0-100)
        deployment_info: Deployment details artifact

    Returns:
        Updated service URL
    """
    import json
    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    # Load deployment info
    with open(deployment_info.path) as f:
        deploy_data = json.load(f)

    print(f"Promoting {model_name} to {traffic_percent}% traffic")

    # Get current InferenceService
    isvc = custom_api.get_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=model_name,
    )

    # Update traffic routing
    if traffic_percent == 100:
        # Full promotion - remove canary
        if "canaryTrafficPercent" in isvc.get("spec", {}).get("predictor", {}):
            del isvc["spec"]["predictor"]["canaryTrafficPercent"]
    else:
        # Canary traffic
        isvc["spec"]["predictor"]["canaryTrafficPercent"] = traffic_percent

    # Apply update
    custom_api.patch_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=model_name,
        body=isvc,
    )

    print(f"Traffic updated to {traffic_percent}%")

    # Get updated URL
    updated_isvc = custom_api.get_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=model_name,
    )

    return updated_isvc.get("status", {}).get("url", deploy_data.get("url", ""))


@dsl.component(
    base_image=SERVING_IMAGE,
    packages_to_install=["kubernetes>=28.0.0"],
)
def rollback_deployment_component(
    model_name: str,
    namespace: str,
    previous_model_uri: str,
) -> str:
    """Rollback to previous model version.

    Args:
        model_name: InferenceService name
        namespace: Kubernetes namespace
        previous_model_uri: Previous model URI to rollback to

    Returns:
        Rollback status message
    """
    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    print(f"Rolling back {model_name} to {previous_model_uri}")

    # Get current InferenceService
    isvc = custom_api.get_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=model_name,
    )

    # Update storage URI
    isvc["spec"]["predictor"]["model"]["storageUri"] = previous_model_uri

    # Remove canary if present
    if "canaryTrafficPercent" in isvc.get("spec", {}).get("predictor", {}):
        del isvc["spec"]["predictor"]["canaryTrafficPercent"]

    # Apply rollback
    custom_api.patch_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        name=model_name,
        body=isvc,
    )

    return f"Rolled back {model_name} to {previous_model_uri}"


@dsl.component(
    base_image=SERVING_IMAGE,
    packages_to_install=["kubernetes>=28.0.0"],
)
def delete_deployment_component(
    model_name: str,
    namespace: str,
) -> str:
    """Delete an InferenceService.

    Args:
        model_name: InferenceService name
        namespace: Kubernetes namespace

    Returns:
        Deletion status message
    """
    from kubernetes import client, config

    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    print(f"Deleting InferenceService {model_name}")

    try:
        custom_api.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
        )
        return f"Deleted InferenceService {model_name}"
    except client.ApiException as e:
        if e.status == 404:
            return f"InferenceService {model_name} not found"
        raise
