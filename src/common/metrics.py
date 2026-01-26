"""Prometheus metrics utilities."""

from functools import lru_cache
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server


# Define standard metrics
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["model", "status"],
)

REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

TOKENS_PROCESSED = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["model", "type"],  # type: prompt or completion
)

TOKENS_PER_SECOND = Gauge(
    "llm_tokens_per_second",
    "Current tokens per second throughput",
    ["model"],
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of active requests",
    ["model"],
)

GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"],
)

MODEL_INFO = Info(
    "llm_model",
    "LLM model information",
)


class MetricsClient:
    """Client for recording metrics."""

    def __init__(self, enabled: bool = True, port: int = 9090):
        self.enabled = enabled
        self.port = port
        self._started = False

    def start_server(self) -> None:
        """Start the Prometheus metrics server."""
        if self.enabled and not self._started:
            start_http_server(self.port)
            self._started = True

    def record_request(
        self,
        model: str,
        status: str,
        latency: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record a completed request.

        Args:
            model: Model name
            status: Request status (success, error)
            latency: Request latency in seconds
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        if not self.enabled:
            return

        REQUEST_COUNT.labels(model=model, status=status).inc()
        REQUEST_LATENCY.labels(model=model).observe(latency)
        TOKENS_PROCESSED.labels(model=model, type="prompt").inc(prompt_tokens)
        TOKENS_PROCESSED.labels(model=model, type="completion").inc(completion_tokens)

    def set_active_requests(self, model: str, count: int) -> None:
        """Set active request count.

        Args:
            model: Model name
            count: Active request count
        """
        if self.enabled:
            ACTIVE_REQUESTS.labels(model=model).set(count)

    def set_tokens_per_second(self, model: str, tps: float) -> None:
        """Set tokens per second throughput.

        Args:
            model: Model name
            tps: Tokens per second
        """
        if self.enabled:
            TOKENS_PER_SECOND.labels(model=model).set(tps)

    def set_gpu_memory(self, device: str, bytes_used: int) -> None:
        """Set GPU memory usage.

        Args:
            device: GPU device identifier
            bytes_used: Memory used in bytes
        """
        if self.enabled:
            GPU_MEMORY_USED.labels(device=device).set(bytes_used)

    def set_model_info(self, **info: str) -> None:
        """Set model information.

        Args:
            **info: Model information key-value pairs
        """
        if self.enabled:
            MODEL_INFO.info(info)

    def request_timer(self, model: str) -> Callable:
        """Context manager for timing requests.

        Args:
            model: Model name

        Returns:
            Context manager that records latency
        """
        return REQUEST_LATENCY.labels(model=model).time()


@lru_cache
def get_metrics_client(enabled: bool = True, port: int = 9090) -> MetricsClient:
    """Get cached metrics client.

    Args:
        enabled: Whether metrics are enabled
        port: Metrics server port

    Returns:
        MetricsClient instance
    """
    return MetricsClient(enabled=enabled, port=port)
