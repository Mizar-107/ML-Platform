"""Ray Serve Embedding Service.

Distributed embedding service using Ray Serve for high-throughput
embedding generation with autoscaling.
"""

import asyncio
from typing import Any

import structlog
from pydantic import BaseModel
from ray import serve

from src.data.embedding.models import BaseEmbeddingModel, get_embedding_model

logger = structlog.get_logger()


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    texts: list[str]
    model_name: str | None = None


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embeddings: list[list[float]]
    model: str
    dimension: int
    num_texts: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    dimension: int


@serve.deployment(
    name="embedding-service",
    num_replicas=2,
    ray_actor_options={"num_gpus": 0.5},
    max_concurrent_queries=100,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 10,
        "upscale_delay_s": 10,
        "downscale_delay_s": 60,
    },
)
class EmbeddingService:
    """Ray Serve deployment for embedding generation."""

    def __init__(
        self,
        model_type: str = "sentence-transformer",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize embedding service.

        Args:
            model_type: Type of embedding model
            model_name: Model name/path
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model: BaseEmbeddingModel | None = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        logger.info(
            "Initializing embedding model",
            model_type=self.model_type,
            model_name=self.model_name,
        )
        self.model = get_embedding_model(self.model_type, self.model_name)
        logger.info(
            "Model initialized",
            dimension=self.model.dimension,
        )

    async def health(self) -> HealthResponse:
        """Health check endpoint.

        Returns:
            Health status
        """
        return HealthResponse(
            status="healthy",
            model=self.model_name,
            dimension=self.model.dimension if self.model else 0,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for texts.

        Args:
            request: Embedding request with texts

        Returns:
            Embedding response with vectors
        """
        if not self.model:
            raise RuntimeError("Model not initialized")

        # Use provided model or default
        if request.model_name and request.model_name != self.model_name:
            logger.warning(
                "Model override requested but not supported in this deployment",
                requested=request.model_name,
                using=self.model_name,
            )

        # Generate embeddings
        logger.debug("Generating embeddings", num_texts=len(request.texts))
        results = self.model.embed(request.texts)

        embeddings = [r.embedding for r in results]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_name,
            dimension=self.model.dimension,
            num_texts=len(request.texts),
        )

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle incoming HTTP requests.

        Args:
            request: Request dictionary

        Returns:
            Response dictionary
        """
        # Check for health endpoint
        if request.get("endpoint") == "health":
            response = await self.health()
            return response.model_dump()

        # Parse embedding request
        emb_request = EmbeddingRequest(**request)
        response = await self.embed(emb_request)
        return response.model_dump()


def deploy_embedding_service(
    model_type: str = "sentence-transformer",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Deploy the embedding service.

    Args:
        model_type: Type of embedding model
        model_name: Model name/path
        host: Bind host
        port: Bind port
    """
    import ray

    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init()

    # Deploy service
    serve.start(http_options={"host": host, "port": port})

    EmbeddingService.bind(model_type=model_type, model_name=model_name)

    logger.info(
        "Embedding service deployed",
        host=host,
        port=port,
        model=model_name,
    )


def create_embedding_client(
    service_url: str = "http://localhost:8000",
) -> "EmbeddingClient":
    """Create a client for the embedding service.

    Args:
        service_url: URL of the embedding service

    Returns:
        EmbeddingClient instance
    """
    return EmbeddingClient(service_url)


class EmbeddingClient:
    """Client for the Ray Serve embedding service."""

    def __init__(self, service_url: str = "http://localhost:8000"):
        """Initialize client.

        Args:
            service_url: URL of the embedding service
        """
        self.service_url = service_url.rstrip("/")

    async def embed_async(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings asynchronously.

        Args:
            texts: List of texts to embed
            model_name: Optional model override

        Returns:
            Embedding response
        """
        import httpx

        request = EmbeddingRequest(texts=texts, model_name=model_name)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.service_url}/embed",
                json=request.model_dump(),
                timeout=60.0,
            )
            response.raise_for_status()
            return EmbeddingResponse(**response.json())

    def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingResponse:
        """Generate embeddings synchronously.

        Args:
            texts: List of texts to embed
            model_name: Optional model override

        Returns:
            Embedding response
        """
        return asyncio.run(self.embed_async(texts, model_name))

    async def health_async(self) -> HealthResponse:
        """Check service health asynchronously.

        Returns:
            Health response
        """
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.service_url}/health",
                timeout=10.0,
            )
            response.raise_for_status()
            return HealthResponse(**response.json())

    def health(self) -> HealthResponse:
        """Check service health synchronously.

        Returns:
            Health response
        """
        return asyncio.run(self.health_async())


if __name__ == "__main__":
    # Deploy service when run directly
    deploy_embedding_service()
