"""vLLM server launcher with health checks.

This module provides utilities for launching and managing
vLLM inference servers with async support.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from src.serving.config import ModelConfig, ServingConfig, VLLMConfig
from src.serving.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class ServerStats:
    """Server statistics container."""

    total_requests: int = 0
    active_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    errors: int = 0


class VLLMServer:
    """vLLM inference server with lifecycle management.

    Handles model loading, server startup, health checks,
    and graceful shutdown.
    """

    def __init__(self, config: ServingConfig):
        """Initialize the vLLM server.

        Args:
            config: Serving configuration
        """
        self.config = config
        self.engine = None
        self.stats = ServerStats()
        self._shutdown_event = asyncio.Event()
        self._model_path: Path | None = None

    async def initialize(self) -> None:
        """Initialize the vLLM engine.

        Downloads model if needed and creates the engine.
        """
        logger.info("Initializing vLLM server...")

        # Load model
        loader = ModelLoader(cache_dir=self.config.model.cache_dir)
        self._model_path = loader.load(self.config.model)

        # Create engine
        self.engine = await create_vllm_engine(
            model_path=str(self._model_path),
            vllm_config=self.config.vllm,
        )

        logger.info("vLLM engine initialized successfully")

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Shutting down vLLM server...")
        self._shutdown_event.set()

        if self.engine:
            # vLLM engine cleanup
            del self.engine
            self.engine = None

        logger.info("Server shutdown complete")

    def is_ready(self) -> bool:
        """Check if server is ready to accept requests.

        Returns:
            True if engine is initialized
        """
        return self.engine is not None

    def is_healthy(self) -> bool:
        """Check if server is healthy.

        Returns:
            True if server is functioning normally
        """
        if not self.is_ready():
            return False

        # Could add more health checks here
        # e.g., memory usage, GPU status, etc.
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_requests": self.stats.total_requests,
            "active_requests": self.stats.active_requests,
            "total_tokens_generated": self.stats.total_tokens_generated,
            "total_prompt_tokens": self.stats.total_prompt_tokens,
            "errors": self.stats.errors,
            "model": self.config.model.model_name_or_path,
            "ready": self.is_ready(),
            "healthy": self.is_healthy(),
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.is_ready():
            raise RuntimeError("Engine not initialized")

        self.stats.active_requests += 1
        self.stats.total_requests += 1

        try:
            # Import vLLM types
            from vllm import SamplingParams

            # Create sampling params
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

            # Generate (async)
            request_id = f"req-{self.stats.total_requests}"
            results = await self.engine.generate(prompt, sampling_params, request_id)

            # Extract text
            output_text = results.outputs[0].text

            # Update stats
            self.stats.total_tokens_generated += len(results.outputs[0].token_ids)
            self.stats.total_prompt_tokens += len(results.prompt_token_ids)

            return output_text

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Generation error: {e}")
            raise

        finally:
            self.stats.active_requests -= 1

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Yields:
            Generated text tokens

        Raises:
            RuntimeError: If engine not initialized
        """
        if not self.is_ready():
            raise RuntimeError("Engine not initialized")

        self.stats.active_requests += 1
        self.stats.total_requests += 1

        try:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

            request_id = f"req-{self.stats.total_requests}"

            async for result in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                # Yield incremental output
                if result.outputs:
                    yield result.outputs[0].text

            # Update stats after completion
            self.stats.total_tokens_generated += max_tokens  # Approximate

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Stream generation error: {e}")
            raise

        finally:
            self.stats.active_requests -= 1


async def create_vllm_engine(
    model_path: str,
    vllm_config: VLLMConfig,
):
    """Create an async vLLM engine.

    Args:
        model_path: Path to model directory
        vllm_config: vLLM configuration

    Returns:
        AsyncLLMEngine instance
    """
    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
    except ImportError as e:
        raise ImportError("vllm is required: pip install vllm") from e

    logger.info(f"Creating vLLM engine for {model_path}")

    # Build engine args
    engine_args = AsyncEngineArgs(
        model=model_path,
        **vllm_config.to_engine_args(),
    )

    # Create engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logger.info("vLLM engine created successfully")
    return engine


@asynccontextmanager
async def vllm_server_context(config: ServingConfig):
    """Context manager for vLLM server lifecycle.

    Args:
        config: Serving configuration

    Yields:
        VLLMServer instance
    """
    server = VLLMServer(config)

    try:
        await server.initialize()
        yield server
    finally:
        await server.shutdown()


def setup_signal_handlers(server: VLLMServer, loop: asyncio.AbstractEventLoop):
    """Setup signal handlers for graceful shutdown.

    Args:
        server: VLLMServer instance
        loop: Event loop
    """

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)


async def run_server(config: ServingConfig) -> None:
    """Run the vLLM server with FastAPI.

    Args:
        config: Serving configuration
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "fastapi and uvicorn are required: pip install fastapi uvicorn"
        ) from e

    # Create FastAPI app
    app = FastAPI(
        title="vLLM Inference Server",
        description="High-throughput LLM inference with vLLM",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Server instance
    server: VLLMServer | None = None

    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.95
        stop: list[str] | None = None

    class CompletionResponse(BaseModel):
        text: str
        model: str

    @app.on_event("startup")
    async def startup():
        nonlocal server
        server = VLLMServer(config)
        await server.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        if server:
            await server.shutdown()

    @app.get("/health")
    async def health():
        if server and server.is_healthy():
            return {"status": "healthy"}
        raise HTTPException(status_code=503, detail="Service unhealthy")

    @app.get("/ready")
    async def ready():
        if server and server.is_ready():
            return {"status": "ready"}
        raise HTTPException(status_code=503, detail="Service not ready")

    @app.get("/stats")
    async def stats():
        if server:
            return server.get_stats()
        raise HTTPException(status_code=503, detail="Server not initialized")

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest):
        if not server or not server.is_ready():
            raise HTTPException(status_code=503, detail="Server not ready")

        try:
            text = await server.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
            return CompletionResponse(
                text=text,
                model=config.model.model_name_or_path,
            )
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Run server
    uvicorn_config = uvicorn.Config(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
    )
    server_instance = uvicorn.Server(uvicorn_config)
    await server_instance.serve()


def main():
    """CLI entrypoint for vLLM server."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to serving configuration YAML",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port",
    )

    args = parser.parse_args()

    # Load config
    config = ServingConfig.from_yaml(args.config)

    # Apply overrides
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.server.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    asyncio.run(run_server(config))


if __name__ == "__main__":
    main()
