"""Batch embedding generation with Ray."""

from typing import Iterator
import structlog

from src.data.embedding.models import BaseEmbeddingModel, EmbeddingResult, get_embedding_model
from src.data.ingestion.chunkers import Chunk

logger = structlog.get_logger()


class BatchEmbeddingGenerator:
    """Generate embeddings in batches, optionally using Ray for distribution."""

    def __init__(
        self,
        model: BaseEmbeddingModel | None = None,
        batch_size: int = 32,
        use_ray: bool = False,
        num_workers: int = 4,
    ):
        self.model = model or get_embedding_model()
        self.batch_size = batch_size
        self.use_ray = use_ray
        self.num_workers = num_workers

    def embed_chunks(
        self,
        chunks: Iterator[Chunk] | list[Chunk],
    ) -> list[tuple[Chunk, EmbeddingResult]]:
        """Generate embeddings for chunks.

        Args:
            chunks: Iterator or list of Chunk objects

        Returns:
            List of (Chunk, EmbeddingResult) tuples
        """
        if self.use_ray:
            return self._embed_with_ray(list(chunks))
        else:
            return self._embed_sequential(list(chunks))

    def _embed_sequential(
        self,
        chunks: list[Chunk],
    ) -> list[tuple[Chunk, EmbeddingResult]]:
        """Generate embeddings sequentially.

        Args:
            chunks: List of chunks

        Returns:
            List of (Chunk, EmbeddingResult) tuples
        """
        results = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [chunk.text for chunk in batch]

            logger.info(
                "Embedding batch",
                batch_start=i,
                batch_size=len(batch),
                total=total_chunks,
            )

            embeddings = self.model.embed(texts)

            for chunk, embedding in zip(batch, embeddings):
                results.append((chunk, embedding))

        logger.info("Embedding complete", total_chunks=total_chunks)
        return results

    def _embed_with_ray(
        self,
        chunks: list[Chunk],
    ) -> list[tuple[Chunk, EmbeddingResult]]:
        """Generate embeddings using Ray for distribution.

        Args:
            chunks: List of chunks

        Returns:
            List of (Chunk, EmbeddingResult) tuples
        """
        import ray

        if not ray.is_initialized():
            ray.init()

        @ray.remote
        class EmbeddingWorker:
            def __init__(self, model_type: str, model_name: str):
                self.model = get_embedding_model(model_type, model_name)

            def embed_batch(
                self, chunks: list[dict]
            ) -> list[tuple[dict, dict]]:
                texts = [c["text"] for c in chunks]
                embeddings = self.model.embed(texts)
                return [
                    (c, {"embedding": e.embedding, "model": e.model, "dimension": e.dimension})
                    for c, e in zip(chunks, embeddings)
                ]

        # Create workers
        workers = [
            EmbeddingWorker.remote("sentence-transformer", None)
            for _ in range(self.num_workers)
        ]

        # Distribute work
        chunk_dicts = [chunk.model_dump() for chunk in chunks]
        batches = [
            chunk_dicts[i : i + self.batch_size]
            for i in range(0, len(chunk_dicts), self.batch_size)
        ]

        # Round-robin assignment
        futures = []
        for i, batch in enumerate(batches):
            worker = workers[i % self.num_workers]
            futures.append(worker.embed_batch.remote(batch))

        # Collect results
        all_results = ray.get(futures)

        results = []
        for batch_results in all_results:
            for chunk_dict, emb_dict in batch_results:
                chunk = Chunk(**chunk_dict)
                embedding = EmbeddingResult(**emb_dict)
                results.append((chunk, embedding))

        return results


def create_batch_generator(
    batch_size: int = 32,
    use_ray: bool = False,
    **kwargs,
) -> BatchEmbeddingGenerator:
    """Create batch embedding generator.

    Args:
        batch_size: Number of items per batch
        use_ray: Whether to use Ray for distribution
        **kwargs: Additional arguments

    Returns:
        BatchEmbeddingGenerator instance
    """
    return BatchEmbeddingGenerator(
        batch_size=batch_size,
        use_ray=use_ray,
        **kwargs,
    )
