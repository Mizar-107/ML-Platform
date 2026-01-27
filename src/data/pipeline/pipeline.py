"""Data Pipeline Orchestrator.

End-to-end pipeline for document ingestion, chunking, embedding generation,
and vector storage for RAG applications.
"""

import hashlib
import uuid
from enum import Enum
from pathlib import Path
from typing import Iterator

import structlog
from pydantic import BaseModel, Field

from src.data.embedding.batch import BatchEmbeddingGenerator
from src.data.embedding.models import get_embedding_model
from src.data.ingestion.chunkers import Chunk, get_chunker
from src.data.ingestion.loaders import Document, get_loader
from src.data.ingestion.parsers import get_parser
from src.data.storage.milvus import MilvusClient, VectorRecord

logger = structlog.get_logger()


class ChunkStrategy(str, Enum):
    """Chunking strategy options."""

    FIXED = "fixed"
    SEMANTIC = "semantic"


class EmbeddingModelType(str, Enum):
    """Embedding model type options."""

    SENTENCE_TRANSFORMER = "sentence-transformer"
    HUGGINGFACE = "huggingface"


class DataPipelineConfig(BaseModel):
    """Configuration for data pipeline execution."""

    # Source configuration
    source_path: str = Field(..., description="S3 path or local directory")
    source_bucket: str | None = Field(None, description="S3 bucket (if not in path)")

    # Chunking configuration
    chunk_strategy: ChunkStrategy = Field(
        ChunkStrategy.SEMANTIC, description="Chunking strategy"
    )
    chunk_size: int = Field(512, description="Target chunk size in characters")
    chunk_overlap: int = Field(50, description="Overlap between chunks")

    # Embedding configuration
    embedding_model_type: EmbeddingModelType = Field(
        EmbeddingModelType.SENTENCE_TRANSFORMER, description="Embedding model type"
    )
    embedding_model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", description="Model name/path"
    )
    embedding_batch_size: int = Field(32, description="Batch size for embedding")
    use_ray: bool = Field(False, description="Use Ray for distributed embedding")
    ray_num_workers: int = Field(4, description="Number of Ray workers")

    # Storage configuration
    milvus_host: str = Field("localhost", description="Milvus host")
    milvus_port: int = Field(19530, description="Milvus port")
    collection_name: str = Field("documents", description="Milvus collection name")

    # Pipeline options
    dry_run: bool = Field(False, description="Dry run mode (no storage)")
    skip_existing: bool = Field(True, description="Skip already processed documents")


class PipelineMetrics(BaseModel):
    """Metrics from pipeline execution."""

    documents_loaded: int = 0
    documents_parsed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_stored: int = 0
    errors: list[str] = Field(default_factory=list)


class DataPipeline:
    """End-to-end data pipeline for RAG document processing."""

    def __init__(self, config: DataPipelineConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.metrics = PipelineMetrics()

        # Initialize components lazily
        self._loader = None
        self._chunker = None
        self._embedding_generator = None
        self._milvus_client = None

    @property
    def loader(self):
        """Get document loader."""
        if self._loader is None:
            self._loader = get_loader(self.config.source_path)
        return self._loader

    @property
    def chunker(self):
        """Get text chunker."""
        if self._chunker is None:
            kwargs = {}
            if self.config.chunk_strategy == ChunkStrategy.FIXED:
                kwargs = {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                }
            else:
                kwargs = {
                    "max_chunk_size": self.config.chunk_size,
                    "min_chunk_size": self.config.chunk_overlap,
                }
            self._chunker = get_chunker(self.config.chunk_strategy.value, **kwargs)
        return self._chunker

    @property
    def embedding_generator(self):
        """Get embedding generator."""
        if self._embedding_generator is None:
            model = get_embedding_model(
                model_type=self.config.embedding_model_type.value,
                model_name=self.config.embedding_model_name,
            )
            self._embedding_generator = BatchEmbeddingGenerator(
                model=model,
                batch_size=self.config.embedding_batch_size,
                use_ray=self.config.use_ray,
                num_workers=self.config.ray_num_workers,
            )
        return self._embedding_generator

    @property
    def milvus_client(self):
        """Get Milvus client."""
        if self._milvus_client is None and not self.config.dry_run:
            model = get_embedding_model(
                model_type=self.config.embedding_model_type.value,
                model_name=self.config.embedding_model_name,
            )
            self._milvus_client = MilvusClient(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.collection_name,
                dimension=model.dimension,
            )
        return self._milvus_client

    def run(self) -> PipelineMetrics:
        """Execute full pipeline.

        Returns:
            Pipeline execution metrics
        """
        logger.info(
            "Starting data pipeline",
            source=self.config.source_path,
            dry_run=self.config.dry_run,
        )

        try:
            # Step 1: Ingest documents
            documents = list(self.ingest())

            # Step 2: Chunk documents
            all_chunks = []
            for doc in documents:
                chunks = list(self.chunk(doc))
                all_chunks.extend(chunks)

            # Step 3: Generate embeddings
            chunk_embeddings = self.embed(all_chunks)

            # Step 4: Store vectors
            if not self.config.dry_run:
                self.store(chunk_embeddings)

            logger.info(
                "Pipeline complete",
                documents=self.metrics.documents_loaded,
                chunks=self.metrics.chunks_created,
                vectors=self.metrics.vectors_stored,
            )

        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            self.metrics.errors.append(str(e))
            raise

        return self.metrics

    def ingest(self) -> Iterator[Document]:
        """Load and parse documents from source.

        Yields:
            Parsed Document objects
        """
        logger.info("Ingesting documents", source=self.config.source_path)

        for doc in self.loader.load(self.config.source_path):
            self.metrics.documents_loaded += 1

            # Parse document
            parser = get_parser(doc.doc_type)
            parsed = parser.parse(
                content=doc.content,
                doc_type=doc.doc_type,
                metadata={"source": doc.source, **doc.metadata},
            )
            self.metrics.documents_parsed += 1

            # Combine parsed elements back into document content
            combined_text = "\n\n".join(elem.text for elem in parsed.elements)
            doc.content = combined_text
            doc.metadata.update(parsed.metadata)

            logger.debug(
                "Ingested document",
                source=doc.source,
                elements=len(parsed.elements),
            )

            yield doc

    def chunk(self, document: Document) -> Iterator[Chunk]:
        """Chunk a document into segments.

        Args:
            document: Document to chunk

        Yields:
            Chunk objects
        """
        for chunk in self.chunker.chunk(
            text=document.content,
            metadata={
                "source": document.source,
                "doc_type": document.doc_type,
                **document.metadata,
            },
        ):
            self.metrics.chunks_created += 1
            yield chunk

    def embed(
        self, chunks: list[Chunk]
    ) -> list[tuple[Chunk, list[float]]]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of chunks to embed

        Returns:
            List of (Chunk, embedding) tuples
        """
        logger.info("Generating embeddings", num_chunks=len(chunks))

        results = self.embedding_generator.embed_chunks(chunks)
        self.metrics.embeddings_generated = len(results)

        # Convert to (chunk, embedding vector) format
        return [(chunk, emb_result.embedding) for chunk, emb_result in results]

    def store(
        self, chunk_embeddings: list[tuple[Chunk, list[float]]]
    ) -> list[str]:
        """Store chunk embeddings in Milvus.

        Args:
            chunk_embeddings: List of (Chunk, embedding) tuples

        Returns:
            List of stored record IDs
        """
        if self.config.dry_run:
            logger.info("Dry run - skipping storage")
            return []

        logger.info("Storing vectors", num_vectors=len(chunk_embeddings))

        records = []
        for chunk, embedding in chunk_embeddings:
            # Generate deterministic ID from content
            content_hash = hashlib.md5(chunk.text.encode()).hexdigest()[:16]
            record_id = f"{content_hash}-{chunk.chunk_index}"

            records.append(
                VectorRecord(
                    id=record_id,
                    vector=embedding,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )

        ids = self.milvus_client.insert(records)
        self.metrics.vectors_stored = len(ids)

        return ids

    def _generate_doc_id(self, source: str) -> str:
        """Generate document ID from source path."""
        return hashlib.md5(source.encode()).hexdigest()


def create_pipeline(config: DataPipelineConfig | dict) -> DataPipeline:
    """Create a data pipeline instance.

    Args:
        config: Pipeline configuration (dict or DataPipelineConfig)

    Returns:
        DataPipeline instance
    """
    if isinstance(config, dict):
        config = DataPipelineConfig(**config)
    return DataPipeline(config)
