"""Data Pipeline CLI.

Command-line interface for running data pipelines locally or in containers.
"""

import json
import sys
from pathlib import Path

import click
import structlog
import yaml

from src.data.pipeline.pipeline import (
    ChunkStrategy,
    DataPipeline,
    DataPipelineConfig,
    EmbeddingModelType,
)

logger = structlog.get_logger()


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    if not path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content)
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        raise click.ClickException(f"Unsupported config format: {path.suffix}")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Data Pipeline CLI for document ingestion and embedding."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source path (S3 URI or local path)",
)
@click.option(
    "--config",
    "-c",
    "config_file",
    help="Configuration file (YAML/JSON)",
)
@click.option(
    "--chunk-strategy",
    type=click.Choice(["fixed", "semantic"]),
    default="semantic",
    help="Chunking strategy",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Target chunk size",
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Embedding model name",
)
@click.option(
    "--milvus-host",
    default="localhost",
    help="Milvus host",
)
@click.option(
    "--milvus-port",
    type=int,
    default=19530,
    help="Milvus port",
)
@click.option(
    "--collection",
    default="documents",
    help="Milvus collection name",
)
@click.option(
    "--use-ray",
    is_flag=True,
    help="Use Ray for distributed processing",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run (no storage)",
)
@click.pass_context
def run(
    ctx: click.Context,
    source: str,
    config_file: str | None,
    chunk_strategy: str,
    chunk_size: int,
    embedding_model: str,
    milvus_host: str,
    milvus_port: int,
    collection: str,
    use_ray: bool,
    dry_run: bool,
) -> None:
    """Run the full data ingestion pipeline."""
    # Start with config file if provided
    config_dict = {}
    if config_file:
        config_dict = load_config_file(config_file)

    # Override with CLI options
    config_dict.update(
        {
            "source_path": source,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "embedding_model_name": embedding_model,
            "milvus_host": milvus_host,
            "milvus_port": milvus_port,
            "collection_name": collection,
            "use_ray": use_ray,
            "dry_run": dry_run,
        }
    )

    config = DataPipelineConfig(**config_dict)

    click.echo(f"Running pipeline for: {source}")
    click.echo(f"  Chunk strategy: {chunk_strategy}")
    click.echo(f"  Embedding model: {embedding_model}")
    click.echo(f"  Milvus: {milvus_host}:{milvus_port}/{collection}")
    click.echo(f"  Dry run: {dry_run}")
    click.echo()

    pipeline = DataPipeline(config)
    metrics = pipeline.run()

    click.echo()
    click.echo("Pipeline completed:")
    click.echo(f"  Documents loaded: {metrics.documents_loaded}")
    click.echo(f"  Chunks created: {metrics.chunks_created}")
    click.echo(f"  Embeddings generated: {metrics.embeddings_generated}")
    click.echo(f"  Vectors stored: {metrics.vectors_stored}")

    if metrics.errors:
        click.echo(f"  Errors: {len(metrics.errors)}")
        for error in metrics.errors:
            click.echo(f"    - {error}")


@cli.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source path (S3 URI or local path)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output directory for chunks (JSON)",
)
@click.option(
    "--chunk-strategy",
    type=click.Choice(["fixed", "semantic"]),
    default="semantic",
    help="Chunking strategy",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Target chunk size",
)
def ingest(
    source: str,
    output: str,
    chunk_strategy: str,
    chunk_size: int,
) -> None:
    """Ingest and chunk documents (no embedding)."""
    from src.data.ingestion.chunkers import get_chunker
    from src.data.ingestion.loaders import get_loader
    from src.data.ingestion.parsers import get_parser

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Ingesting documents from: {source}")

    loader = get_loader(source)
    chunker = get_chunker(
        chunk_strategy,
        chunk_size=chunk_size if chunk_strategy == "fixed" else None,
        max_chunk_size=chunk_size if chunk_strategy == "semantic" else None,
    )

    total_docs = 0
    total_chunks = 0

    for doc in loader.load(source):
        total_docs += 1

        # Parse document
        parser = get_parser(doc.doc_type)
        parsed = parser.parse(doc.content, doc.doc_type, {"source": doc.source})

        # Chunk
        combined_text = "\n\n".join(elem.text for elem in parsed.elements)
        chunks = list(
            chunker.chunk(combined_text, {"source": doc.source, "doc_type": doc.doc_type})
        )
        total_chunks += len(chunks)

        # Save chunks
        doc_name = Path(doc.source).stem
        chunk_file = output_path / f"{doc_name}_chunks.json"
        chunk_data = [chunk.model_dump() for chunk in chunks]
        chunk_file.write_text(json.dumps(chunk_data, indent=2))

        click.echo(f"  {doc.source}: {len(chunks)} chunks")

    click.echo()
    click.echo(f"Total: {total_docs} documents, {total_chunks} chunks")
    click.echo(f"Output: {output_path}")


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Input chunks file or directory (JSON)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output file for embeddings (JSON)",
)
@click.option(
    "--model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Embedding model",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for embedding",
)
@click.option(
    "--use-ray",
    is_flag=True,
    help="Use Ray for distributed processing",
)
def embed(
    input_path: str,
    output: str,
    model: str,
    batch_size: int,
    use_ray: bool,
) -> None:
    """Generate embeddings for chunked documents."""
    from src.data.embedding.batch import BatchEmbeddingGenerator
    from src.data.embedding.models import get_embedding_model
    from src.data.ingestion.chunkers import Chunk

    input_dir = Path(input_path)
    output_file = Path(output)

    # Load chunks
    chunks = []
    if input_dir.is_file():
        chunk_files = [input_dir]
    else:
        chunk_files = list(input_dir.glob("*_chunks.json"))

    for chunk_file in chunk_files:
        data = json.loads(chunk_file.read_text())
        for item in data:
            chunks.append(Chunk(**item))

    click.echo(f"Loaded {len(chunks)} chunks from {len(chunk_files)} files")

    # Generate embeddings
    embedding_model = get_embedding_model("sentence-transformer", model)
    generator = BatchEmbeddingGenerator(
        model=embedding_model,
        batch_size=batch_size,
        use_ray=use_ray,
    )

    click.echo(f"Generating embeddings with {model}...")
    results = generator.embed_chunks(chunks)

    # Save results
    output_data = []
    for chunk, emb_result in results:
        output_data.append(
            {
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "embedding": emb_result.embedding,
                "model": emb_result.model,
                "dimension": emb_result.dimension,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, indent=2))

    click.echo(f"Saved {len(results)} embeddings to {output_file}")


@cli.command()
def version() -> None:
    """Show version information."""
    click.echo("Data Pipeline CLI v0.1.0")
    click.echo("Part of LLM MLOps Platform")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
