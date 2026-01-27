"""Data Ingestion Pipeline for Kubeflow Pipelines.

End-to-end pipeline for document ingestion, parsing, chunking,
embedding generation, and vector storage.
"""

from kfp import dsl
from kfp.dsl import PipelineTask

from pipelines.data.components import (
    chunk_component,
    embed_component,
    ingest_component,
    parse_component,
    store_component,
)


@dsl.pipeline(
    name="data-ingestion-pipeline",
    description="End-to-end document ingestion and embedding pipeline for RAG",
)
def data_ingestion_pipeline(
    source_path: str,
    chunk_strategy: str = "semantic",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    milvus_host: str = "milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "documents",
) -> None:
    """Data ingestion pipeline.

    Args:
        source_path: S3 URI or path to source documents
        chunk_strategy: Chunking strategy ("semantic" or "fixed")
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        embedding_model: Embedding model name
        embedding_batch_size: Batch size for embedding
        milvus_host: Milvus host address
        milvus_port: Milvus port
        collection_name: Milvus collection name
    """
    # Step 1: Ingest documents
    ingest_task = ingest_component(source_path=source_path)
    ingest_task.set_display_name("Ingest Documents")
    ingest_task.set_caching_options(enable_caching=False)

    # Step 2: Parse documents
    parse_task = parse_component(
        input_documents=ingest_task.outputs["output_documents"],
    )
    parse_task.set_display_name("Parse Documents")
    parse_task.after(ingest_task)

    # Step 3: Chunk documents
    chunk_task = chunk_component(
        input_parsed=parse_task.outputs["output_parsed"],
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_task.set_display_name("Chunk Documents")
    chunk_task.after(parse_task)

    # Step 4: Generate embeddings
    embed_task = embed_component(
        input_chunks=chunk_task.outputs["output_chunks"],
        model_name=embedding_model,
        batch_size=embedding_batch_size,
    )
    embed_task.set_display_name("Generate Embeddings")
    embed_task.after(chunk_task)

    # Request GPU for embedding
    embed_task.set_accelerator_type("nvidia.com/gpu")
    embed_task.set_accelerator_limit(1)

    # Step 5: Store in Milvus
    store_task = store_component(
        input_embeddings=embed_task.outputs["output_embeddings"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )
    store_task.set_display_name("Store Vectors")
    store_task.after(embed_task)


@dsl.pipeline(
    name="embedding-only-pipeline",
    description="Embedding-only pipeline for pre-chunked documents",
)
def embedding_pipeline(
    chunks_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 32,
    milvus_host: str = "milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "documents",
) -> None:
    """Embedding-only pipeline for pre-chunked documents.

    Args:
        chunks_path: Path to pre-chunked documents (JSON)
        embedding_model: Embedding model name
        embedding_batch_size: Batch size for embedding
        milvus_host: Milvus host address
        milvus_port: Milvus port
        collection_name: Milvus collection name
    """
    # Note: This would need a custom component to load pre-chunked data
    # For now, this shows the pattern for a simpler pipeline

    # Compile-time import to avoid issues
    from kfp.dsl import Artifact, Input, Output

    @dsl.component(base_image="python:3.10-slim")
    def load_chunks(
        chunks_path: str,
        output_chunks: Output[dsl.Dataset],
    ) -> int:
        """Load pre-chunked documents."""
        import json
        from pathlib import Path

        import boto3

        # Handle S3 or local path
        if chunks_path.startswith("s3://"):
            parts = chunks_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            chunks = json.loads(response["Body"].read().decode())
        else:
            chunks = json.loads(Path(chunks_path).read_text())

        output_path = Path(output_chunks.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(chunks, indent=2))

        return len(chunks)

    # Load chunks
    load_task = load_chunks(chunks_path=chunks_path)
    load_task.set_display_name("Load Chunks")

    # Generate embeddings
    embed_task = embed_component(
        input_chunks=load_task.outputs["output_chunks"],
        model_name=embedding_model,
        batch_size=embedding_batch_size,
    )
    embed_task.set_display_name("Generate Embeddings")
    embed_task.after(load_task)

    # Store in Milvus
    store_task = store_component(
        input_embeddings=embed_task.outputs["output_embeddings"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )
    store_task.set_display_name("Store Vectors")
    store_task.after(embed_task)


def compile_pipeline(
    output_path: str = "data_ingestion_pipeline.yaml",
) -> None:
    """Compile the pipeline to YAML.

    Args:
        output_path: Output path for compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=data_ingestion_pipeline,
        package_path=output_path,
    )
    print(f"Pipeline compiled to: {output_path}")


if __name__ == "__main__":
    compile_pipeline()
