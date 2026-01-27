"""Kubeflow Pipeline Components for Data Processing.

Lightweight component wrappers around src.data modules for use in KFP pipelines.
"""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output


# Component base image
PIPELINE_IMAGE = "your-registry.com/pipeline-components:latest"


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=["boto3>=1.34.0", "pydantic>=2.5.0"],
)
def ingest_component(
    source_path: str,
    output_documents: Output[Dataset],
) -> NamedTuple("Outputs", [("num_documents", int)]):
    """Load documents from S3 or local path.

    Args:
        source_path: S3 URI or local path to documents
        output_documents: Output dataset with loaded documents

    Returns:
        Number of documents loaded
    """
    import json
    from pathlib import Path

    from src.data.ingestion.loaders import get_loader

    loader = get_loader(source_path)
    documents = []

    for doc in loader.load(source_path):
        documents.append(
            {
                "content": doc.content,
                "source": doc.source,
                "doc_type": doc.doc_type,
                "metadata": doc.metadata,
            }
        )

    # Write to output artifact
    output_path = Path(output_documents.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(documents, indent=2))

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["num_documents"])
    return outputs(len(documents))


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=["unstructured>=0.11.0", "pydantic>=2.5.0"],
)
def parse_component(
    input_documents: Input[Dataset],
    output_parsed: Output[Dataset],
) -> NamedTuple("Outputs", [("num_elements", int)]):
    """Parse documents into structured elements.

    Args:
        input_documents: Dataset with raw documents
        output_parsed: Output dataset with parsed elements

    Returns:
        Total number of parsed elements
    """
    import json
    from pathlib import Path

    from src.data.ingestion.parsers import get_parser

    # Load input
    input_path = Path(input_documents.path)
    documents = json.loads(input_path.read_text())

    parsed_docs = []
    total_elements = 0

    for doc in documents:
        parser = get_parser(doc["doc_type"])
        parsed = parser.parse(
            content=doc["content"],
            doc_type=doc["doc_type"],
            metadata={"source": doc["source"], **doc["metadata"]},
        )

        # Combine elements
        combined_text = "\n\n".join(elem.text for elem in parsed.elements)
        total_elements += len(parsed.elements)

        parsed_docs.append(
            {
                "content": combined_text,
                "source": doc["source"],
                "doc_type": doc["doc_type"],
                "metadata": {**doc["metadata"], **parsed.metadata},
                "num_elements": len(parsed.elements),
            }
        )

    # Write output
    output_path = Path(output_parsed.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parsed_docs, indent=2))

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["num_elements"])
    return outputs(total_elements)


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=["pydantic>=2.5.0"],
)
def chunk_component(
    input_parsed: Input[Dataset],
    output_chunks: Output[Dataset],
    strategy: str = "semantic",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> NamedTuple("Outputs", [("num_chunks", int)]):
    """Split documents into chunks.

    Args:
        input_parsed: Dataset with parsed documents
        output_chunks: Output dataset with chunks
        strategy: Chunking strategy ("semantic" or "fixed")
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Total number of chunks created
    """
    import json
    from pathlib import Path

    from src.data.ingestion.chunkers import get_chunker

    # Load input
    input_path = Path(input_parsed.path)
    documents = json.loads(input_path.read_text())

    # Configure chunker
    kwargs = {}
    if strategy == "fixed":
        kwargs = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    else:
        kwargs = {"max_chunk_size": chunk_size, "min_chunk_size": chunk_overlap}

    chunker = get_chunker(strategy, **kwargs)

    all_chunks = []
    for doc in documents:
        for chunk in chunker.chunk(
            text=doc["content"],
            metadata={
                "source": doc["source"],
                "doc_type": doc["doc_type"],
                **doc["metadata"],
            },
        ):
            all_chunks.append(
                {
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "metadata": chunk.metadata,
                }
            )

    # Write output
    output_path = Path(output_chunks.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_chunks, indent=2))

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["num_chunks"])
    return outputs(len(all_chunks))


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=[
        "sentence-transformers>=2.2.0",
        "torch>=2.1.0",
        "pydantic>=2.5.0",
    ],
)
def embed_component(
    input_chunks: Input[Dataset],
    output_embeddings: Output[Dataset],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> NamedTuple("Outputs", [("num_embeddings", int), ("dimension", int)]):
    """Generate embeddings for chunks.

    Args:
        input_chunks: Dataset with text chunks
        output_embeddings: Output dataset with embeddings
        model_name: Embedding model name
        batch_size: Batch size for embedding

    Returns:
        Number of embeddings and embedding dimension
    """
    import json
    from pathlib import Path

    from src.data.embedding.models import get_embedding_model
    from src.data.ingestion.chunkers import Chunk

    # Load input
    input_path = Path(input_chunks.path)
    chunk_data = json.loads(input_path.read_text())
    chunks = [Chunk(**c) for c in chunk_data]

    # Generate embeddings
    model = get_embedding_model("sentence-transformer", model_name)

    all_results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        embeddings = model.embed(texts)

        for chunk, emb in zip(batch, embeddings):
            all_results.append(
                {
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "embedding": emb.embedding,
                    "model": emb.model,
                    "dimension": emb.dimension,
                }
            )

    # Write output
    output_path = Path(output_embeddings.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["num_embeddings", "dimension"])
    return outputs(len(all_results), model.dimension)


@dsl.component(
    base_image=PIPELINE_IMAGE,
    packages_to_install=["pymilvus>=2.3.0", "pydantic>=2.5.0"],
)
def store_component(
    input_embeddings: Input[Dataset],
    milvus_host: str = "milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "documents",
) -> NamedTuple("Outputs", [("num_stored", int)]):
    """Store embeddings in Milvus.

    Args:
        input_embeddings: Dataset with embeddings
        milvus_host: Milvus host
        milvus_port: Milvus port
        collection_name: Collection name

    Returns:
        Number of vectors stored
    """
    import hashlib
    import json
    from pathlib import Path

    from src.data.storage.milvus import MilvusClient, VectorRecord

    # Load input
    input_path = Path(input_embeddings.path)
    embeddings_data = json.loads(input_path.read_text())

    if not embeddings_data:
        from collections import namedtuple

        outputs = namedtuple("Outputs", ["num_stored"])
        return outputs(0)

    # Get dimension from first embedding
    dimension = embeddings_data[0]["dimension"]

    # Create Milvus client
    client = MilvusClient(
        host=milvus_host,
        port=milvus_port,
        collection_name=collection_name,
        dimension=dimension,
    )

    # Prepare records
    records = []
    for item in embeddings_data:
        content_hash = hashlib.md5(item["text"].encode()).hexdigest()[:16]
        record_id = f"{content_hash}-{item['chunk_index']}"

        records.append(
            VectorRecord(
                id=record_id,
                vector=item["embedding"],
                text=item["text"],
                metadata=item["metadata"],
            )
        )

    # Insert in batches
    batch_size = 1000
    total_stored = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.insert(batch)
        total_stored += len(batch)

    from collections import namedtuple

    outputs = namedtuple("Outputs", ["num_stored"])
    return outputs(total_stored)
