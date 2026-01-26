"""Milvus vector database operations."""

from typing import Any
import structlog

from pydantic import BaseModel

logger = structlog.get_logger()


class VectorRecord(BaseModel):
    """Represents a vector record for Milvus."""

    id: str
    vector: list[float]
    text: str
    metadata: dict


class MilvusClient:
    """Client for Milvus vector database operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "documents",
        dimension: int = 384,
    ):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension

        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        logger.info("Connected to Milvus", host=host, port=port)

        # Create or get collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        """Get existing collection or create new one."""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

        if utility.has_collection(self.collection_name):
            logger.info("Using existing collection", name=self.collection_name)
            return Collection(self.collection_name)

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document embeddings for RAG",
        )

        collection = Collection(self.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        collection.create_index("vector", index_params)

        logger.info("Created collection", name=self.collection_name, dimension=self.dimension)
        return collection

    def insert(self, records: list[VectorRecord]) -> list[str]:
        """Insert vector records.

        Args:
            records: List of VectorRecord objects

        Returns:
            List of inserted IDs
        """
        data = [
            [r.id for r in records],
            [r.vector for r in records],
            [r.text for r in records],
            [r.metadata for r in records],
        ]

        self.collection.insert(data)
        self.collection.flush()

        logger.info("Inserted records", count=len(records))
        return [r.id for r in records]

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_expr: Optional filter expression

        Returns:
            List of search results with id, text, score, metadata
        """
        self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["text", "metadata"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata"),
            })

        logger.debug("Search complete", num_results=len(hits))
        return hits

    def delete(self, ids: list[str]) -> int:
        """Delete records by ID.

        Args:
            ids: List of record IDs to delete

        Returns:
            Number of deleted records
        """
        expr = f'id in {ids}'
        self.collection.delete(expr)
        logger.info("Deleted records", count=len(ids))
        return len(ids)

    def count(self) -> int:
        """Get total number of records."""
        return self.collection.num_entities


def create_milvus_client(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "documents",
    dimension: int = 384,
) -> MilvusClient:
    """Create Milvus client.

    Args:
        host: Milvus host
        port: Milvus port
        collection_name: Collection name
        dimension: Vector dimension

    Returns:
        MilvusClient instance
    """
    return MilvusClient(
        host=host,
        port=port,
        collection_name=collection_name,
        dimension=dimension,
    )
