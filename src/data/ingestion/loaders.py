"""Document loaders for various file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import boto3
from pydantic import BaseModel


class Document(BaseModel):
    """Represents a loaded document."""

    content: str
    metadata: dict
    source: str
    doc_type: str


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str) -> Iterator[Document]:
        """Load documents from source."""
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if loader supports the source type."""
        pass


class S3Loader(BaseLoader):
    """Load documents from S3 bucket."""

    def __init__(self, bucket: str | None = None):
        self.s3_client = boto3.client("s3")
        self.bucket = bucket

    def load(self, source: str) -> Iterator[Document]:
        """Load documents from S3 path.

        Args:
            source: S3 path in format s3://bucket/prefix or just prefix if bucket set

        Yields:
            Document objects for each file in the path
        """
        if source.startswith("s3://"):
            parts = source[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.bucket
            prefix = source

        if not bucket:
            raise ValueError("Bucket must be specified either in source or constructor")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read().decode("utf-8", errors="ignore")

                yield Document(
                    content=content,
                    metadata={
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "etag": obj["ETag"],
                    },
                    source=f"s3://{bucket}/{key}",
                    doc_type=Path(key).suffix.lower(),
                )

    def supports(self, source: str) -> bool:
        """Check if source is an S3 path."""
        return source.startswith("s3://") or self.bucket is not None


class LocalFileLoader(BaseLoader):
    """Load documents from local filesystem."""

    def __init__(self, extensions: list[str] | None = None):
        self.extensions = extensions or [".txt", ".md", ".pdf", ".docx", ".html"]

    def load(self, source: str) -> Iterator[Document]:
        """Load documents from local path.

        Args:
            source: Local file or directory path

        Yields:
            Document objects for each file
        """
        path = Path(source)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = [
                f for f in path.rglob("*") if f.is_file() and f.suffix in self.extensions
            ]
        else:
            raise ValueError(f"Source not found: {source}")

        for file_path in files:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            yield Document(
                content=content,
                metadata={
                    "size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime,
                },
                source=str(file_path.absolute()),
                doc_type=file_path.suffix.lower(),
            )

    def supports(self, source: str) -> bool:
        """Check if source is a local path."""
        return Path(source).exists()


def get_loader(source: str) -> BaseLoader:
    """Get appropriate loader for the source.

    Args:
        source: Source path (S3 or local)

    Returns:
        Loader instance that can handle the source
    """
    loaders = [S3Loader(), LocalFileLoader()]

    for loader in loaders:
        if loader.supports(source):
            return loader

    raise ValueError(f"No loader found for source: {source}")
