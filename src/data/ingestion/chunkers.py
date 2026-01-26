"""Text chunking strategies for RAG."""

from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a text chunk."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size chunks with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text into fixed-size chunks.

        Args:
            text: Text to chunk
            metadata: Additional metadata to include

        Yields:
            Chunk objects
        """
        metadata = metadata or {}

        # Split by separator
        words = text.split(self.separator)
        current_chunk: list[str] = []
        current_length = 0
        chunk_index = 0
        start_char = 0

        for word in words:
            word_length = len(word) + 1  # +1 for separator

            if current_length + word_length > self.chunk_size and current_chunk:
                # Emit current chunk
                chunk_text = self.separator.join(current_chunk)
                yield Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata=metadata,
                )

                # Calculate overlap for next chunk
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 <= self.chunk_overlap:
                        overlap_words.insert(0, w)
                        overlap_length += len(w) + 1
                    else:
                        break

                start_char = start_char + len(chunk_text) - overlap_length
                current_chunk = overlap_words
                current_length = overlap_length
                chunk_index += 1

            current_chunk.append(word)
            current_length += word_length

        # Emit final chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            yield Chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata=metadata,
            )


class SemanticChunker(BaseChunker):
    """Split text at semantic boundaries (sentences, paragraphs)."""

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        respect_sentence_boundary: bool = True,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundary = respect_sentence_boundary

    def chunk(self, text: str, metadata: dict | None = None) -> Iterator[Chunk]:
        """Split text at semantic boundaries.

        Args:
            text: Text to chunk
            metadata: Additional metadata to include

        Yields:
            Chunk objects
        """
        import re

        metadata = metadata or {}

        # Split by paragraphs first
        paragraphs = re.split(r"\n\n+", text)

        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max size
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Emit current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    yield Chunk(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(current_chunk),
                        metadata=metadata,
                    )
                    chunk_index += 1
                    start_char += len(current_chunk) + 2

                # If paragraph itself is too large, split by sentences
                if len(para) > self.max_chunk_size and self.respect_sentence_boundary:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                            if current_chunk:
                                yield Chunk(
                                    text=current_chunk.strip(),
                                    chunk_index=chunk_index,
                                    start_char=start_char,
                                    end_char=start_char + len(current_chunk),
                                    metadata=metadata,
                                )
                                chunk_index += 1
                                start_char += len(current_chunk) + 1
                            current_chunk = sentence
                        else:
                            current_chunk = (
                                f"{current_chunk} {sentence}".strip()
                                if current_chunk
                                else sentence
                            )
                else:
                    current_chunk = para
            else:
                current_chunk = (
                    f"{current_chunk}\n\n{para}" if current_chunk else para
                )

        # Emit final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            yield Chunk(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata=metadata,
            )


def get_chunker(strategy: str = "semantic", **kwargs) -> BaseChunker:
    """Get chunker by strategy name.

    Args:
        strategy: Chunking strategy ("fixed", "semantic")
        **kwargs: Additional arguments for the chunker

    Returns:
        Chunker instance
    """
    chunkers = {
        "fixed": FixedSizeChunker,
        "semantic": SemanticChunker,
    }

    chunker_class = chunkers.get(strategy)
    if not chunker_class:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunker_class(**kwargs)
