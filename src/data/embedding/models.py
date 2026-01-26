"""Embedding model wrappers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Represents embedding result."""

    embedding: list[float]
    model: str
    dimension: int

    class Config:
        arbitrary_types_allowed = True


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embedding for a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    """Embedding model using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        results = []
        for emb in embeddings:
            results.append(
                EmbeddingResult(
                    embedding=emb.tolist(),
                    model=self.model_name,
                    dimension=self._dimension,
                )
            )

        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            EmbeddingResult object
        """
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class HuggingFaceModel(BaseEmbeddingModel):
    """Embedding model using HuggingFace transformers directly."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get dimension from model config
        self._dimension = self.model.config.hidden_size

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Apply mean pooling to model output."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        import torch
        import torch.nn.functional as F

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)

        embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        results = []
        for emb in embeddings.cpu().numpy():
            results.append(
                EmbeddingResult(
                    embedding=emb.tolist(),
                    model=self.model_name,
                    dimension=self._dimension,
                )
            )

        return results

    def embed_query(self, query: str) -> EmbeddingResult:
        """Generate embedding for a single query."""
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


def get_embedding_model(
    model_type: str = "sentence-transformer",
    model_name: str | None = None,
    **kwargs,
) -> BaseEmbeddingModel:
    """Get embedding model by type.

    Args:
        model_type: Model type ("sentence-transformer", "huggingface")
        model_name: Model name/path
        **kwargs: Additional arguments

    Returns:
        Embedding model instance
    """
    if model_type == "sentence-transformer":
        name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerModel(model_name=name, **kwargs)
    elif model_type == "huggingface":
        name = model_name or "BAAI/bge-small-en-v1.5"
        return HuggingFaceModel(model_name=name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
