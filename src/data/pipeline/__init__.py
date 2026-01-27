"""Data Pipeline Module.

Provides end-to-end orchestration for data ingestion, chunking, embedding,
and vector storage.
"""

from src.data.pipeline.pipeline import DataPipeline, DataPipelineConfig

__all__ = ["DataPipeline", "DataPipelineConfig"]
