"""Embedding pipeline: structural chunking of historical budgets + OpenAI embeddings.

Vectors are generated in memory and returned over HTTP; nothing is persisted yet
(pgvector arrives in Session 08).
"""

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder

__all__ = ["JSONStructuralChunker", "OpenAIEmbedder"]
