"""Embedding pipeline: structural chunking of historical budgets + OpenAI embeddings.

Chunks and their vectors are persisted in PostgreSQL + pgvector (see ``app.db``).
"""

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder

__all__ = ["JSONStructuralChunker", "OpenAIEmbedder"]
