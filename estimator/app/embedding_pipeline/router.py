"""POST /embeddings/ingest: budgets in, vectorised chunks + stats out."""

from __future__ import annotations

from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI, OpenAIError

from app.config import get_settings
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    api_key = get_settings().OPENAI_API_KEY
    if not api_key:
        log.error("embeddings_unavailable", reason="OPENAI_API_KEY is not set")
        raise HTTPException(status_code=500, detail="Embedding service is not available")
    return OpenAIEmbedder(OpenAI(api_key=api_key))


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    chunks = chunker.chunk(request.budgets)
    try:
        embedded = embedder.embed_many(chunks)
    except OpenAIError as exc:
        # Generic message to the client; the detail stays in the logs.
        log.error("embedding_failed", error_type=type(exc).__name__, error=str(exc)[:400])
        raise HTTPException(status_code=500, detail="Embedding generation failed") from exc

    total_tokens = sum(chunk.token_count for chunk in embedded)
    stats = {
        "total_budgets": len(request.budgets),
        "total_chunks": len(embedded),
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost_usd(total_tokens),
    }
    log.info("ingest_completed", **stats)
    return IngestResponse(chunks=embedded, stats=stats)
