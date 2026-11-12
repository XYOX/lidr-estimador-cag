"""POST /embeddings/ingest (persist budget + chunks) and POST /search (semantic search)."""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI, OpenAIError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import EMBEDDING_DIMENSION, Chunk, Document
from app.db.session import get_session
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.pipeline import retrieve
from app.embedding_pipeline.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

log = structlog.get_logger()

router = APIRouter(tags=["embeddings"])

CHUNK_TYPE = "budget_component"


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


@router.post("/embeddings/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder = Depends(get_embedder),
):
    started = time.perf_counter()

    existing_id = await session.scalar(
        select(Document.id).where(Document.source_path == request.source_path)
    )
    if existing_id is not None:
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": existing_id},
        )

    # Single transaction: if the embedder fails nothing is committed, so there are no
    # orphan documents without chunks.
    try:
        document = Document(
            source_path=request.source_path,
            document_type=request.document_type,
            metadata_={
                "budget_id": request.content.budget_id,
                "client_sector": request.content.client_metadata.sector,
                "year": request.content.year,
            },
        )
        session.add(document)
        await session.flush()  # assigns document.id

        chunks = chunker.chunk([request.content])
        # The embedder is synchronous (OpenAI sync client): keep the event loop free.
        try:
            embedded = await asyncio.to_thread(embedder.embed_many, chunks)
        except OpenAIError as exc:
            log.error("embedding_failed", error_type=type(exc).__name__, error=str(exc)[:400])
            raise HTTPException(status_code=500, detail="Embedding generation failed") from exc

        session.add_all(
            Chunk(
                document_id=document.id,
                chunk_type=CHUNK_TYPE,
                content=item.text,
                embedding=item.embedding,
                metadata_={
                    **item.metadata,
                    "chunk_id": item.chunk_id,
                    "token_count": item.token_count,
                },
            )
            for item in embedded
        )
        await session.commit()
    except BaseException:
        await session.rollback()
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "ingest_completed",
        document_id=document.id,
        chunks_created=len(embedded),
        ingestion_time_ms=elapsed_ms,
    )
    return IngestResponse(
        document_id=document.id,
        chunks_created=len(embedded),
        embedding_dimension=EMBEDDING_DIMENSION,
        ingestion_time_ms=elapsed_ms,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> SearchResponse:
    started = time.perf_counter()
    settings = get_settings()
    mode = request.mode or settings.RETRIEVAL_MODE
    rerank = settings.RERANK_ENABLED if request.rerank is None else request.rerank
    try:
        result = await retrieve(
            session, embedder, request.query, mode=mode, rerank=rerank, top_k=request.k
        )
    except OpenAIError as exc:
        log.error("embedding_failed", error_type=type(exc).__name__, error=str(exc)[:400])
        raise HTTPException(status_code=500, detail="Embedding generation failed") from exc

    # Plain vector search has no fused/reranked score: the distance is the ordering signal.
    scored = mode == "hybrid" or rerank
    return SearchResponse(
        query=request.query,
        k=request.k,
        mode=mode,
        reranked=rerank,
        search_time_ms=int((time.perf_counter() - started) * 1000),
        timings_ms=result.timings_ms,
        results=[
            SearchResult(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                chunk_type=c.chunk_type,
                content=c.content,
                distance=c.distance,
                score=c.score if scored else None,
                metadata=c.metadata,
            )
            for c in result.candidates
        ],
    )
