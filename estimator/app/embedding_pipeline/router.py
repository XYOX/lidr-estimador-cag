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
    try:
        query_vector = await asyncio.to_thread(embedder.embed_one, request.query)
    except OpenAIError as exc:
        log.error("embedding_failed", error_type=type(exc).__name__, error=str(exc)[:400])
        raise HTTPException(status_code=500, detail="Embedding generation failed") from exc

    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_type,
            Chunk.content,
            Chunk.metadata_,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(request.k)
    )
    rows = (await session.execute(stmt)).all()

    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=int((time.perf_counter() - started) * 1000),
        results=[
            SearchResult(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                distance=row.distance,
                metadata=row.metadata_,
            )
            for row in rows
        ],
    )
