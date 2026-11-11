"""Retrieval orchestration: vector / hybrid recall, optional cross-encoder rerank."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import structlog
from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import CrossEncoderReranker

log = structlog.get_logger()

Mode = Literal["vector", "hybrid"]


@dataclass
class Candidate:
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    metadata: dict[str, Any]
    distance: float | None = None  # cosine distance, set when the vector branch found it
    lexical_rank: float | None = None  # ts_rank_cd, set when the lexical branch found it
    score: float | None = None  # final ordering score (RRF or reranker), informational


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    timings_ms: dict[str, int] = field(default_factory=dict)


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker.from_settings()


_COLUMNS = (Chunk.id, Chunk.document_id, Chunk.chunk_type, Chunk.content, Chunk.metadata_)


def _to_candidate(row, **extra) -> Candidate:
    return Candidate(
        chunk_id=row.id,
        document_id=row.document_id,
        chunk_type=row.chunk_type,
        content=row.content,
        metadata=row.metadata_,
        **extra,
    )


async def vector_search(
    session: AsyncSession, query_vector: list[float], limit: int
) -> list[Candidate]:
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = select(*_COLUMNS, distance.label("distance")).order_by(distance).limit(limit)
    return [_to_candidate(row, distance=row.distance) for row in (await session.execute(stmt)).all()]


async def lexical_search(session: AsyncSession, query: str, limit: int) -> list[Candidate]:
    """Keyword search over ``content_tsv``, ranked by ``ts_rank_cd``.

    ``plainto_tsquery`` ANDs every term, which returns nothing for a long free-text project
    description. The terms are OR-ed instead, and ``ts_rank_cd`` rewards chunks that match
    more of them (and closer together).
    """
    config = get_settings().FTS_CONFIG
    plain = func.plainto_tsquery(cast(config, REGCONFIG), query)
    or_query = func.to_tsquery(
        cast(config, REGCONFIG), func.replace(cast(plain, Text), "&", "|")
    ).cast(TSQUERY)
    rank = func.ts_rank_cd(Chunk.content_tsv, or_query)
    stmt = (
        select(*_COLUMNS, rank.label("rank"))
        .where(Chunk.content_tsv.op("@@")(or_query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [
        _to_candidate(row, lexical_rank=row.rank) for row in (await session.execute(stmt)).all()
    ]


def fuse_candidates(
    vector: list[Candidate], lexical: list[Candidate], *, rrf_k: int, limit: int
) -> list[Candidate]:
    by_id: dict[int, Candidate] = {}
    for candidate in (*vector, *lexical):
        merged = by_id.setdefault(candidate.chunk_id, candidate)
        if merged is not candidate:  # found by both branches: keep both signals
            merged.distance = merged.distance if merged.distance is not None else candidate.distance
            merged.lexical_rank = (
                merged.lexical_rank if merged.lexical_rank is not None else candidate.lexical_rank
            )
    fused = rrf_fuse(
        [[c.chunk_id for c in vector], [c.chunk_id for c in lexical]], k=rrf_k
    )
    out = []
    for chunk_id, score in fused[:limit]:
        by_id[chunk_id].score = score
        out.append(by_id[chunk_id])
    return out


async def retrieve(
    session: AsyncSession,
    embedder: OpenAIEmbedder,
    query: str,
    *,
    mode: Mode | None = None,
    rerank: bool | None = None,
    top_k: int | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalResult:
    """Return the best ``top_k`` chunks for ``query``.

    ``mode`` and ``rerank`` default to ``RETRIEVAL_MODE`` / ``RERANK_ENABLED``. Plain vector
    search is cut straight to ``top_k``; hybrid fusion and reranking recall ``RECALL_K``
    candidates per branch first, then keep the best ``top_k``.
    """
    settings = get_settings()
    mode = mode or settings.RETRIEVAL_MODE
    rerank = settings.RERANK_ENABLED if rerank is None else rerank
    top_k = top_k or settings.RERANK_TOP_N
    # Fusion and reranking both need deep candidate lists; plain vector search does not.
    recall_k = settings.RECALL_K if (rerank or mode == "hybrid") else top_k
    timings: dict[str, int] = {}

    started = time.perf_counter()
    query_vector = await asyncio.to_thread(embedder.embed_one, query)
    timings["embed"] = int((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    vector = await vector_search(session, query_vector, recall_k)
    if mode == "hybrid":
        lexical = await lexical_search(session, query, recall_k)
        candidates = fuse_candidates(vector, lexical, rrf_k=settings.RRF_K, limit=recall_k)
    else:
        candidates = vector
    timings["recall"] = int((time.perf_counter() - started) * 1000)

    if rerank:
        started = time.perf_counter()
        reranker = reranker or get_reranker()
        # The model is CPU-bound and synchronous: keep the event loop free.
        scores = await asyncio.to_thread(reranker.score, query, [c.content for c in candidates])
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.score = score
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        timings["rerank"] = int((time.perf_counter() - started) * 1000)

    log.info("retrieval_completed", mode=mode, rerank=rerank, recall_k=recall_k, timings_ms=timings)
    return RetrievalResult(candidates=candidates[:top_k], timings_ms=timings)
