"""OpenAI embedder: text-embedding-3-small at its default 1536 dimensions, batched."""

from __future__ import annotations

import time
from collections.abc import Callable

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.chunker import EMBEDDING_MODEL, count_tokens
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

# USD per 1M input tokens for text-embedding-3-small. OpenAI changes prices over time: update here.
PRICE_PER_MILLION_TOKENS_USD = 0.02
BATCH_SIZE = 100
# Waits before each retry after a RateLimitError: 3 retries, 4 attempts in total.
RETRY_DELAYS_S = (1, 2, 4)


def estimate_cost_usd(total_tokens: int) -> float:
    return round(total_tokens * PRICE_PER_MILLION_TOKENS_USD / 1_000_000, 8)


class OpenAIEmbedder:
    def __init__(
        self,
        client: OpenAI,
        *,
        batch_size: int = BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self._sleep = sleep

    def embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text], tokens=count_tokens(text))[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self._embed_batch(
                [chunk.text for chunk in batch], tokens=sum(chunk.token_count for chunk in batch)
            )
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _embed_batch(self, texts: list[str], *, tokens: int) -> list[list[float]]:
        """One API call for the whole batch, retrying only on rate limits (1s, 2s, 4s)."""
        started = time.perf_counter()
        for delay in (*RETRY_DELAYS_S, None):
            try:
                response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
                break
            except RateLimitError:
                if delay is None:
                    raise
                log.warning("embedding_rate_limited", retry_in_s=delay, chunks=len(texts))
                self._sleep(delay)

        log.info(
            "embedding_batch_processed",
            chunks=len(texts),
            tokens=tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        # The API returns one item per input with its position in ``index``.
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
