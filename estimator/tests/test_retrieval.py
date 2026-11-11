"""RRF fusion, hybrid/rerank orchestration and the /search switches, without DB or network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.db.session import get_session
from app.embedding_pipeline.router import get_embedder
from app.main import app
from app.retrieval import pipeline
from app.retrieval.fusion import rrf_fuse
from app.retrieval.pipeline import Candidate, fuse_candidates, retrieve
from evals.retrieval.run import ndcg_at_k, precision_at_k, recall_at_k


def _cand(i: int, **kw) -> Candidate:
    return Candidate(
        chunk_id=i, document_id=1, chunk_type="budget_component", content=f"doc {i}",
        metadata={"chunk_id": f"B::C{i}"}, **kw,
    )


class FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [0.0]


class FakeReranker:
    """Scores documents by their number: higher number = more relevant."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        return [float(d.split()[1]) for d in documents]


@pytest.fixture
def fake_branches(monkeypatch):
    seen: dict[str, int] = {}

    async def vector(session, vec, limit):
        seen["vector_limit"] = limit
        return [_cand(i, distance=i / 10) for i in (1, 2, 3, 4, 5, 6)][:limit]

    async def lexical(session, query, limit):
        seen["lexical_limit"] = limit
        return [_cand(i, lexical_rank=1.0) for i in (6, 5, 9)][:limit]

    monkeypatch.setattr(pipeline, "vector_search", vector)
    monkeypatch.setattr(pipeline, "lexical_search", lexical)
    return seen


def test_rrf_rewards_items_found_by_both_lists() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["c", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "c"  # rank 3 + rank 1 beats rank 1 alone
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_score_is_the_sum_of_reciprocal_ranks() -> None:
    (item, score), = rrf_fuse([["a"], ["a"]], k=10)
    assert item == "a"
    assert score == pytest.approx(2 / 11)


def test_rrf_k_controls_how_much_the_top_rank_dominates() -> None:
    low = dict(rrf_fuse([["a"]], k=0))["a"]
    high = dict(rrf_fuse([["a"]], k=100))["a"]
    assert low > high


def test_rrf_rejects_negative_k() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=-1)


def test_fuse_candidates_keeps_both_signals_for_shared_chunks() -> None:
    fused = fuse_candidates(
        [_cand(1, distance=0.2), _cand(2, distance=0.3)],
        [_cand(2, lexical_rank=0.9), _cand(3, lexical_rank=0.5)],
        rrf_k=60, limit=10,
    )
    assert [c.chunk_id for c in fused][0] == 2
    assert fused[0].distance == 0.3 and fused[0].lexical_rank == 0.9


async def test_vector_without_rerank_takes_top_k_straight_from_recall(fake_branches) -> None:
    result = await retrieve(None, FakeEmbedder(), "q", mode="vector", rerank=False, top_k=3)
    assert [c.chunk_id for c in result.candidates] == [1, 2, 3]
    assert fake_branches == {"vector_limit": 3}  # no lexical branch, no wide recall
    assert "rerank" not in result.timings_ms


async def test_hybrid_fuses_vector_and_lexical(fake_branches) -> None:
    result = await retrieve(None, FakeEmbedder(), "q", mode="hybrid", rerank=False, top_k=3)
    ids = [c.chunk_id for c in result.candidates]
    assert fake_branches["vector_limit"] == get_settings().RECALL_K  # deep lists for fusion
    assert set(ids[:2]) == {5, 6}  # found by both branches, so they lead the fused ranking
    assert len(ids) == 3


async def test_rerank_recalls_wide_then_keeps_top_k(fake_branches) -> None:
    reranker = FakeReranker()
    result = await retrieve(
        None, FakeEmbedder(), "q", mode="vector", rerank=True, top_k=2, reranker=reranker
    )
    assert fake_branches["vector_limit"] == get_settings().RECALL_K
    assert reranker.calls == 1
    assert [c.chunk_id for c in result.candidates] == [6, 5]  # reordered by the cross-encoder
    assert "rerank" in result.timings_ms


async def test_rerank_off_never_touches_the_reranker(fake_branches) -> None:
    reranker = FakeReranker()
    await retrieve(None, FakeEmbedder(), "q", rerank=False, reranker=reranker)
    assert reranker.calls == 0


async def test_rerank_defaults_come_from_settings(fake_branches, monkeypatch) -> None:
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    get_settings.cache_clear()
    reranker = FakeReranker()
    try:
        result = await retrieve(None, FakeEmbedder(), "q", reranker=reranker)
    finally:
        get_settings.cache_clear()
    assert reranker.calls == 1 and "rerank" in result.timings_ms
    assert fake_branches["lexical_limit"] == get_settings().RECALL_K


def test_search_endpoint_exposes_mode_and_rerank(monkeypatch) -> None:
    captured = {}

    async def fake_retrieve(session, embedder, query, *, mode, rerank, top_k):
        captured.update(mode=mode, rerank=rerank, top_k=top_k)
        return pipeline.RetrievalResult(
            candidates=[_cand(1, distance=0.1, score=0.7)], timings_ms={"embed": 1}
        )

    monkeypatch.setattr("app.embedding_pipeline.router.retrieve", fake_retrieve)

    async def _session():
        yield None

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/search", json={"query": "q", "k": 3, "mode": "hybrid", "rerank": True}
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    body = response.json()
    assert captured == {"mode": "hybrid", "rerank": True, "top_k": 3}
    assert body["mode"] == "hybrid" and body["reranked"] is True
    assert body["results"][0]["score"] == 0.7


def test_search_endpoint_rejects_unknown_mode() -> None:
    with TestClient(app) as client:
        assert client.post("/search", json={"query": "q", "mode": "magic"}).status_code == 422


def test_golden_metrics() -> None:
    relevant = {"a", "b"}
    assert precision_at_k(["a", "x", "b", "y", "z"], relevant) == pytest.approx(0.4)
    assert recall_at_k(["a", "x", "b", "y", "z"], relevant) == 1.0
    assert ndcg_at_k(["a", "b", "x", "y", "z"], relevant) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z", "a", "b"], relevant) < 1.0
