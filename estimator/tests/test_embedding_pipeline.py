"""Chunker, embedder (batching + retries) and POST /embeddings/ingest, all without the network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.embedding_pipeline.chunker import JSONStructuralChunker, count_tokens
from app.embedding_pipeline.embedder import (
    BATCH_SIZE,
    OpenAIEmbedder,
    estimate_cost_usd,
)
from app.embedding_pipeline.router import get_embedder
from app.db.session import get_session
from app.embedding_pipeline.schemas import Budget, Chunk
from app.main import app

SAMPLE = Path(__file__).parent.parent / "data" / "budgets_sample.json"


def _budgets() -> list[Budget]:
    return [Budget.model_validate(b) for b in json.loads(SAMPLE.read_text())["budgets"]]


def _rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)


class FakeEmbeddings:
    """``client.embeddings`` double: the vector of each input is [len(text), index-in-call]."""

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.failures = list(failures or [])

    def create(self, *, model: str, input: list[str]):
        assert model == "text-embedding-3-small"
        self.calls.append(input)
        if self.failures:
            raise self.failures.pop(0)
        # Returned reversed to prove the embedder restores the input order from ``index``.
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), float(i)]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def _embedder(fake: FakeEmbeddings, sleeps: list[float] | None = None) -> OpenAIEmbedder:
    sleeps = sleeps if sleeps is not None else []
    return OpenAIEmbedder(SimpleNamespace(embeddings=fake), sleep=sleeps.append)


def _chunk(i: int) -> Chunk:
    text = f"chunk number {i}"
    return Chunk(chunk_id=f"B::{i}", text=text, metadata={}, token_count=count_tokens(text))


# ---- chunker ----


def test_one_chunk_per_component_with_traceable_ids() -> None:
    budgets = _budgets()
    chunks = JSONStructuralChunker().chunk(budgets)

    assert len(chunks) == sum(len(b.components) for b in budgets) == 24
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert "BUD-2024-014::AUTH-001" in {c.chunk_id for c in chunks}


def test_chunk_text_prepends_parent_context() -> None:
    chunk = next(c for c in JSONStructuralChunker().chunk(_budgets()) if c.chunk_id == "BUD-2024-014::AUTH-001")

    assert chunk.text.startswith(
        "[Project: Mobile banking app with instant transfers and card management]\n"
        "[Client sector: fintech | Year: 2024 | Main tech: Flutter]\n\n"
        "Component: Authentication backend\n"
    )
    assert "Tech stack: Node.js, PostgreSQL, Keycloak" in chunk.text
    assert chunk.text.endswith("Complexity: high\nEstimated hours: 220")


def test_metadata_is_filterable_and_not_in_the_embedded_text() -> None:
    chunk = JSONStructuralChunker().chunk(_budgets())[0]

    assert chunk.metadata == {
        "budget_id": "BUD-2024-014",
        "component_id": "AUTH-001",
        "client_sector": "fintech",
        "main_technology": "Flutter",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 220.0,
    }
    assert "budget_id" not in chunk.text


def test_token_count_matches_tiktoken() -> None:
    chunk = JSONStructuralChunker().chunk(_budgets())[0]
    assert chunk.token_count == count_tokens(chunk.text) > 0


def test_long_description_is_not_split() -> None:
    budget = _budgets()[0].model_copy(deep=True)
    budget.components = [budget.components[0].model_copy(update={"description": "word " * 3000})]
    chunks = JSONStructuralChunker().chunk([budget])

    assert len(chunks) == 1
    assert chunks[0].token_count > 3000


def test_duplicate_component_ids_are_rejected() -> None:
    raw = json.loads(SAMPLE.read_text())["budgets"][0]
    raw["components"][1]["component_id"] = raw["components"][0]["component_id"]
    with pytest.raises(ValueError, match="duplicated component_id"):
        Budget.model_validate(raw)


# ---- embedder ----


def test_embed_one_returns_the_vector() -> None:
    assert _embedder(FakeEmbeddings()).embed_one("hello") == [5.0, 0.0]


def test_embed_many_batches_100_chunks_per_call_and_keeps_order() -> None:
    fake = FakeEmbeddings()
    chunks = [_chunk(i) for i in range(250)]

    embedded = _embedder(fake).embed_many(chunks)

    assert [len(call) for call in fake.calls] == [BATCH_SIZE, BATCH_SIZE, 50]
    assert [e.chunk_id for e in embedded] == [c.chunk_id for c in chunks]
    # position inside its batch comes back in the vector: order was restored per batch
    assert embedded[0].embedding[1] == 0.0 and embedded[99].embedding[1] == 99.0
    assert embedded[100].embedding[1] == 0.0
    assert embedded[0].text == chunks[0].text and embedded[0].token_count == chunks[0].token_count


def test_embed_many_with_no_chunks_makes_no_call() -> None:
    fake = FakeEmbeddings()
    assert _embedder(fake).embed_many([]) == []
    assert fake.calls == []


def test_rate_limit_retries_with_1_2_4_seconds_then_succeeds() -> None:
    fake = FakeEmbeddings(failures=[_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
    sleeps: list[float] = []

    vector = _embedder(fake, sleeps).embed_one("hello")

    assert vector == [5.0, 0.0]
    assert sleeps == [1, 2, 4]
    assert len(fake.calls) == 4


def test_rate_limit_gives_up_after_three_retries() -> None:
    fake = FakeEmbeddings(failures=[_rate_limit_error() for _ in range(4)])
    sleeps: list[float] = []

    with pytest.raises(openai.RateLimitError):
        _embedder(fake, sleeps).embed_one("hello")

    assert sleeps == [1, 2, 4]


def test_other_errors_propagate_without_retry() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    bad_request = openai.BadRequestError("nope", response=httpx.Response(400, request=request), body=None)
    fake = FakeEmbeddings(failures=[bad_request])
    sleeps: list[float] = []

    with pytest.raises(openai.BadRequestError):
        _embedder(fake, sleeps).embed_one("hello")

    assert sleeps == [] and len(fake.calls) == 1


def test_estimated_cost_uses_two_cents_per_million_tokens() -> None:
    assert estimate_cost_usd(1_000_000) == pytest.approx(0.02)
    assert estimate_cost_usd(0) == 0


# ---- endpoint ----


class FakeSession:
    """AsyncSession double: records adds/commits/rollbacks, fakes the duplicate lookup."""

    def __init__(self, existing_id: int | None = None) -> None:
        self.existing_id = existing_id
        self.added: list = []
        self.commits = 0
        self.rollbacks = 0

    async def scalar(self, stmt):
        return self.existing_id

    def add(self, obj) -> None:
        self.added.append(obj)

    def add_all(self, objs) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 42

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _payload() -> dict:
    return {
        "source_path": "data/budgets/budget_2024_014.json",
        "document_type": "historical_budget",
        "content": json.loads(SAMPLE.read_text())["budgets"][0],
    }


@pytest.fixture
def ingest_client():
    fake = FakeEmbeddings()
    session = FakeSession()
    app.dependency_overrides[get_embedder] = lambda: _embedder(fake)

    async def _session():
        yield session

    app.dependency_overrides[get_session] = _session
    with TestClient(app) as client:
        yield client, fake, session
    app.dependency_overrides.pop(get_embedder, None)
    app.dependency_overrides.pop(get_session, None)


def test_ingest_persists_document_and_chunks_in_one_commit(ingest_client) -> None:
    client, fake, session = ingest_client
    payload = _payload()
    n_components = len(payload["content"]["components"])

    response = client.post("/embeddings/ingest", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == 42
    assert body["chunks_created"] == n_components
    assert body["embedding_dimension"] == 1536
    assert body["ingestion_time_ms"] >= 0
    assert len(fake.calls) == 1  # one batched embeddings call, not chunk by chunk
    assert len(session.added) == 1 + n_components
    assert session.commits == 1 and session.rollbacks == 0
    chunk = session.added[1]
    assert chunk.chunk_type == "budget_component"
    assert chunk.metadata_["client_sector"] == "fintech"


def test_ingest_duplicate_source_path_is_409() -> None:
    session = FakeSession(existing_id=7)

    async def _session():
        yield session

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_embedder] = lambda: _embedder(FakeEmbeddings())
    try:
        response = TestClient(app).post("/embeddings/ingest", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": 7}
    assert session.added == [] and session.commits == 0


def test_ingest_validation_errors_are_422(ingest_client) -> None:
    client, _, _ = ingest_client

    assert client.post("/embeddings/ingest", json={"source_path": "x"}).status_code == 422
    broken = _payload()
    broken["content"]["components"][0]["complexity"] = "impossible"
    assert client.post("/embeddings/ingest", json=broken).status_code == 422


def test_ingest_embedder_failure_rolls_back_without_commit(ingest_client) -> None:
    client, fake, session = ingest_client
    fake.failures = [openai.APIConnectionError(request=httpx.Request("POST", "https://x"))]

    response = client.post("/embeddings/ingest", json=_payload())

    assert response.status_code == 500
    assert response.json() == {"detail": "Embedding generation failed"}
    assert session.commits == 0 and session.rollbacks == 1
