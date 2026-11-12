"""Every conversational turn emits one aggregated ``turn_observed`` event with 13 fields."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import estimation as estimation_module
from app.services.llm_wrapper import _estimate_cost, _record_usage
from tests.conftest import FakeLLMWrapper

FORM = {
    "transcript": "We need an estimate for the Nimbus CRM, React + Postgres, team of 3.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

REQUIRED_FIELDS = {
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
}


class RecordingLogger:
    """Stands in for the module logger: ``cache_logger_on_first_use`` (set in app.main) makes
    ``structlog.testing.capture_logs`` unreliable once any other test has logged."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def info(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class UsageReportingWrapper(FakeLLMWrapper):
    """Fake that reports 1000 in / 200 out tokens per call, like the real Instructor hook."""

    def complete_structured_chat(self, **kwargs):
        _record_usage(
            SimpleNamespace(
                model="gpt-4o-mini-2024-07-18",
                usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
            )
        )
        return super().complete_structured_chat(**kwargs)


@pytest.fixture
def usage_client(conversational_client, fake_wrapper):
    client, store = conversational_client
    wrapper = UsageReportingWrapper()
    wrapper.scripted = fake_wrapper.scripted
    client.app.dependency_overrides.clear()  # rewire the service onto the usage-reporting fake
    from app.dependencies import get_estimation_service, get_session_store
    from app.services.estimation import EstimationService

    service = EstimationService(
        llm_wrapper=wrapper,
        exact_cache=None,
        semantic_cache=None,
        openai_client=None,
        metadata_extractor_model="gpt-4o-mini",
    )
    client.app.dependency_overrides[get_estimation_service] = lambda: service
    client.app.dependency_overrides[get_session_store] = lambda: store
    return client, wrapper


def test_turn_observed_has_all_fields_and_aggregates_usage(usage_client, monkeypatch) -> None:
    client, _wrapper = usage_client
    recorder = RecordingLogger()
    monkeypatch.setattr(estimation_module, "log", recorder)
    session_id = client.post("/sessions").json()["session_id"]

    response = client.post(f"/sessions/{session_id}/estimate", data=FORM)
    assert response.status_code == 200, response.text

    events = [entry for entry in recorder.events if entry["event"] == "turn_observed"]
    assert len(events) == 1
    event = events[0]
    assert REQUIRED_FIELDS <= event.keys()
    assert event["turn_index"] == 1
    assert event["session_id"] == session_id
    assert event["enriched_transcript_chars"] == len(FORM["transcript"])
    assert event["attachments_total_chars"] == 0
    assert event["cache_hit_kind"] == "none"
    # Actor + metadata extractor: two LLM calls summed into the turn.
    assert event["llm_calls"] == 2
    assert event["tokens_in"] == 2000
    assert event["tokens_out"] == 400
    assert event["cost_usd"] == pytest.approx(_estimate_cost("gpt-4o-mini", 2000, 400))
    assert event["cost_usd"] > 0


def test_turn_index_increments_and_session_exposes_last_turn(usage_client) -> None:
    client, _wrapper = usage_client
    session_id = client.post("/sessions").json()["session_id"]

    for expected in (1, 2, 3):
        client.post(f"/sessions/{session_id}/estimate", data=FORM)
        info = client.get(f"/sessions/{session_id}").json()
        assert info["turns_completed"] == expected
        assert info["last_turn"]["turn_index"] == expected


def test_usage_does_not_leak_between_turns(usage_client) -> None:
    client, _wrapper = usage_client
    session_id = client.post("/sessions").json()["session_id"]

    client.post(f"/sessions/{session_id}/estimate", data=FORM)
    first = client.get(f"/sessions/{session_id}").json()["last_turn"]
    client.post(f"/sessions/{session_id}/estimate", data=FORM)
    second = client.get(f"/sessions/{session_id}").json()["last_turn"]

    assert second["tokens_in"] == first["tokens_in"] == 2000


def test_cost_resolves_dated_model_snapshots() -> None:
    assert _estimate_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert _estimate_cost("unknown-model", 1_000_000, 1_000_000) == 0.0
