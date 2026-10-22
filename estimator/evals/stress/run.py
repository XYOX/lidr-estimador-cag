"""Stress runner: drive the CAG through multi-turn and large-attachment scenarios, dump a CSV.

One CSV row per turn with the ``turn_observed`` fields plus one column per metric. In HTTP mode
(``--http``) it talks to a running estimator; otherwise it uses the in-process ``TestClient``,
same as ``evals/run.py``. Either way the per-turn observation is read back from
``GET /sessions/{id}`` (``last_turn`` plus the summary / anchors / metadata snapshot), not from the
logs: it works in both transports and does not depend on where stdout goes.

Scenarios: ``growing``, ``pivot``, ``contradiction`` (20 turns each, sweep N in 1/3/6/10/20 is
read off prefixes of that one run) and ``attachments`` (one initial estimation per size in
``--attachment-sizes``; 0 = no attachment baseline).

    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction,attachments \\
        --attachment-sizes 0,5,20,50,100 --repeats 3 --output evals/stress/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from evals.stress.fixtures.build_pdfs import PdfFixture, ensure_fixtures
from evals.stress.metrics import (
    DURABLE_LOCATIONS,
    MEMORY_LOCATIONS,
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    snapshot_locations,
)
from evals.stress.scenarios import SCENARIO_NAMES, Scenario, build_scenario

ATTACHMENTS = "attachments"
ATTACHMENT_TRANSCRIPT = (
    "Estimate the project described in the attached requirements specification. In the summary, "
    "name every external system or technology the specification says the platform must integrate with."
)
ATTACHMENT_NEEDLE_PLACES = ("summary", "phases")

TURN_OBSERVED_FIELDS = (
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
    "llm_calls",
)
RUN_FIELDS = ("scenario", "attachment_kb", "repeat", "turns_planned", "failed_attempts", "error")
METRIC_FIELDS = (
    "latency_budget",
    "cost_budget",
    "facts_declared",
    "recall_durable",
    "recall_visible",
    "first_fact_durable",
    "first_fact_visible",
    "stale_in_memory",
    "fact_locations",
    "project_name",
    "metadata_technologies",
    "attachment_markers_total",
    "attachment_markers_in_context",
    "attachment_markers_recalled",
)
CSV_COLUMNS = RUN_FIELDS + TURN_OBSERVED_FIELDS + METRIC_FIELDS


class Transport:
    """``httpx`` client against a live server, or the in-process ``TestClient``."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create_session(self) -> str:
        response = self.client.post("/sessions")
        response.raise_for_status()
        return response.json()["session_id"]

    def estimate(self, session_id: str, data: dict[str, str], pdf: Path | None) -> Any:
        if pdf is None:
            return self.client.post(f"/sessions/{session_id}/estimate", data=data)
        with pdf.open("rb") as handle:
            files = [("attachments", (pdf.name, handle.read(), "application/pdf"))]
        return self.client.post(f"/sessions/{session_id}/estimate", data=data, files=files)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        response = self.client.get(f"/sessions/{session_id}")
        response.raise_for_status()
        return response.json()


@contextmanager
def open_transport(http_base: str | None) -> Iterator[Transport]:
    if http_base:
        with httpx.Client(base_url=http_base, timeout=300.0) as client:
            yield Transport(client)
        return

    # In-process: same isolation trick as evals/run.py, a singleton store shared by all requests.
    from fastapi.testclient import TestClient

    from app.dependencies import get_session_store
    from app.main import app
    from app.sessions.store import SessionStore

    store = SessionStore(max_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        with TestClient(app) as client:
            yield Transport(client)
    finally:
        app.dependency_overrides.pop(get_session_store, None)


def estimate_with_retries(
    transport: Transport, session_id: str, data: dict[str, str], pdf: Path | None, retries: int
) -> tuple[Any, int]:
    """POST the turn, retrying 5xx. Returns ``(last_response, failed_attempts)``.

    A failed turn leaves the session untouched (the error is raised before the history is
    appended), so resending the same turn is safe. Failures are counted, not hidden: they are the
    "hard failure" rate of the pipeline (Instructor exhausting its validator retries).
    """
    failed = 0
    response = transport.estimate(session_id, data, pdf)
    while response.status_code >= 500 and failed < retries:
        failed += 1
        response = transport.estimate(session_id, data, pdf)
    if response.status_code >= 500:
        failed += 1
    return response, failed


def _form(transcript: str) -> dict[str, str]:
    return {
        "transcript": transcript,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }


def _result_text(payload: dict[str, Any]) -> str:
    result = payload["result"]
    return " ".join([result["summary"], *(f"{p['name']} {p['summary']}" for p in result["phases"])])


def _memory_snapshot(info: dict[str, Any], scenario: Scenario, turn_index: int) -> dict[str, Any]:
    """Memory locations the LLM sees next turn. ``window`` is rebuilt from what the runner sent.

    The window holds the last ``messages_in_window / 2`` turns, so the transcripts of those turns
    (they contain every fact verbatim) stand for it without exposing raw messages over the API.
    """
    window_turns = (info["last_turn"]["messages_in_window"] // 2) if info.get("last_turn") else 0
    window = [t.transcript for t in scenario.turns if turn_index - window_turns < t.turn_index <= turn_index]
    return {
        "summary": info.get("summary") or "",
        "anchors": info.get("anchors") or [],
        "metadata": info.get("metadata") or {},
        "window": window,
    }


def _memory_columns(
    scenario: Scenario, turn_index: int, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Fact-tracker columns: facts declared at turns k < N, looked up in the snapshot after turn N."""
    declared = scenario.facts_declared_before(turn_index)
    locations: dict[str, list[str]] = {}
    durable_scores: list[float] = []
    visible_scores: list[float] = []
    for _, fact in declared:
        metric_args = dict(fact=fact.needle, aliases=fact.aliases)
        durable_scores.append(MemoryDriftMetric(where=DURABLE_LOCATIONS, **metric_args).evaluate(snapshot).score)
        visible_scores.append(MemoryDriftMetric(where=MEMORY_LOCATIONS, **metric_args).evaluate(snapshot).score)
        locations[fact.label] = snapshot_locations(snapshot, fact.needle, fact.aliases)

    stale = [
        term.label
        for term in scenario.stale_terms
        if turn_index >= term.from_turn
        and any(
            loc in DURABLE_LOCATIONS
            for loc in snapshot_locations(snapshot, term.needle, term.aliases)
        )
    ]

    def mean(scores: list[float]) -> float | str:
        return round(sum(scores) / len(scores), 4) if scores else ""

    return {
        "facts_declared": len(declared),
        "recall_durable": mean(durable_scores),
        "recall_visible": mean(visible_scores),
        "first_fact_durable": durable_scores[0] if durable_scores else "",
        "first_fact_visible": visible_scores[0] if visible_scores else "",
        "stale_in_memory": ";".join(stale),
        "fact_locations": json.dumps(locations, ensure_ascii=False),
    }


def _attachment_columns(
    fixture: PdfFixture | None, max_chars: int, response_text: str
) -> dict[str, Any]:
    if fixture is None:
        return {}
    in_context = [m for m in fixture.markers if m.char_offset + len(m.technology) <= max_chars]
    recalled = [m for m in in_context if m.technology.lower() in response_text.lower()]
    return {
        "attachment_markers_total": len(fixture.markers),
        "attachment_markers_in_context": len(in_context),
        "attachment_markers_recalled": len(recalled),
    }


def _row(
    *,
    scenario: str,
    attachment_kb: int,
    repeat: int,
    turns_planned: int,
    info: dict[str, Any],
    budgets: tuple[LatencyBudgetMetric, CostBudgetMetric],
    extra: dict[str, Any],
    failed_attempts: int = 0,
) -> dict[str, Any]:
    observation = info["last_turn"]
    latency_metric, cost_metric = budgets
    row: dict[str, Any] = {
        "scenario": scenario,
        "attachment_kb": attachment_kb,
        "repeat": repeat,
        "turns_planned": turns_planned,
        "failed_attempts": failed_attempts,
        "error": "",
        **{field: observation[field] for field in TURN_OBSERVED_FIELDS},
        "latency_budget": latency_metric.evaluate(observation).score,
        "cost_budget": cost_metric.evaluate(observation).score,
        "project_name": (info.get("metadata") or {}).get("project_name") or "",
        "metadata_technologies": json.dumps(
            (info.get("metadata") or {}).get("mentioned_technologies", []), ensure_ascii=False
        ),
    }
    row.update(extra)
    return row


def run_conversation(
    transport: Transport,
    scenario: Scenario,
    *,
    repeat: int,
    turns: int,
    budgets: tuple[LatencyBudgetMetric, CostBudgetMetric],
    retries: int = 0,
) -> Iterator[dict[str, Any]]:
    session_id = transport.create_session()
    for turn in scenario.turns[:turns]:
        response, failed = estimate_with_retries(
            transport, session_id, _form(turn.transcript), None, retries
        )
        if response.status_code != 200:
            yield _error_row(
                scenario.name, 0, repeat, turns, turn.turn_index, session_id, response, failed
            )
            return
        info = transport.snapshot(session_id)
        snapshot = _memory_snapshot(info, scenario, turn.turn_index)
        yield _row(
            scenario=scenario.name,
            attachment_kb=0,
            repeat=repeat,
            turns_planned=turns,
            info=info,
            budgets=budgets,
            extra=_memory_columns(scenario, turn.turn_index, snapshot),
            failed_attempts=failed,
        )


def run_attachment(
    transport: Transport,
    size_kb: int,
    fixture: PdfFixture | None,
    *,
    repeat: int,
    max_chars: int,
    budgets: tuple[LatencyBudgetMetric, CostBudgetMetric],
    retries: int = 0,
) -> Iterator[dict[str, Any]]:
    session_id = transport.create_session()
    response, failed = estimate_with_retries(
        transport, session_id, _form(ATTACHMENT_TRANSCRIPT), fixture.path if fixture else None, retries
    )
    if response.status_code != 200:
        yield _error_row(ATTACHMENTS, size_kb, repeat, 1, 1, session_id, response, failed)
        return
    info = transport.snapshot(session_id)
    yield _row(
        scenario=ATTACHMENTS,
        attachment_kb=size_kb,
        repeat=repeat,
        turns_planned=1,
        info=info,
        budgets=budgets,
        extra=_attachment_columns(fixture, max_chars, _result_text(response.json())),
        failed_attempts=failed,
    )


def _error_row(
    scenario: str,
    attachment_kb: int,
    repeat: int,
    turns_planned: int,
    turn_index: int,
    session_id: str,
    response: Any,
    failed_attempts: int = 0,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "attachment_kb": attachment_kb,
        "repeat": repeat,
        "turns_planned": turns_planned,
        "failed_attempts": failed_attempts,
        "turn_index": turn_index,
        "session_id": session_id,
        "error": f"HTTP {response.status_code}: {response.text[:200]}",
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--http", default=None, help="Base URL of a running estimator (default: in-process)")
    parser.add_argument(
        "--scenarios",
        default=",".join((*SCENARIO_NAMES, ATTACHMENTS)),
        help=f"comma list of: {', '.join((*SCENARIO_NAMES, ATTACHMENTS))}",
    )
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100", help="thousands of chars; 0 = none")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turns", type=int, default=20, help="turns per multi-turn conversation")
    parser.add_argument(
        "--turn-retries", type=int, default=3, help="resends of a turn that fails with 5xx before aborting"
    )
    parser.add_argument("--latency-budget-ms", type=int, default=10_000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.005)
    parser.add_argument(
        "--max-attachment-chars",
        type=int,
        default=Settings.model_fields["MAX_ATTACHMENT_CHARS"].default,
        help="the server's MAX_ATTACHMENT_CHARS, used to tell truncated markers from forgotten ones",
    )
    parser.add_argument("--output", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args(argv)

    scenarios = _parse_csv_list(args.scenarios)
    unknown = set(scenarios) - {*SCENARIO_NAMES, ATTACHMENTS}
    if unknown:
        parser.error(f"unknown scenarios: {sorted(unknown)}")
    sizes = [int(size) for size in _parse_csv_list(args.attachment_sizes)]

    max_chars = args.max_attachment_chars
    fixtures = ensure_fixtures(tuple(sizes)) if ATTACHMENTS in scenarios else {}
    budgets = (LatencyBudgetMetric(args.latency_budget_ms), CostBudgetMetric(args.cost_budget_usd))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    errors = 0
    with open_transport(args.http) as transport, args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, restval="")
        writer.writeheader()

        def emit(rows: Iterator[dict[str, Any]]) -> None:
            nonlocal rows_written, errors
            for row in rows:
                writer.writerow(row)
                out.flush()  # a crash or Ctrl-C keeps everything gathered so far
                rows_written += 1
                if row["error"]:
                    errors += 1
                    print(f"  ERROR {row['scenario']} r{row['repeat']} turn {row['turn_index']}: {row['error']}")
                else:
                    print(
                        f"  {row['scenario']:<13} kb={row['attachment_kb']:<3} r{row['repeat']} "
                        f"turn {row['turn_index']:>2}  {row['latency_ms']:>6} ms  "
                        f"in={row['tokens_in']:>6} out={row['tokens_out']:>5}  ${row['cost_usd']:.5f}"
                        + (f"  (retried x{row['failed_attempts']})" if row["failed_attempts"] else ""),
                        flush=True,
                    )

        for name in scenarios:
            print(f"== {name}", flush=True)
            for repeat in range(1, args.repeats + 1):
                if name == ATTACHMENTS:
                    for size in sizes:
                        emit(
                            run_attachment(
                                transport, size, fixtures.get(size), repeat=repeat,
                                max_chars=max_chars, budgets=budgets, retries=args.turn_retries,
                            )
                        )
                else:
                    emit(
                        run_conversation(
                            transport, build_scenario(name), repeat=repeat,
                            turns=args.turns, budgets=budgets, retries=args.turn_retries,
                        )
                    )

    print(f"\n{rows_written} rows ({errors} errors) -> {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
