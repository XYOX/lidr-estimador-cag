"""Budget and memory metrics for the CAG stress test.

They live next to the runner and not in ``evals/metrics.py`` because they score a different
object: the ``turn_observed`` payload (latency, cost) and the ``GET /sessions/{id}`` snapshot
(summary, anchors, metadata), not a ``(GoldenCase, EstimationResult)`` pair. They reuse
``MetricResult`` so the result shape is the same one the golden-dataset suite reports.

All three are deterministic: threshold comparisons and case-insensitive substring matches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from evals.metrics import MetricResult

MEMORY_LOCATIONS = ("summary", "anchors", "metadata", "window")
# What survives the sliding window. ``window`` is not durable memory: the fact is still verbatim
# in the last turns, but it is evicted as soon as the conversation moves on.
DURABLE_LOCATIONS = ("summary", "anchors", "metadata")


class LatencyBudgetMetric:
    """1.0 if ``latency_ms`` <= ``budget_ms``; 0.0 otherwise. The budget is inclusive."""

    name = "latency_budget"

    def __init__(self, budget_ms: int) -> None:
        self.budget_ms = budget_ms

    def evaluate(self, observation: Mapping[str, Any]) -> MetricResult:
        latency_ms = observation["latency_ms"]
        passed = latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"latency {latency_ms} ms {'<=' if passed else '>'} budget {self.budget_ms} ms",
        )


class CostBudgetMetric:
    """1.0 if ``cost_usd`` <= ``budget_usd``; 0.0 otherwise. The budget is inclusive."""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, observation: Mapping[str, Any]) -> MetricResult:
        cost_usd = observation["cost_usd"]
        passed = cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"cost ${cost_usd:.6f} {'<=' if passed else '>'} budget ${self.budget_usd:.6f}",
        )


def snapshot_locations(snapshot: Mapping[str, Any], fact: str, aliases: Sequence[str] = ()) -> list[str]:
    """Locations of the snapshot whose text contains ``fact`` (or an alias), case-insensitive."""
    needles = [fact.lower(), *(alias.lower() for alias in aliases)]
    found: list[str] = []
    for location in MEMORY_LOCATIONS:
        text = _location_text(snapshot, location).lower()
        if any(needle in text for needle in needles):
            found.append(location)
    return found


def _location_text(snapshot: Mapping[str, Any], location: str) -> str:
    value = snapshot.get(location)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    return "\n".join(str(item) for item in value)


class MemoryDriftMetric:
    """1.0 if the fact declared at turn k is in the snapshot of a later turn N (N > k).

    ``snapshot`` holds one entry per memory location: ``summary`` (str), ``anchors`` (list of
    str), ``metadata`` (the ProjectMetadata dict) and, optionally, ``window`` (list of str with
    the turns still in the sliding window). ``where`` picks which locations count; the default
    is durable memory only, so a fact that is merely still inside the window does not pass.

    ``aliases`` are extra spellings of the same fact (``30000`` / ``30,000``): the match stays
    an exact case-insensitive substring, there is no fuzzy or semantic matching.
    """

    name = "memory_drift"

    def __init__(
        self,
        fact: str,
        where: Sequence[str] = DURABLE_LOCATIONS,
        aliases: Sequence[str] = (),
    ) -> None:
        unknown = set(where) - set(MEMORY_LOCATIONS)
        if unknown:
            raise ValueError(f"unknown memory locations {sorted(unknown)}; use {MEMORY_LOCATIONS}")
        self.fact = fact
        self.where = tuple(where)
        self.aliases = tuple(aliases)

    def evaluate(self, session_snapshot: Mapping[str, Any]) -> MetricResult:
        hits = [
            location
            for location in snapshot_locations(session_snapshot, self.fact, self.aliases)
            if location in self.where
        ]
        passed = bool(hits)
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=(
                f"fact {self.fact!r} found in {hits}"
                if passed
                else f"fact {self.fact!r} not found in {list(self.where)}"
            ),
        )
