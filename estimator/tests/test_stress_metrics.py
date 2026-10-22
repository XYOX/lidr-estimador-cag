"""Unit tests for the stress metrics: one that passes, one that fails, one on the boundary."""

from __future__ import annotations

import pytest

from evals.metrics import MetricResult
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


class TestLatencyBudget:
    def test_passes_under_budget(self) -> None:
        result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 1200})
        assert isinstance(result, MetricResult)
        assert (result.score, result.passed) == (1.0, True)

    def test_fails_over_budget(self) -> None:
        result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4001})
        assert (result.score, result.passed) == (0.0, False)
        assert "4001" in result.details

    def test_budget_is_inclusive(self) -> None:
        assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4000}).passed

    def test_missing_field_fails_hard(self) -> None:
        with pytest.raises(KeyError):
            LatencyBudgetMetric(budget_ms=4000).evaluate({})


class TestCostBudget:
    def test_passes_under_budget(self) -> None:
        assert CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.0004}).passed

    def test_fails_over_budget(self) -> None:
        result = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.0101})
        assert (result.score, result.passed) == (0.0, False)

    def test_budget_is_inclusive_and_zero_cost_passes(self) -> None:
        assert CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.01}).passed
        assert CostBudgetMetric(budget_usd=0.0).evaluate({"cost_usd": 0.0}).passed


SNAPSHOT = {
    "summary": "The client is building Nimbus, a CRM with Okta login.",
    "anchors": ["Budget is locked at 30000 EUR."],
    "metadata": {"project_name": "Nimbus", "mentioned_technologies": ["React", "Postgres"]},
    "window": ["Add an audit log of every change."],
}


class TestMemoryDrift:
    def test_found_in_summary(self) -> None:
        result = MemoryDriftMetric("Okta").evaluate(SNAPSHOT)
        assert (result.score, result.passed) == (1.0, True)
        assert "summary" in result.details

    def test_match_is_case_insensitive(self) -> None:
        assert MemoryDriftMetric("NIMBUS").evaluate(SNAPSHOT).passed
        assert MemoryDriftMetric("postgres").evaluate(SNAPSHOT).passed  # inside metadata

    def test_found_in_anchors(self) -> None:
        assert MemoryDriftMetric("30000", where=["anchors"]).evaluate(SNAPSHOT).passed

    def test_missing_fact_fails(self) -> None:
        result = MemoryDriftMetric("Flutter").evaluate(SNAPSHOT)
        assert (result.score, result.passed) == (0.0, False)

    def test_where_restricts_the_locations(self) -> None:
        # Present in the summary, but the caller only accepts anchors.
        assert not MemoryDriftMetric("Okta", where=["anchors"]).evaluate(SNAPSHOT).passed

    def test_window_is_not_durable_memory_by_default(self) -> None:
        assert not MemoryDriftMetric("audit log").evaluate(SNAPSHOT).passed
        assert MemoryDriftMetric("audit log", where=["window"]).evaluate(SNAPSHOT).passed

    def test_alias_matches_another_spelling(self) -> None:
        snapshot = {**SNAPSHOT, "anchors": [], "summary": "Budget capped at 30,000 EUR."}
        assert not MemoryDriftMetric("30000").evaluate(snapshot).passed
        assert MemoryDriftMetric("30000", aliases=["30,000"]).evaluate(snapshot).passed

    def test_empty_snapshot_and_none_summary_do_not_crash(self) -> None:
        assert not MemoryDriftMetric("Nimbus").evaluate({}).passed
        assert not MemoryDriftMetric("Nimbus").evaluate({"summary": None, "anchors": []}).passed

    def test_unknown_location_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown memory locations"):
            MemoryDriftMetric("Nimbus", where=["summary", "database"])
