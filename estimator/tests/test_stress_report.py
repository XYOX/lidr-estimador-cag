"""The report tables are derived from the CSV: check percentiles and the table plumbing."""

from __future__ import annotations

import csv
import math


from evals.stress.report import load_rows, percentile, render
from evals.stress.run import CSV_COLUMNS


def _row(**overrides) -> dict:
    base = dict.fromkeys(CSV_COLUMNS, "")
    base.update(
        scenario="growing", attachment_kb=0, repeat=1, turns_planned=20, error="", turn_index=1,
        session_id="s", tokens_in=1000, tokens_out=200, cost_usd=0.001, latency_ms=2000,
        cache_hit_kind="none", llm_calls=2, recall_durable=1.0, recall_visible=1.0,
        first_fact_durable=1.0, first_fact_visible=1.0, stale_in_memory="",
    )
    base.update(overrides)
    return base


def test_percentile_is_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7], 95) == 7
    assert math.isnan(percentile([], 95))


def test_error_rows_are_dropped_on_load(tmp_path) -> None:
    path = tmp_path / "r.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(_row())
        writer.writerow(_row(error="HTTP 502: boom"))
    assert len(load_rows(path)) == 1


def test_render_has_summary_and_the_three_curves() -> None:
    rows = [_row(turn_index=n, cost_usd=0.001 * n) for n in (1, 3, 6, 10, 20)]
    rows.append(_row(scenario="attachments", attachment_kb=5, turns_planned=1, latency_ms=3000,
                     attachment_markers_total=4, attachment_markers_in_context=4,
                     attachment_markers_recalled=3, recall_durable="", recall_visible=""))
    report = render(rows)

    for heading in ("### Summary", "### Curve 1", "### Curve 2", "### Curve 3"):
        assert heading in report
    assert "20.00x" in report  # last turn costs 20x the first in this synthetic run
    assert "0.75" in report  # 3 of 4 attachment markers recalled
    assert "nan" not in report.lower()
