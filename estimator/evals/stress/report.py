"""Turn ``results.csv`` into the Markdown tables of REPORT.md (summary + the three curves).

The numbers in REPORT.md come from this script, never typed by hand; only the reading paragraphs
are written by a person.

    uv run python -m evals.stress.report --csv evals/stress/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evals.stress.scenarios import SCENARIO_LENGTHS

TOKEN_BINS = (0, 2_000, 4_000, 6_000, 8_000, 12_000, 16_000, math.inf)
CUMULATIVE_TURNS = (1, 3, 6, 10, 15, 20)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation): the value a real request actually had."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def load_all_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(path: Path) -> list[dict]:
    """Rows of turns that completed; aborted turns (error rows) are counted separately."""
    return [row for row in load_all_rows(path) if not row["error"]]


def _num(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value != "" else None


def _values(rows: list[dict], key: str) -> list[float]:
    return [v for v in (_num(row, key) for row in rows) if v is not None]


def _fmt(value: float | None, spec: str = ".0f") -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def _table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def summary_table(rows: list[dict]) -> str:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], int(row["attachment_kb"]))].append(row)

    body = []
    for (scenario, kb), group in sorted(groups.items()):
        latencies = _values(group, "latency_ms")
        sessions = {(r["repeat"]) for r in group}
        total_cost = sum(_values(group, "cost_usd")) / max(len(sessions), 1)
        exact = mean(r["cache_hit_kind"] == "exact" for r in group) * 100
        semantic = mean(r["cache_hit_kind"] == "semantic" for r in group) * 100
        recall = _values(group, "recall_durable")
        visible = _values(group, "recall_visible")
        markers_possible = sum(_values(group, "attachment_markers_in_context"))
        markers_recalled = sum(_values(group, "attachment_markers_recalled"))
        body.append(
            [
                scenario,
                str(kb),
                str(len(group)),
                _fmt(percentile(latencies, 50)),
                _fmt(percentile(latencies, 95)),
                _fmt(total_cost, ".4f"),
                f"{exact:.0f}%",
                f"{semantic:.0f}%",
                _fmt(mean(recall) if recall else None, ".2f"),
                _fmt(mean(visible) if visible else None, ".2f"),
                _fmt(markers_recalled / markers_possible if markers_possible else None, ".2f"),
                str(int(sum(_values(group, "failed_attempts")))),
            ]
        )
    return _table(
        [
            "scenario", "attach KB", "turns", "P50 ms", "P95 ms", "cost USD / session",
            "exact hit", "semantic hit", "recall durable", "recall visible", "attach recall",
            "failed attempts",
        ],
        body,
    )


def latency_vs_tokens(rows: list[dict]) -> str:
    body = []
    for low, high in zip(TOKEN_BINS, TOKEN_BINS[1:]):
        group = [r for r in rows if low <= float(r["tokens_in"]) < high]
        if not group:
            continue
        latencies = _values(group, "latency_ms")
        label = f"{low:,}-{high:,.0f}" if high != math.inf else f">= {low:,}"
        body.append(
            [
                label,
                str(len(group)),
                _fmt(percentile(latencies, 50)),
                _fmt(percentile(latencies, 95)),
                _fmt(mean(_values(group, "cost_usd")), ".5f"),
                _fmt(mean(_values(group, "llm_calls")), ".1f"),
            ]
        )
    return _table(["tokens_in (turn total)", "n", "P50 ms", "P95 ms", "mean cost USD", "llm calls"], body)


def cumulative_cost_vs_turn(rows: list[dict]) -> str:
    by_scenario: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["scenario"] == "attachments":
            continue
        by_scenario[row["scenario"]][row["repeat"]][int(row["turn_index"])] = float(row["cost_usd"])

    body = []
    for scenario, repeats in sorted(by_scenario.items()):
        cells = [scenario]
        for turn in CUMULATIVE_TURNS:
            totals = [
                sum(cost for t, cost in turns.items() if t <= turn)
                for turns in repeats.values()
                if turn in turns
            ]
            cells.append(_fmt(mean(totals) if totals else None, ".4f"))
        first = [turns[1] for turns in repeats.values() if 1 in turns]
        last = [turns[max(turns)] for turns in repeats.values() if turns]
        ratio = mean(last) / mean(first) if first and last and mean(first) else None
        cells.append(_fmt(ratio, ".2f") + "x")
        body.append(cells)
    return _table(
        ["scenario", *[f"turn {t}" for t in CUMULATIVE_TURNS], "last / first turn cost"], body
    )


def recall_vs_n(rows: list[dict]) -> str:
    body = []
    for scenario in sorted({r["scenario"] for r in rows} - {"attachments"}):
        for n in SCENARIO_LENGTHS:
            group = [r for r in rows if r["scenario"] == scenario and int(r["turn_index"]) == n]
            durable = _values(group, "recall_durable")
            visible = _values(group, "recall_visible")
            first_durable = _values(group, "first_fact_durable")
            first_visible = _values(group, "first_fact_visible")
            stale = [r for r in group if r["stale_in_memory"]]
            body.append(
                [
                    scenario,
                    str(n),
                    _fmt(mean(durable) if durable else None, ".2f"),
                    _fmt(mean(visible) if visible else None, ".2f"),
                    _fmt(mean(first_durable) if first_durable else None, ".2f"),
                    _fmt(mean(first_visible) if first_visible else None, ".2f"),
                    f"{len(stale)}/{len(group)}" if group else "-",
                ]
            )
    return _table(
        [
            "scenario", "N (turns)", "recall durable", "recall visible",
            "project name durable", "project name visible", "stale term in memory",
        ],
        body,
    )


def render(rows: list[dict]) -> str:
    return "\n\n".join(
        [
            "### Summary\n\n" + summary_table(rows),
            "### Curve 1 - latency vs tokens_in\n\n" + latency_vs_tokens(rows),
            "### Curve 2 - cumulative cost (USD) vs turn\n\n" + cumulative_cost_vs_turn(rows),
            "### Curve 3 - fact recall vs N\n\n" + recall_vs_n(rows),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("evals/stress/results.csv"))
    args = parser.parse_args()
    everything = load_all_rows(args.csv)
    rows = [row for row in everything if not row["error"]]
    aborted = [row for row in everything if row["error"]]
    print(render(rows))
    print(
        f"\nCompleted turns: {len(rows)}. Aborted sessions (all retries failed): {len(aborted)}. "
        f"Failed attempts that a retry recovered: "
        f"{int(sum(_values(rows, 'failed_attempts')))}."
    )


if __name__ == "__main__":
    main()
