"""Invariants of the synthetic scenarios and of the PDF fixtures the stress runner relies on."""

from __future__ import annotations

import pytest

from evals.stress.fixtures.build_pdfs import MARKERS, build_text, ensure_fixtures
from evals.stress.scenarios import MAX_TURNS, SCENARIO_NAMES, build_scenario


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenario_shape(name: str) -> None:
    scenario = build_scenario(name)
    assert [t.turn_index for t in scenario.turns] == list(range(1, MAX_TURNS + 1))
    assert all(len(t.transcript) >= 20 for t in scenario.turns)  # endpoint min_length
    assert scenario.turns[0].fact is not None  # turn 1 declares the project name


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_fact_needle_appears_in_the_turn_that_declares_it(name: str) -> None:
    for turn in build_scenario(name).turns:
        if turn.fact:
            assert turn.fact.needle.lower() in turn.transcript.lower(), (name, turn.turn_index)


def test_facts_declared_before_is_strict() -> None:
    scenario = build_scenario("growing")
    assert scenario.facts_declared_before(1) == []
    assert [k for k, _ in scenario.facts_declared_before(3)] == [1, 2]


def test_pivot_and_contradiction_declare_their_stale_terms() -> None:
    pivot = build_scenario("pivot")
    assert pivot.turns[0].transcript.count("React") == 1
    assert pivot.stale_terms[0].from_turn == 5
    assert "Flutter" in pivot.turns[4].transcript

    contradiction = build_scenario("contradiction")
    assert "30000" in contradiction.turns[2].transcript
    assert "80000" in contradiction.turns[7].transcript
    assert contradiction.stale_terms[0].from_turn == 8


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        build_scenario("nope")


def test_pdf_fixtures_are_deterministic_and_calibrated(tmp_path) -> None:
    assert build_text(5) == build_text(5)

    first = ensure_fixtures((5,), directory=tmp_path)[5]
    first_bytes = first.path.read_bytes()
    second = ensure_fixtures((5,), directory=tmp_path)[5]
    assert second.path.read_bytes() == first_bytes
    assert 5_000 <= first.extracted_chars <= 5_400
    assert [m.technology for m in first.markers] == [tech for _, tech in MARKERS]
    # Markers are planted in increasing order of position.
    offsets = [m.char_offset for m in first.markers]
    assert offsets == sorted(offsets)
