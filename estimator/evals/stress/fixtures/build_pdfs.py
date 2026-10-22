"""Deterministic synthetic PDFs for the attachment stress test.

Sizes are calibrated in characters of *extracted text* (what ends up in the prompt), not in PDF
bytes: 5 / 20 / 50 / 100 thousand characters, roughly 2 / 8 / 20 / 40 pages of plain text. The
100k file overshoots ``MAX_ATTACHMENT_CHARS`` (60 000) on purpose, to measure how truncation cuts.

Each document is a fake requirements specification. Four "integration requirements" are planted at
fixed relative positions (5 %, 35 %, 65 %, 95 %), each naming a distinctive technology. The runner
asks the model to name every external system in the spec, so which markers show up in the answer
says how much of the attachment the model actually used, and which ones truncation removed.

The PDFs are not committed: ``ensure_fixtures()`` regenerates them, byte-identical, on demand.

    uv run python -m evals.stress.fixtures.build_pdfs
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF

FIXTURES_DIR = Path(__file__).parent
SIZES_KB = (5, 20, 50, 100)
CHARS_PER_KB = 1000
SEED = 20260101

# (relative position in the document, technology the requirement names)
MARKERS = ((0.05, "Kerberos"), (0.35, "Elasticsearch"), (0.65, "Kafka"), (0.95, "Snowflake"))

_SUBJECTS = (
    "The platform", "The reporting module", "The scheduling service", "The administration console",
    "The notification engine", "The document workflow", "The billing component", "The audit trail",
)
_VERBS = (
    "shall validate", "must record", "shall expose", "must reconcile", "shall archive",
    "must display", "shall synchronise", "must calculate", "shall restrict", "must export",
)
_OBJECTS = (
    "every customer record before it is saved", "the history of changes per user",
    "the list of pending approvals", "daily totals grouped by department",
    "attachments up to the agreed size limit", "the status of each open request",
    "the configuration of each regional office", "overdue items older than thirty days",
    "the permissions granted to each role", "a printable summary of the current period",
)
_CLAUSES = (
    "within the response time agreed with the business owner.",
    "without requiring manual intervention from the operations team.",
    "and keep the result available for later consultation.",
    "following the naming conventions of the organisation.",
    "and report any inconsistency to the responsible administrator.",
    "so that the finance department can verify it at month end.",
)


@dataclass(frozen=True)
class Marker:
    technology: str
    char_offset: int  # position of the technology name in the extracted text


@dataclass(frozen=True)
class PdfFixture:
    size_kb: int
    path: Path
    extracted_chars: int
    markers: tuple[Marker, ...]


def _requirement(rng: random.Random, number: int) -> str:
    return (
        f"R-{number:04d}. {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
        f"{rng.choice(_CLAUSES)}"
    )


def _marker_sentence(technology: str, number: int) -> str:
    return (
        f"R-{number:04d}. Integration requirement: the platform must integrate with {technology} "
        f"as an external system of record."
    )


def build_text(size_kb: int) -> str:
    """Spec text of about ``size_kb`` thousand characters with the four markers planted."""
    target = size_kb * CHARS_PER_KB
    rng = random.Random(SEED)
    marker_at = {int(target * position): tech for position, tech in MARKERS}
    pending = sorted(marker_at)

    paragraphs: list[str] = ["Software Requirements Specification - Synthetic Stress Document", ""]
    length = sum(len(p) + 1 for p in paragraphs)
    number = 1
    while length < target:
        if pending and length >= pending[0]:
            paragraph = _marker_sentence(marker_at[pending.pop(0)], number)
        else:
            paragraph = _requirement(rng, number)
        paragraphs.append(paragraph)
        length += len(paragraph) + 1
        number += 1
    return "\n".join(paragraphs)


def _write_pdf(text: str, path: Path) -> None:
    pdf = FPDF()
    pdf.set_creation_date(datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc))
    pdf.set_compression(False)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for paragraph in text.split("\n"):
        if paragraph:
            pdf.multi_cell(0, 5, paragraph, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(5)
    pdf.output(str(path))


def build_fixture(size_kb: int, directory: Path = FIXTURES_DIR) -> PdfFixture:
    # Imported here so the fixtures module does not pull the app in at import time.
    from app.attachments.extractor import extract_text

    path = directory / f"attach_{size_kb}kb.pdf"
    _write_pdf(build_text(size_kb), path)

    extracted = extract_text(filename=path.name, content=path.read_bytes(), max_chars=10**9)
    markers = tuple(
        Marker(tech, extracted.find(tech)) for _, tech in MARKERS if extracted.find(tech) >= 0
    )
    return PdfFixture(size_kb, path, len(extracted), markers)


def ensure_fixtures(
    sizes_kb: tuple[int, ...] = SIZES_KB, directory: Path = FIXTURES_DIR
) -> dict[int, PdfFixture]:
    """(Re)generate the PDFs for ``sizes_kb`` and return them keyed by size. 0 means no file."""
    return {size: build_fixture(size, directory) for size in sizes_kb if size > 0}


if __name__ == "__main__":
    for fixture in ensure_fixtures().values():
        print(
            f"{fixture.path.name}: {fixture.extracted_chars} chars, "
            f"{[(m.technology, m.char_offset) for m in fixture.markers]}"
        )
