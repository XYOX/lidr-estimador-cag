"""Structural chunker for historical budgets: one budget component = one chunk.

No overlap and no fixed-size splitting on purpose. An unusually long description is a data point
to discuss, not something to hide: ``token_count`` makes it visible.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

EMBEDDING_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


class JSONStructuralChunker:
    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        return [
            self._chunk_component(budget, component)
            for budget in budgets
            for component in budget.components
        ]

    @staticmethod
    def _chunk_component(budget: Budget, component: BudgetComponent) -> Chunk:
        sector = budget.client_metadata.sector
        # The parent header (project, sector, year, tech) is part of the embedded text: without it
        # an "Authentication backend" component would lose which client and sector it belongs to.
        text = (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {sector} | Year: {budget.year} | Main tech: {budget.main_technology}]\n"
            f"\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours:g}"
        )
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=count_tokens(text),
        )
