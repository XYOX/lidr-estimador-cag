"""Pydantic models for the embedding pipeline (input budgets, chunks, ingest contract)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Complexity = Literal["low", "medium", "high"]
MetadataValue = str | int | float


class ClientMetadata(BaseModel):
    # The universe of sectors is not closed in the historical data, so it stays a free string.
    sector: str = Field(min_length=1)
    country: str | None = None
    company_size: str | None = None

    @field_validator("sector")
    @classmethod
    def _normalise_sector(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("sector must not be blank")
        return value


class BudgetComponent(BaseModel):
    component_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tech_stack: list[str] = Field(default_factory=list)
    complexity: Complexity
    estimated_hours: float = Field(ge=0)


class Budget(BaseModel):
    budget_id: str = Field(min_length=1)
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1)
    main_technology: str = Field(min_length=1)
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: float = Field(ge=0)
    components: list[BudgetComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_component_ids(self) -> Budget:
        # chunk_id is "{budget_id}::{component_id}", so duplicates would collide.
        ids = [component.component_id for component in self.components]
        duplicated = sorted({cid for cid in ids if ids.count(cid) > 1})
        if duplicated:
            raise ValueError(f"duplicated component_id in {self.budget_id}: {duplicated}")
        return self


class Chunk(BaseModel):
    chunk_id: str
    text: str
    # Filterable fields: they travel with the chunk but are NOT embedded.
    metadata: dict[str, MetadataValue]
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(min_length=1)


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    # total_budgets, total_chunks, total_tokens, estimated_cost_usd
    stats: dict[str, int | float]
