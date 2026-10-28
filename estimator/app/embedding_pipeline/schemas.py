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
    source_path: str = Field(min_length=1)
    document_type: str = Field(min_length=1, max_length=50)
    # One complete budget, exactly as the chunker consumes it.
    content: Budget


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    k: int
    search_time_ms: int
    results: list[SearchResult]
