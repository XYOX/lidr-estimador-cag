"""full-text search: generated tsvector column on chunks + GIN index

The budgets are written in English, hence the 'english' text search configuration. If the
corpus language changes, the generated expression and ``Settings.FTS_CONFIG`` change together.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-11
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR,
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_content_tsv_gin", "chunks", ["content_tsv"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv_gin", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
