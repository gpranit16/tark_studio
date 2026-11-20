"""enable vector extension

Revision ID: 0001
Revises: 
Create Date: 2026-11-18 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension for dense vector storage and HNSW indexing
    # (Note: PostgreSQL 13+ natively provides gen_random_uuid(), so uuid-ossp is not needed)
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')


def downgrade() -> None:
    op.execute('DROP EXTENSION IF EXISTS "vector";')

