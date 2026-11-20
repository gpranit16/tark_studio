"""add is_archived column to conversations

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-18 17:52:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_conversations_is_archived", "conversations", ["is_archived"])


def downgrade() -> None:
    op.drop_index("ix_conversations_is_archived", table_name="conversations")
    op.drop_column("conversations", "is_archived")
