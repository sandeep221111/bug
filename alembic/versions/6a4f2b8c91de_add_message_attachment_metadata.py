"""add message attachment metadata

Revision ID: 6a4f2b8c91de
Revises: 1d5280e25742
Create Date: 2026-09-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6a4f2b8c91de"
down_revision: Union[str, Sequence[str], None] = "1d5280e25742"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("attachment_storage_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("attachment_name", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("attachment_mime_type", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "messages",
        sa.Column("attachment_size_bytes", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "attachment_size_bytes")
    op.drop_column("messages", "attachment_mime_type")
    op.drop_column("messages", "attachment_name")
    op.drop_column("messages", "attachment_storage_key")
