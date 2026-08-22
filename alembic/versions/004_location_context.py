"""location and questionnaire on edit sessions

Revision ID: 0004_location_context
Revises: 0003_instructions
Create Date: 2026-08-21

A quotation needs to know where the room is and what the owner wants done.
Both are captured once per session and travel with it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_location_context"
down_revision = "0003_instructions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("edit_sessions",
                  sa.Column("location", sa.JSON, nullable=False,
                            server_default="{}"))
    op.add_column("edit_sessions",
                  sa.Column("questionnaire", sa.JSON, nullable=False,
                            server_default="{}"))


def downgrade() -> None:
    op.drop_column("edit_sessions", "questionnaire")
    op.drop_column("edit_sessions", "location")