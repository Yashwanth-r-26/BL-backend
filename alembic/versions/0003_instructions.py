"""free-text instruction steps

Revision ID: 0003_instructions
Revises: 0002_catalogue_editing
Create Date: 2026-08-21

An edit can now come from the user's own words rather than a catalogue pick,
so a step may have no SKU and carries the instruction text instead.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_instructions"
down_revision = "0002_catalogue_editing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("edit_steps", sa.Column("instruction", sa.Text))
    with op.batch_alter_table("edit_steps") as batch:
        batch.alter_column("replacement_sku", existing_type=sa.String(128),
                           nullable=True)


def downgrade() -> None:
    # Instruction steps have no SKU, so they cannot survive a column that
    # forbids nulls. Removing them is the only honest way back.
    op.execute("DELETE FROM edit_steps WHERE replacement_sku IS NULL")
    with op.batch_alter_table("edit_steps") as batch:
        batch.alter_column("replacement_sku", existing_type=sa.String(128),
                           nullable=False)
    op.drop_column("edit_steps", "instruction")