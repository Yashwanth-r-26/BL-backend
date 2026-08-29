"""Accounts, and design ownership.

Adds the users table and a nullable owner on edit_sessions.

Nullable is the whole point of the migration's shape: there is no sensible
user to backfill existing sessions with, and inventing a placeholder owner
would either hide them from everybody or hand them to somebody. They stay
unowned, and every endpoint keeps serving them exactly as before.

Revision ID: 0005_accounts
Revises: 0004_location_context
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_accounts"
down_revision = "0004_location_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("display_name", sa.String(80), nullable=False, server_default=""),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Unique on the stored (already lower-cased) address, so two accounts
    # cannot differ only by capitalisation.
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.add_column("edit_sessions", sa.Column("user_id", sa.String(64), nullable=True))
    op.add_column("edit_sessions", sa.Column("title", sa.String(120), nullable=True))
    op.create_index("ix_edit_sessions_user_id", "edit_sessions", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_edit_sessions_user_id", table_name="edit_sessions")
    # Batch mode so this also works on SQLite, where dropping a column means
    # rebuilding the table. Postgres ignores the ceremony and drops directly.
    with op.batch_alter_table("edit_sessions") as batch:
        batch.drop_column("title")
        batch.drop_column("user_id")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
