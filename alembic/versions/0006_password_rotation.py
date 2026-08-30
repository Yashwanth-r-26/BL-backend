"""Record when a password last changed, so old tokens can be refused.

Tokens here are 30-day JWTs with no server-side session table and no
revocation list. Without this column, changing a password would not sign
anybody else out: a token minted before the change stays valid for up to a
month, which makes "change my password" useless in the one situation it
matters -- somebody else has it.

Storing the moment of change lets every request compare the token's `iat`
against it and refuse anything older. Nullable, because every account that
existed before this has never rotated and every one of its tokens is still
legitimate.

Revision ID: 0006_password_rotation
Revises: 0005_accounts
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_password_rotation"
down_revision = "0005_accounts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("password_changed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("password_changed_at")
