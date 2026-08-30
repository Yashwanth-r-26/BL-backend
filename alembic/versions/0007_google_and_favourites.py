"""Google identities, saved products, and the OAuth hand-back.

Three additions, all of them consequences of the app becoming accounts-only:

**users.google_sub / password_hash nullable.** An account created through
Google has no password and never will. Storing a random unusable hash to
satisfy a NOT NULL would be a lie the login path then has to see through;
nullable says what is true, and `verify_password` refuses a null outright, so
password sign-in is simply impossible for those accounts.

**user_favourites.** The wishlist was a list of SKUs in device storage. Once
the app requires an account there is no reason for it not to follow that
account, and every reason for it to.

**oauth_exchanges.** Google redirects to the backend, not to the app, because
Google will not accept a custom scheme as a redirect URI. The backend therefore
has to hand the result back across a deep link, and a bearer token in a URL can
end up in logs. Instead it hands over a single-use code with a short life,
which the app trades for a token over POST. That needs somewhere to record
which codes have been spent -- a signed token could not, because anything
self-contained is replayable until it expires.

Revision ID: 0007_google_and_favourites
Revises: 0006_password_rotation
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_google_and_favourites"
down_revision = "0006_password_rotation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_sub", sa.String(64), nullable=True))
    op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)
    with op.batch_alter_table("users") as batch:
        batch.alter_column("password_hash", existing_type=sa.String(255), nullable=True)

    op.create_table(
        "user_favourites",
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("sku", sa.String(128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("user_id", "sku"),
    )
    op.create_index("ix_user_favourites_user_id", "user_favourites", ["user_id"])

    op.create_table(
        "oauth_exchanges",
        sa.Column("code", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("oauth_exchanges")
    op.drop_index("ix_user_favourites_user_id", table_name="user_favourites")
    op.drop_table("user_favourites")
    op.drop_index("ix_users_google_sub", table_name="users")
    with op.batch_alter_table("users") as batch:
        batch.alter_column("password_hash", existing_type=sa.String(255), nullable=False)
        batch.drop_column("google_sub")
