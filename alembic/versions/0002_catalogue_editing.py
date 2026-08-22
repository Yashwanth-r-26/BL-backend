"""catalogue items + edit sessions/steps

Revision ID: 0002_catalogue_editing
Revises: 0001_initial
Create Date: 2026-07-27

Adds the store catalogue (products offered when a user selects an object in
their photo) and the photo-editing session tables (append-only step chains).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_catalogue_editing"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalogue_items",
        sa.Column("sku", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("object_class", sa.String(64), nullable=False, index=True),
        sa.Column("description", sa.Text),
        sa.Column("width_mm", sa.Integer, nullable=False),
        sa.Column("depth_mm", sa.Integer, nullable=False),
        sa.Column("height_mm", sa.Integer, nullable=False),
        sa.Column("image_ref", sa.Text),
        sa.Column("display_price", sa.Numeric(14, 2), nullable=False),
        sa.Column("currency", sa.String(8), nullable=False, server_default="INR"),
        sa.Column("vendor", sa.String(128)),
        sa.Column("style_tags", sa.JSON, nullable=False),
        sa.Column("active", sa.Integer, nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("width_mm > 0 AND depth_mm > 0 AND height_mm > 0", name="ck_cat_dims"),
        sa.CheckConstraint("display_price >= 0", name="ck_cat_price"),
    )
    op.create_index("ix_catalogue_class_active", "catalogue_items", ["object_class", "active"])

    op.create_table(
        "edit_sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scene_id", sa.String(64), nullable=False, index=True),
        sa.Column("room_id", sa.String(64), nullable=False),
        sa.Column("original_image_ref", sa.Text, nullable=False),
        sa.Column("detections", sa.JSON, nullable=False),
        sa.Column("current_step_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "edit_steps",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(64),
                  sa.ForeignKey("edit_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_step_id", sa.Integer,
                  sa.ForeignKey("edit_steps.id", ondelete="RESTRICT")),
        sa.Column("detection_id", sa.String(64), nullable=False),
        sa.Column("detection_label", sa.String(128), nullable=False),
        sa.Column("replacement_sku", sa.String(128),
                  sa.ForeignKey("catalogue_items.sku", ondelete="RESTRICT"), nullable=False),
        sa.Column("result_image_ref", sa.Text, nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("notes", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_edit_steps_session", "edit_steps", ["session_id"])


def downgrade() -> None:
    op.drop_table("edit_steps")
    op.drop_table("edit_sessions")
    op.drop_index("ix_catalogue_class_active", table_name="catalogue_items")
    op.drop_table("catalogue_items")