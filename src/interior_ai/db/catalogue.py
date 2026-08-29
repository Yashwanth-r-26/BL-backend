"""Catalogue and photo-editing session storage.

Two additions to the schema:

**catalogue_items** -- the store's products. This is what the overlay offers
when a user selects an object in their photo: "here are 5 sofas we sell".
Each row carries the object class it can replace, physical dimensions (so the
fit engine can veto a sofa that will not fit the room), a reference image, and
a display price. Prices here are *display* prices for the picker; the
authoritative money still flows through price_history -> snapshot at quote
time, keyed by the same sku.

**edit_sessions / edit_steps** -- the iteration loop. A user swaps objects in
their photo as many times as they like before committing. Each step is
append-only: the step records which detected object was replaced, with which
catalogue sku, and the resulting image reference. Append-only for the same
reason scene versions are -- "undo" is just pointing at an earlier step, and
the final quote can name the exact step it priced.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .models import Base, JSONType


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CatalogueItemRow(Base):
    """A purchasable product the user can swap into their photo."""

    __tablename__ = "catalogue_items"

    sku: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    object_class: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    # Physical dimensions -- the fit engine uses these to veto items that
    # cannot physically fit the room, before they are ever offered.
    width_mm: Mapped[int] = mapped_column(Integer, nullable=False)
    depth_mm: Mapped[int] = mapped_column(Integer, nullable=False)
    height_mm: Mapped[int] = mapped_column(Integer, nullable=False)
    # Reference image of the product, used both in the picker UI and as the
    # visual reference handed to the image-edit model for the swap.
    image_ref: Mapped[str | None] = mapped_column(Text)
    # Display price for the picker. Authoritative pricing still goes through
    # price_history at quote time under the same sku.
    display_price: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    vendor: Mapped[str | None] = mapped_column(String(128))
    style_tags: Mapped[dict] = mapped_column(JSONType, nullable=False, default=list)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    __table_args__ = (
        CheckConstraint("width_mm > 0 AND depth_mm > 0 AND height_mm > 0", name="ck_cat_dims"),
        CheckConstraint("display_price >= 0", name="ck_cat_price"),
        Index("ix_catalogue_class_active", "object_class", "active"),
    )


class EditSession(Base):
    """One user's photo-editing loop against one scene room.

    Holds the original photo and the detected objects (as JSON), and owns an
    ordered chain of steps. ``current_step_id`` points at whichever step the
    user currently considers "the image" -- undo/redo is moving this pointer,
    never deleting steps.
    """

    __tablename__ = "edit_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: Who owns this design. Nullable, and deliberately so: every session
    #: created before accounts existed has no owner, and the operator console
    #: still creates sessions with no user at all. Ownership is a claim a
    #: session may carry, not a requirement -- an unowned session behaves
    #: exactly as it did before.
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: What to call this design in a list. Derived from the room type at
    #: creation; the alternative is a list of identical rows.
    title: Mapped[str | None] = mapped_column(String(120))
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    room_id: Mapped[str] = mapped_column(String(64), nullable=False)
    original_image_ref: Mapped[str] = mapped_column(Text, nullable=False)
    # Detected objects: [{id, label, object_class, box:[x0,y0,x1,y1] in
    # normalised 0-1000 coords, confidence}]
    detections: Mapped[dict] = mapped_column(JSONType, nullable=False, default=list)
    current_step_id: Mapped[int | None] = mapped_column(Integer)
    #: Where the room is: country, city, currency, city tier. Quotes are
    #: meaningless without it -- identical work costs different money in
    #: Bengaluru and in a tier-3 town.
    location: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    #: The scope and preference answers that shape a quote: which categories
    #: are in play, quality tier, budget band, timeline, occupancy.
    questionnaire: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    steps: Mapped[list["EditStep"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="EditStep.id"
    )


class EditStep(Base):
    """One replacement in the loop. Append-only.

    ``parent_step_id`` forms the chain (None = branched from the original
    photo). The step records exactly what changed -- which detection, which
    catalogue sku -- so the final quote can enumerate every swap that led to
    the image being priced.
    """

    __tablename__ = "edit_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("edit_sessions.id", ondelete="CASCADE"), nullable=False
    )
    parent_step_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("edit_steps.id", ondelete="RESTRICT")
    )
    detection_id: Mapped[str] = mapped_column(String(64), nullable=False)
    detection_label: Mapped[str] = mapped_column(String(128), nullable=False)
    # Null for a free-text instruction ("paint this wall sage"), which changes
    # the photo without putting a product in it -- and so contributes nothing
    # to the quote.
    replacement_sku: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("catalogue_items.sku", ondelete="RESTRICT")
    )
    #: The user's own words, when the step came from an instruction rather
    #: than a catalogue pick. Kept verbatim so the history reads as what was
    #: actually asked for.
    instruction: Mapped[str | None] = mapped_column(Text)
    result_image_ref: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[dict] = mapped_column(JSONType, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    session: Mapped[EditSession] = relationship(back_populates="steps")

    __table_args__ = (Index("ix_edit_steps_session", "session_id"),)