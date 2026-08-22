"""SQLAlchemy models.

Two structural decisions worth stating.

**Scene versions are rows, not updates.** ``scene_versions`` has no UPDATE path
in the repository. A new version is an INSERT pointing at its parent. The
scene graph promises immutability; storing it in a mutable table would make
that promise unenforceable the first time someone writes a quick fix.

**Prices are two tables, one of them derived.** ``price_history`` is
append-only truth; ``price_current`` is a projection that can be dropped and
rebuilt from history at any time. Keeping the projection lets the hot path do
one indexed lookup instead of a window function over the whole log.

Room geometry is stored as JSONB rather than PostGIS. The geometry work happens
in Shapely and CP-SAT, not in SQL -- adding a spatial extension would buy
indexing we do not query on, and cost a dependency in every environment.
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
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


# JSONB on Postgres, plain JSON elsewhere -- lets the suite run on SQLite.
JSONType = JSONB().with_variant(JSON(), "sqlite")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    client_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    scenes: Mapped[list["SceneVersion"]] = relationship(back_populates="project")


class SceneVersion(Base):
    """One immutable scene version.

    ``scene_id`` groups versions of the same scene; ``version_id`` is the
    primary key. ``parent_version_id`` is a self-reference forming the audit
    chain a quote can be replayed against.
    """

    __tablename__ = "scene_versions"

    version_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_version_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("scene_versions.version_id", ondelete="RESTRICT")
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    project_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="SET NULL")
    )
    payload: Mapped[dict] = mapped_column(JSONType, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False, index=True
    )

    project: Mapped[Project | None] = relationship(back_populates="scenes")
    quotes: Mapped[list["QuoteRecord"]] = relationship(back_populates="scene_version")

    __table_args__ = (
        UniqueConstraint("scene_id", "version", name="uq_scene_version"),
        CheckConstraint("version > 0", name="ck_scene_version_positive"),
        Index("ix_scene_versions_scene_version", "scene_id", "version"),
    )


class PriceHistory(Base):
    """Append-only price observations.

    No UPDATE, no DELETE. A correction is a new row with a later
    ``observed_at``; the wrong number stays visible, which is the point -- a
    quote that used it must still be explainable.
    """

    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    vendor: Mapped[str] = mapped_column(String(128), nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )
    source: Mapped[str | None] = mapped_column(String(255))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_price_non_negative"),
        Index("ix_price_history_sku_observed", "sku", "observed_at"),
    )


class PriceCurrent(Base):
    """Latest-price projection. Derived; safe to rebuild from history."""

    __tablename__ = "price_current"

    sku: Mapped[str] = mapped_column(String(128), primary_key=True)
    vendor: Mapped[str] = mapped_column(String(128), nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    history_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("price_history.id", ondelete="RESTRICT"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    __table_args__ = (CheckConstraint("amount >= 0", name="ck_current_non_negative"),)


class QuoteRecord(Base):
    """A generated quote, pinned to the scene version it was priced from."""

    __tablename__ = "quotes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    scene_version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("scene_versions.version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="INR")
    total: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    stale_total: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False, default=0)
    is_complete: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    warnings: Mapped[dict] = mapped_column(JSONType, nullable=False, default=list)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    scene_version: Mapped[SceneVersion] = relationship(back_populates="quotes")
    lines: Mapped[list["QuoteLine"]] = relationship(
        back_populates="quote", cascade="all, delete-orphan"
    )


class QuoteLine(Base):
    """A BOQ line with its price frozen at quote time.

    The vendor, unit price, and ``price_observed_at`` are *copied* here rather
    than joined from ``price_current``. Joining would make historical quotes
    change when vendor rates move, which defeats the whole design.
    """

    __tablename__ = "quote_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    quote_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("quotes.id", ondelete="CASCADE"), nullable=False
    )
    sku: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(14, 3), nullable=False)
    unit: Mapped[str] = mapped_column(String(16), nullable=False)
    basis: Mapped[str] = mapped_column(Text, nullable=False)
    room_id: Mapped[str | None] = mapped_column(String(64))

    price_status: Mapped[str] = mapped_column(String(16), nullable=False)
    vendor: Mapped[str | None] = mapped_column(String(128))
    unit_price: Mapped[float | None] = mapped_column(Numeric(14, 2))
    line_total: Mapped[float | None] = mapped_column(Numeric(14, 2))
    price_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    price_age_days: Mapped[int | None] = mapped_column(Integer)

    quote: Mapped[QuoteRecord] = relationship(back_populates="lines")

    __table_args__ = (
        Index("ix_quote_lines_quote", "quote_id"),
        CheckConstraint("quantity >= 0", name="ck_quantity_non_negative"),
    )
