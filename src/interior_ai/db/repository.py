"""Repository layer -- persistence for scenes, prices, and quotes.

The repositories are deliberately thin. Business rules live in the core
package; this layer's only job is to make the storage honour the same
invariants the in-memory objects do:

* a scene version is inserted, never updated
* a price observation is appended, and the current-price projection follows
* a quote line copies its price rather than referencing it

:class:`SqlPriceBook` implements the same interface as the in-memory
:class:`~interior_ai.pricing.prices.PriceBook`, so the pricing engine works
against either without knowing which it has.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.enums import PriceStatus, Unit
from ..core.scene import Scene
from ..pricing.engine import Quote
from ..pricing.prices import PriceObservation, PriceSnapshot, STALE_AFTER_DAYS
from .models import (
    Base,
    PriceCurrent,
    PriceHistory,
    Project,
    QuoteLine,
    QuoteRecord,
    SceneVersion,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    """Force timezone-awareness.

    SQLite drops tzinfo on round-trip. Comparing a naive datetime to an aware
    one raises, and it would raise inside price staleness arithmetic -- the
    worst place for it, since that path decides whether money is trustworthy.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def make_engine(url: str | None = None, **kwargs) -> Engine:
    """Build an engine, normalising the URL and tuning for serverless Postgres.

    Neon (and most hosted Postgres) hand out ``postgresql://`` URLs, while
    SQLAlchemy 2 needs an explicit driver. Neon also *suspends idle
    computes*, so a pooled connection that worked a minute ago may be dead:
    ``pool_pre_ping`` checks liveness before handing a connection out, and a
    short ``pool_recycle`` stops us clinging to sockets the far end has
    already dropped. Without these the first request after an idle period
    fails with a confusing "server closed the connection" error.
    """
    import os

    url = url or os.getenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    if url.startswith("postgresql+psycopg://"):
        defaults = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "pool_size": 5,
            "max_overflow": 5,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        # Neon requires TLS; make that explicit rather than relying on the
        # caller having pasted the sslmode parameter.
        if "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"

    return create_engine(url, future=True, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create tables directly. Alembic owns real deployments; this is for
    tests and local spin-up."""
    Base.metadata.create_all(engine)


# ------------------------------------------------------------- scenes


class SceneRepository:
    """Append-only scene version storage."""

    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _serialise(scene: Scene) -> dict:
        return json.loads(scene.model_dump_json())

    def save(self, scene: Scene, *, project_id: str | None = None) -> SceneVersion:
        """Insert a scene version.

        Re-saving an existing ``version_id`` is a no-op rather than an error --
        the pipeline can legitimately try to persist the same version twice
        (once from the orchestrator, once from the API layer), and that is not
        a conflict.
        """
        existing = self.session.get(SceneVersion, scene.version_id)
        if existing is not None:
            return existing

        row = SceneVersion(
            version_id=scene.version_id,
            scene_id=scene.id,
            parent_version_id=scene.parent_version_id,
            version=scene.version,
            project_id=project_id or scene.project_id,
            payload=self._serialise(scene),
            notes=scene.notes,
            created_at=scene.created_at,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get_version(self, version_id: str) -> Scene | None:
        row = self.session.get(SceneVersion, version_id)
        if row is None:
            return None
        return Scene.model_validate(row.payload)

    def get_latest(self, scene_id: str) -> Scene | None:
        stmt = (
            select(SceneVersion)
            .where(SceneVersion.scene_id == scene_id)
            .order_by(SceneVersion.version.desc())
            .limit(1)
        )
        row = self.session.execute(stmt).scalar_one_or_none()
        return Scene.model_validate(row.payload) if row else None

    def list_versions(self, scene_id: str) -> list[SceneVersion]:
        stmt = (
            select(SceneVersion)
            .where(SceneVersion.scene_id == scene_id)
            .order_by(SceneVersion.version.asc())
        )
        return list(self.session.execute(stmt).scalars())

    def lineage(self, version_id: str) -> list[SceneVersion]:
        """Walk parent links from a version back to the root.

        This is what makes an old quote defensible: given the version a quote
        names, you can reconstruct every edit that led to it.
        """
        out: list[SceneVersion] = []
        seen: set[str] = set()
        current = self.session.get(SceneVersion, version_id)
        while current is not None and current.version_id not in seen:
            out.append(current)
            seen.add(current.version_id)
            if current.parent_version_id is None:
                break
            current = self.session.get(SceneVersion, current.parent_version_id)
        return out


# ------------------------------------------------------------- prices


class PriceRepository:
    """Append-only price history plus its current-price projection."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, obs: PriceObservation) -> PriceHistory:
        row = PriceHistory(
            sku=obs.sku,
            vendor=obs.vendor,
            unit=str(obs.unit),
            amount=obs.amount,
            currency=obs.currency,
            observed_at=obs.observed_at,
            source=obs.source,
        )
        self.session.add(row)
        self.session.flush()
        self._project(row)
        return row

    def _project(self, row: PriceHistory) -> None:
        """Update ``price_current`` if this observation is the newest.

        Guarded on ``observed_at`` rather than insert order: backfilling an old
        observation is legitimate, and it must not clobber a newer price just
        because it arrived later.
        """
        current = self.session.get(PriceCurrent, row.sku)
        row_at = _aware(row.observed_at)
        if current is not None and _aware(current.observed_at) > row_at:
            return

        if current is None:
            current = PriceCurrent(sku=row.sku)
            self.session.add(current)

        current.vendor = row.vendor
        current.unit = row.unit
        current.amount = row.amount
        current.currency = row.currency
        current.observed_at = row.observed_at
        current.history_id = row.id
        self.session.flush()

    def history_for(self, sku: str) -> list[PriceHistory]:
        stmt = (
            select(PriceHistory)
            .where(PriceHistory.sku == sku)
            .order_by(PriceHistory.observed_at.asc())
        )
        return list(self.session.execute(stmt).scalars())

    def current(self, sku: str) -> PriceCurrent | None:
        return self.session.get(PriceCurrent, sku)

    def rebuild_projection(self) -> int:
        """Recompute ``price_current`` from history.

        Exists because the projection is derived data. If it ever disagrees
        with the log, the log wins and this repairs it -- no manual surgery.
        """
        self.session.execute(delete(PriceCurrent))
        self.session.flush()

        rows = list(
            self.session.execute(
                select(PriceHistory).order_by(PriceHistory.observed_at.asc())
            ).scalars()
        )
        latest: dict[str, PriceHistory] = {}
        for r in rows:
            prev = latest.get(r.sku)
            if prev is None or _aware(r.observed_at) >= _aware(prev.observed_at):
                latest[r.sku] = r

        for sku, r in latest.items():
            self.session.add(
                PriceCurrent(
                    sku=sku,
                    vendor=r.vendor,
                    unit=r.unit,
                    amount=r.amount,
                    currency=r.currency,
                    observed_at=r.observed_at,
                    history_id=r.id,
                )
            )
        self.session.flush()
        return len(latest)


class SqlPriceBook:
    """Database-backed PriceBook.

    Interface-compatible with the in-memory version, so
    :class:`~interior_ai.pricing.engine.PricingEngine` cannot tell them apart.
    """

    def __init__(self, session: Session, *, stale_after_days: int = STALE_AFTER_DAYS) -> None:
        self.repo = PriceRepository(session)
        self._stale_after = timedelta(days=stale_after_days)

    def record(self, obs: PriceObservation) -> None:
        self.repo.record(obs)

    def record_many(self, observations: list[PriceObservation]) -> None:
        for o in observations:
            self.record(o)

    def history_for(self, sku: str) -> tuple[PriceObservation, ...]:
        return tuple(
            PriceObservation(
                sku=r.sku,
                vendor=r.vendor,
                unit=Unit(r.unit),
                amount=Decimal(str(r.amount)),
                currency=r.currency,
                observed_at=_aware(r.observed_at),
                source=r.source,
            )
            for r in self.repo.history_for(sku)
        )

    def current(self, sku: str) -> PriceObservation | None:
        row = self.repo.current(sku)
        if row is None:
            return None
        return PriceObservation(
            sku=row.sku,
            vendor=row.vendor,
            unit=Unit(row.unit),
            amount=Decimal(str(row.amount)),
            currency=row.currency,
            observed_at=_aware(row.observed_at),
        )

    def snapshot(self, sku: str, *, as_of: datetime | None = None) -> PriceSnapshot:
        as_of = as_of or _now()
        obs = self.current(sku)
        if obs is None:
            return PriceSnapshot(
                sku=sku,
                vendor=None,
                unit=None,
                amount=None,
                currency="INR",
                observed_at=None,
                status=PriceStatus.UNPRICED,
                note="no observation in price history",
            )

        age = as_of - obs.observed_at
        age_days = max(0, age.days)
        status = PriceStatus.STALE if age > self._stale_after else PriceStatus.FRESH
        return PriceSnapshot(
            sku=obs.sku,
            vendor=obs.vendor,
            unit=obs.unit,
            amount=obs.amount,
            currency=obs.currency,
            observed_at=obs.observed_at,
            status=status,
            age_days=age_days,
            note=(
                f"price is {age_days} days old, older than the "
                f"{self._stale_after.days}-day freshness window"
                if status is PriceStatus.STALE
                else None
            ),
        )

    def snapshot_many(
        self, skus: list[str], *, as_of: datetime | None = None
    ) -> dict[str, PriceSnapshot]:
        as_of = as_of or _now()
        return {s: self.snapshot(s, as_of=as_of) for s in skus}


# ------------------------------------------------------------- quotes


class QuoteRepository:
    """Persists quotes with their prices frozen into the lines."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, quote: Quote, *, quote_id: str | None = None) -> QuoteRecord:
        qid = quote_id or uuid.uuid4().hex
        record = QuoteRecord(
            id=qid,
            scene_id=quote.scene_id,
            scene_version_id=quote.scene_version_id,
            currency=quote.currency,
            total=quote.total,
            stale_total=quote.stale_total,
            is_complete=1 if quote.is_complete else 0,
            warnings=list(quote.warnings()),
            generated_at=quote.generated_at,
        )
        self.session.add(record)

        for line in quote.lines:
            self.session.add(
                QuoteLine(
                    quote_id=qid,
                    sku=line.sku,
                    description=line.description,
                    quantity=line.quantity,
                    unit=str(line.unit),
                    basis=line.basis,
                    room_id=line.room_id,
                    price_status=str(line.price.status),
                    vendor=line.price.vendor,
                    unit_price=line.price.amount,
                    line_total=line.line_total,
                    price_observed_at=line.price.observed_at,
                    price_age_days=line.price.age_days,
                )
            )
        self.session.flush()
        return record

    def get(self, quote_id: str) -> QuoteRecord | None:
        return self.session.get(QuoteRecord, quote_id)

    def for_scene_version(self, version_id: str) -> list[QuoteRecord]:
        stmt = select(QuoteRecord).where(QuoteRecord.scene_version_id == version_id)
        return list(self.session.execute(stmt).scalars())


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, name: str, *, client_name: str | None = None, project_id: str | None = None) -> Project:
        row = Project(
            id=project_id or uuid.uuid4().hex, name=name, client_name=client_name
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get(self, project_id: str) -> Project | None:
        return self.session.get(Project, project_id)