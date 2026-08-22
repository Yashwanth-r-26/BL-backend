"""Bill of quantities -- takeoff joined to frozen prices.

This is where geometry becomes money. Each BOQ line pairs a derived quantity
with a :class:`~interior_ai.pricing.prices.PriceSnapshot` frozen at quote time,
so the line total can be recomputed and defended long after vendor rates have
moved on.

The totals deliberately separate three buckets:

* ``total`` -- money we can actually stand behind.
* ``stale_total`` -- included in the total, but flagged. The price is the best
  available; a human decides whether to re-verify.
* ``unpriced_lines`` -- excluded from the total and listed separately. These
  are the holes in the quote, and they are loud on purpose. A missing price
  that quietly becomes zero produces a confident underquote that nobody
  notices until the invoice arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from ..core.enums import PriceStatus, Unit
from ..core.scene import Room, Scene
from .prices import PriceBook, PriceSnapshot
from .takeoff import TakeoffLine, room_takeoff


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _money(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class BOQLine:
    """A priced quantity. Immutable once created -- this is quote evidence."""

    sku: str
    description: str
    quantity: Decimal
    unit: Unit
    price: PriceSnapshot
    basis: str
    room_id: str | None = None

    @property
    def line_total(self) -> Decimal | None:
        if self.price.amount is None:
            return None
        return _money(self.quantity * self.price.amount)

    @property
    def status(self) -> PriceStatus:
        return self.price.status

    def explain(self) -> str:
        total = self.line_total
        money = f"{self.price.currency} {total}" if total is not None else "UNPRICED"
        return f"{self.description} -- {self.quantity} {self.unit} -> {money}"


@dataclass(frozen=True)
class Quote:
    """A complete, reproducible bill of quantities.

    ``scene_version_id`` is the load-bearing field: it names the exact
    immutable scene this quote was derived from, so the whole document can be
    regenerated and checked byte-for-byte months later.
    """

    scene_id: str
    scene_version_id: str
    lines: tuple[BOQLine, ...]
    currency: str = "INR"
    generated_at: datetime = field(default_factory=_now)

    @property
    def priced_lines(self) -> tuple[BOQLine, ...]:
        return tuple(l for l in self.lines if l.price.is_usable)

    @property
    def unpriced_lines(self) -> tuple[BOQLine, ...]:
        return tuple(l for l in self.lines if not l.price.is_usable)

    @property
    def stale_lines(self) -> tuple[BOQLine, ...]:
        return tuple(l for l in self.lines if l.status is PriceStatus.STALE)

    @property
    def total(self) -> Decimal:
        return _money(
            sum((l.line_total or Decimal(0) for l in self.priced_lines), Decimal(0))
        )

    @property
    def stale_total(self) -> Decimal:
        return _money(
            sum((l.line_total or Decimal(0) for l in self.stale_lines), Decimal(0))
        )

    @property
    def is_complete(self) -> bool:
        """Whether every line carries a price. False means holes exist."""
        return not self.unpriced_lines

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.unpriced_lines:
            skus = ", ".join(sorted({l.sku for l in self.unpriced_lines}))
            out.append(
                f"{len(self.unpriced_lines)} line(s) have no price on record "
                f"and are excluded from the total: {skus}"
            )
        if self.stale_lines:
            oldest = max(
                (l.price.age_days or 0 for l in self.stale_lines), default=0
            )
            out.append(
                f"{len(self.stale_lines)} line(s) use prices older than the "
                f"freshness window (oldest {oldest} days); "
                f"{self.currency} {self.stale_total} of the total is affected"
            )
        return tuple(out)


class PricingEngine:
    """Turns a scene into a quote."""

    def __init__(self, book: PriceBook) -> None:
        self.book = book

    def price_lines(
        self, lines: list[TakeoffLine], *, as_of: datetime | None = None
    ) -> list[BOQLine]:
        as_of = as_of or _now()
        out: list[BOQLine] = []
        for tl in lines:
            snap = self.book.snapshot(tl.sku, as_of=as_of)
            out.append(
                BOQLine(
                    sku=tl.sku,
                    description=tl.description,
                    quantity=tl.quantity,
                    unit=tl.unit,
                    price=snap,
                    basis=tl.basis,
                    room_id=tl.room_id,
                )
            )
        return out

    def quote_room(
        self, scene: Scene, room: Room, *, as_of: datetime | None = None, **takeoff_kwargs
    ) -> Quote:
        lines = room_takeoff(room, **takeoff_kwargs)
        return Quote(
            scene_id=scene.id,
            scene_version_id=scene.version_id,
            lines=tuple(self.price_lines(lines, as_of=as_of)),
        )

    def quote_scene(
        self, scene: Scene, *, as_of: datetime | None = None, **takeoff_kwargs
    ) -> Quote:
        """Price every room in the scene against one consistent timestamp.

        The shared ``as_of`` matters: pricing rooms at slightly different
        instants could classify the same sku as fresh in one room and stale in
        another, in a single document.
        """
        as_of = as_of or _now()
        all_lines: list[TakeoffLine] = []
        for room in scene.rooms:
            all_lines.extend(room_takeoff(room, **takeoff_kwargs))
        return Quote(
            scene_id=scene.id,
            scene_version_id=scene.version_id,
            lines=tuple(self.price_lines(all_lines, as_of=as_of)),
        )
