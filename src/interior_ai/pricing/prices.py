"""Price history, current projection, and frozen snapshots.

A quote is a promise about money, and it has to still be explainable months
later when someone disputes it. That rules out reading live prices at render
time: the number on the PDF and the number in the database drift apart the
moment a vendor updates a rate, and there is then no way to prove what was
quoted.

So the model is three layers:

1. ``price_history`` -- append-only. Observations are never updated or deleted,
   only superseded by a newer row for the same (sku, vendor).
2. ``price_current`` -- a *projection*: the latest observation per sku. Derived,
   never authoritative, safe to rebuild from history at any time.
3. ``PriceSnapshot`` -- frozen into the BOQ line at quote time, carrying vendor
   and ``observed_at``. This is what makes a March quote reproducible in
   September.

Two policies that look like small details and are not:

**Stale prices are flagged, not hidden.** An observation older than 7 days is
still the best information available. Suppressing it produces a quote with a
hole; showing it with a staleness marker lets a human decide.

**Unpriced items surface rather than defaulting to zero.** A missing price that
silently becomes 0.00 produces a quote that is confidently too cheap, and the
error is invisible until someone builds the thing. An UNPRICED line is loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ..core.enums import PriceStatus, Unit

STALE_AFTER_DAYS = 7


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PriceObservation:
    """One immutable price sighting. Rows of ``price_history``."""

    sku: str
    vendor: str
    unit: Unit
    amount: Decimal
    currency: str = "INR"
    observed_at: datetime = field(default_factory=_now)
    source: str | None = None

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("price cannot be negative")


@dataclass(frozen=True)
class PriceSnapshot:
    """A price frozen into a quote line.

    Carries everything needed to defend the number later: what it was, who
    quoted it, when it was seen, and whether it was already stale when used.
    """

    sku: str
    vendor: str | None
    unit: Unit | None
    amount: Decimal | None
    currency: str
    observed_at: datetime | None
    status: PriceStatus
    age_days: int | None = None
    note: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.status is not PriceStatus.UNPRICED

    def explain(self) -> str:
        if self.status is PriceStatus.UNPRICED:
            return f"{self.sku}: no price on record -- must be quoted manually"
        assert self.observed_at is not None
        base = (
            f"{self.sku}: {self.currency} {self.amount} per {self.unit} "
            f"from {self.vendor}, observed {self.observed_at.date().isoformat()}"
        )
        if self.status is PriceStatus.STALE:
            return base + f" (STALE -- {self.age_days} days old)"
        return base


class PriceBook:
    """Append-only price store with a current-price projection.

    In Postgres this is two tables; in memory it is a list plus a dict. The
    semantics are identical, which is deliberate -- the repository layer swaps
    the backing store without changing how pricing behaves.
    """

    def __init__(self, *, stale_after_days: int = STALE_AFTER_DAYS) -> None:
        self._history: list[PriceObservation] = []
        self._stale_after = timedelta(days=stale_after_days)

    # ------------------------------------------------------------ writes

    def record(self, obs: PriceObservation) -> None:
        """Append an observation. Never overwrites; history is immutable."""
        self._history.append(obs)

    def record_many(self, observations: list[PriceObservation]) -> None:
        for o in observations:
            self.record(o)

    # ------------------------------------------------------------- reads

    @property
    def history(self) -> tuple[PriceObservation, ...]:
        return tuple(self._history)

    def history_for(self, sku: str) -> tuple[PriceObservation, ...]:
        return tuple(o for o in self._history if o.sku == sku)

    def current(self, sku: str) -> PriceObservation | None:
        """Latest observation for a sku -- the ``price_current`` projection.

        Recomputed from history rather than cached, so the projection can never
        disagree with the log it is derived from.
        """
        candidates = [o for o in self._history if o.sku == sku]
        if not candidates:
            return None
        return max(candidates, key=lambda o: o.observed_at)

    def current_all(self) -> dict[str, PriceObservation]:
        out: dict[str, PriceObservation] = {}
        for o in self._history:
            prev = out.get(o.sku)
            if prev is None or o.observed_at > prev.observed_at:
                out[o.sku] = o
        return out

    # --------------------------------------------------------- snapshots

    def snapshot(self, sku: str, *, as_of: datetime | None = None) -> PriceSnapshot:
        """Freeze the current price for a sku into a quote-ready snapshot."""
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
