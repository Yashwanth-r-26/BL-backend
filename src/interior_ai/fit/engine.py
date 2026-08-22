"""Fit engine -- pure geometry, no ML.

"Will this sofa fit here" is a geometry question with a provable answer. Asking
a model to guess at it produces something that is right most of the time and
unaccountable when it is wrong. This module proves it instead.

Two design rules:

**Cheapest-first gating.** Gates run in increasing cost order -- integer
comparisons before polygon intersections. An object 800 mm too wide is rejected
by three subtractions; there is no reason to build its Shapely box first.

**Every rejection is measured.** A gate never returns bare "cannot place". It
returns "too wide by 800 mm", because the caller's next move is to suggest a
smaller item or a different wall, and it cannot do either from a boolean. This
is what ``overage_mm`` on every :class:`Rejection` is for.

Gate order:
    1. width          2. depth         3. height
    4. wall-requirement                5. room containment
    6. collision      7. door swing    8. front clearance
    9. circulation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from shapely.geometry import Polygon

from ..core.enums import RejectionCode
from ..core.geometry import (
    all_door_swings,
    blocked_region,
    distance_to_nearest_wall,
    front_clearance_polygon,
    obstacle_polygon,
    placement_polygon,
    room_polygon,
)
from ..core.scene import CatalogueItem, Placement, Room, Vec2
from ..core.units import apply_tolerance

DEFAULT_TOLERANCE = 0.08
WALL_SNAP_MM = 150
DEFAULT_CIRCULATION_MM = 600


@dataclass(frozen=True)
class Rejection:
    """A single failed gate, with the measurement that failed it."""

    code: RejectionCode
    message: str
    overage_mm: int
    detail: dict[str, int] | None = None

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class FitResult:
    """Outcome of testing one placement.

    ``rejections`` may hold more than one entry when ``collect_all`` is set --
    useful for UI ("this fails on three counts") but off by default because
    short-circuiting is the whole point of ordering the gates.
    """

    ok: bool
    placement: Placement | None
    rejections: tuple[Rejection, ...] = ()

    @property
    def first_reason(self) -> str | None:
        return self.rejections[0].message if self.rejections else None

    def codes(self) -> tuple[RejectionCode, ...]:
        return tuple(r.code for r in self.rejections)


def _room_span(room: Room) -> tuple[int, int]:
    minx, miny, maxx, maxy = room.bounds
    return (maxx - minx, maxy - miny)


class FitEngine:
    """Geometric feasibility checker.

    ``tolerance`` expands the *room's* allowances, not the object -- a 2410 mm
    sofa against a 2400 mm wall is a measurement artefact from a phone camera,
    not a real conflict, and rejecting it makes the product feel broken.
    """

    def __init__(
        self,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        circulation_mm: int = DEFAULT_CIRCULATION_MM,
        wall_snap_mm: int = WALL_SNAP_MM,
    ) -> None:
        self.tolerance = tolerance
        self.circulation_mm = circulation_mm
        self.wall_snap_mm = wall_snap_mm

    # ---------------------------------------------------------------- gates

    def _gate_width(self, item: CatalogueItem, room: Room, yaw: int) -> Rejection | None:
        eff_w, _ = item.footprint.rotated(yaw)  # type: ignore[arg-type]
        span_x, _ = _room_span(room)
        allowed = apply_tolerance(span_x, self.tolerance)
        if eff_w > allowed:
            over = eff_w - span_x
            return Rejection(
                code=RejectionCode.TOO_WIDE,
                message=f"too wide by {over} mm ({eff_w} mm wide, room spans {span_x} mm)",
                overage_mm=over,
                detail={"item_mm": eff_w, "room_mm": span_x},
            )
        return None

    def _gate_depth(self, item: CatalogueItem, room: Room, yaw: int) -> Rejection | None:
        _, eff_d = item.footprint.rotated(yaw)  # type: ignore[arg-type]
        _, span_y = _room_span(room)
        allowed = apply_tolerance(span_y, self.tolerance)
        if eff_d > allowed:
            over = eff_d - span_y
            return Rejection(
                code=RejectionCode.TOO_DEEP,
                message=f"too deep by {over} mm ({eff_d} mm deep, room spans {span_y} mm)",
                overage_mm=over,
                detail={"item_mm": eff_d, "room_mm": span_y},
            )
        return None

    def _gate_height(self, item: CatalogueItem, room: Room) -> Rejection | None:
        h = item.footprint.height_mm
        if h > room.ceiling_height_mm:
            over = h - room.ceiling_height_mm
            return Rejection(
                code=RejectionCode.TOO_TALL,
                message=(
                    f"too tall by {over} mm ({h} mm tall, ceiling {room.ceiling_height_mm} mm)"
                ),
                overage_mm=over,
                detail={"item_mm": h, "ceiling_mm": room.ceiling_height_mm},
            )
        return None

    def _gate_wall(self, item: CatalogueItem, room: Room, placement: Placement) -> Rejection | None:
        if not item.requires_wall:
            return None
        dist = distance_to_nearest_wall(room, placement)
        if dist > self.wall_snap_mm:
            over = int(round(dist - self.wall_snap_mm))
            return Rejection(
                code=RejectionCode.WALL_REQUIRED,
                message=(
                    f"must sit against a wall but stands {int(round(dist))} mm away "
                    f"({over} mm beyond the {self.wall_snap_mm} mm snap distance)"
                ),
                overage_mm=over,
                detail={"distance_mm": int(round(dist)), "snap_mm": self.wall_snap_mm},
            )
        return None

    def _containment_slack_mm(self, room: Room) -> int:
        """How far outside the walls a box may sit before we call it a fault.

        The same measurement-uncertainty allowance the dimension gates use,
        expressed as a linear distance. Without this, tolerance is inconsistent:
        gate 1 waves through a 2410 mm sofa in a 2400 mm room and gate 5 rejects
        it for the 10 mm it just forgave, which reads as a bug from outside.
        """
        span_x, span_y = _room_span(room)
        return int(round(min(span_x, span_y) * self.tolerance / 2.0))

    def _gate_containment(self, room: Room, placement: Placement) -> Rejection | None:
        rp = room_polygon(room)
        pp = placement_polygon(placement)
        if rp.contains(pp):
            return None

        slack = self._containment_slack_mm(room)
        if slack > 0 and rp.buffer(slack).contains(pp):
            return None

        outside = pp.difference(rp)
        area = outside.area
        # Report the linear overhang, which is what a user can act on, rather
        # than an area in mm^2 that means nothing to anyone.
        minx, miny, maxx, maxy = pp.bounds
        rminx, rminy, rmaxx, rmaxy = rp.bounds
        overhang = max(
            rminx - minx,
            rminy - miny,
            maxx - rmaxx,
            maxy - rmaxy,
            0,
        )
        # Net of the slack we already forgave, so the number quoted is the
        # amount by which the placement actually failed.
        over = max(1, int(round(overhang)) - slack)
        return Rejection(
            code=RejectionCode.OUT_OF_ROOM,
            message=(
                f"extends {over} mm outside the room "
                f"({int(round(area / 1e6))} m² of it beyond the walls)"
            ),
            overage_mm=over,
            detail={"overhang_mm": over},
        )

    def _gate_collision(self, room: Room, placement: Placement) -> Rejection | None:
        pp = placement_polygon(placement)
        for other in room.placements:
            if other.id == placement.id:
                continue
            op = placement_polygon(other)
            if pp.intersects(op) and pp.intersection(op).area > 0:
                inter = pp.intersection(op)
                iminx, iminy, imaxx, imaxy = inter.bounds
                over = int(round(min(imaxx - iminx, imaxy - iminy)))
                return Rejection(
                    code=RejectionCode.COLLISION,
                    message=(
                        f"overlaps {other.object_class.value} by {over} mm "
                        f"({int(inter.area / 1000)} cm² of shared floor)"
                    ),
                    overage_mm=over,
                    detail={"overlap_mm": over},
                )
        for obs in room.obstacles:
            op = obstacle_polygon(obs)
            if pp.intersects(op) and pp.intersection(op).area > 0:
                inter = pp.intersection(op)
                iminx, iminy, imaxx, imaxy = inter.bounds
                over = int(round(min(imaxx - iminx, imaxy - iminy)))
                return Rejection(
                    code=RejectionCode.COLLISION,
                    message=f"overlaps fixed obstacle '{obs.label}' by {over} mm",
                    overage_mm=over,
                    detail={"overlap_mm": over},
                )
        return None

    def _gate_door_swing(self, room: Room, placement: Placement) -> Rejection | None:
        pp = placement_polygon(placement)
        for sw in all_door_swings(room):
            if pp.intersects(sw) and pp.intersection(sw).area > 0:
                inter = pp.intersection(sw)
                iminx, iminy, imaxx, imaxy = inter.bounds
                over = int(round(min(imaxx - iminx, imaxy - iminy)))
                return Rejection(
                    code=RejectionCode.DOOR_SWING_BLOCKED,
                    message=(
                        f"blocks the door swing by {over} mm "
                        f"({int(inter.area / 1000)} cm² of the arc)"
                    ),
                    overage_mm=over,
                    detail={"intrusion_mm": over},
                )
        return None

    def _gate_front_clearance(
        self, item: CatalogueItem, room: Room, placement: Placement
    ) -> Rejection | None:
        need = item.clearance_front_mm
        if need <= 0:
            return None
        zone = front_clearance_polygon(placement, need)
        if zone is None or zone.is_empty:
            return None

        rp = room_polygon(room)
        obstructions: list[Polygon] = []
        if not rp.contains(zone):
            obstructions.append(zone.difference(rp))
        for other in room.placements:
            if other.id == placement.id:
                continue
            op = placement_polygon(other)
            if zone.intersects(op):
                obstructions.append(zone.intersection(op))
        for obs in room.obstacles:
            op = obstacle_polygon(obs)
            if zone.intersects(op):
                obstructions.append(zone.intersection(op))

        worst = 0
        for ob in obstructions:
            if ob.is_empty or ob.area <= 0:
                continue
            ominx, ominy, omaxx, omaxy = ob.bounds
            fx, fy = placement.facing_vector()
            depth = (omaxy - ominy) if fx == 0 else (omaxx - ominx)
            worst = max(worst, int(round(depth)))

        if worst > 0:
            return Rejection(
                code=RejectionCode.FRONT_CLEARANCE,
                message=(
                    f"needs {need} mm clear in front but {worst} mm of that is blocked"
                ),
                overage_mm=worst,
                detail={"required_mm": need, "blocked_mm": worst},
            )
        return None

    def _gate_circulation(self, room: Room, placement: Placement) -> Rejection | None:
        """Check a walkable path still exists past this object.

        Approximated as: the free floor left after placing everything must not
        be pinched below the circulation width anywhere adjacent to this item.
        A full path-planning check is overkill for a gate that exists to catch
        "you have walled yourself into the corner".
        """
        rp = room_polygon(room)
        occupied = blocked_region(room, include_placements=True)
        pp = placement_polygon(placement)
        occupied = occupied.union(pp) if not occupied.is_empty else pp

        free = rp.difference(occupied)
        if free.is_empty:
            return Rejection(
                code=RejectionCode.CIRCULATION_BLOCKED,
                message=(
                    f"leaves no free floor at all; {self.circulation_mm} mm of "
                    "walking space is required"
                ),
                overage_mm=self.circulation_mm,
                detail={"required_mm": self.circulation_mm, "available_mm": 0},
            )

        # Erode by half the circulation width: if the free space survives, a
        # corridor of that width fits through it.
        eroded = free.buffer(-self.circulation_mm / 2.0)
        if eroded.is_empty:
            return Rejection(
                code=RejectionCode.CIRCULATION_BLOCKED,
                message=(
                    f"pinches the walkway below {self.circulation_mm} mm of "
                    "clear passage"
                ),
                overage_mm=self.circulation_mm,
                detail={"required_mm": self.circulation_mm},
            )
        return None

    # -------------------------------------------------------------- public

    def check(
        self,
        item: CatalogueItem,
        room: Room,
        origin: Vec2,
        yaw: int = 0,
        *,
        collect_all: bool = False,
        skip_circulation: bool = False,
    ) -> FitResult:
        """Test a candidate placement against every gate, cheapest first."""
        rejections: list[Rejection] = []

        def record(r: Rejection | None) -> bool:
            """Returns True if we should stop."""
            if r is None:
                return False
            rejections.append(r)
            return not collect_all

        # Cheap integer gates first -- no polygons built yet.
        if record(self._gate_width(item, room, yaw)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_depth(item, room, yaw)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_height(item, room)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))

        placement = Placement(
            sku=item.sku,
            object_class=item.object_class,
            origin=origin,
            footprint=item.footprint,
            yaw=yaw,  # type: ignore[arg-type]
        )

        # Now the polygon gates.
        if record(self._gate_wall(item, room, placement)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_containment(room, placement)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_collision(room, placement)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_door_swing(room, placement)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if record(self._gate_front_clearance(item, room, placement)):
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        if not skip_circulation:
            if record(self._gate_circulation(room, placement)):
                return FitResult(ok=False, placement=None, rejections=tuple(rejections))

        if rejections:
            return FitResult(ok=False, placement=None, rejections=tuple(rejections))
        return FitResult(ok=True, placement=placement, rejections=())

    def first_fit(
        self,
        item: CatalogueItem,
        room: Room,
        candidates: Iterable[tuple[Vec2, int]],
        **kwargs,
    ) -> FitResult:
        """Try candidate (origin, yaw) pairs, returning the first that passes.

        On total failure, returns the rejection from the *last* candidate --
        callers that want the best near-miss should use :meth:`check` directly.
        """
        last: FitResult | None = None
        for origin, yaw in candidates:
            res = self.check(item, room, origin, yaw, **kwargs)
            if res.ok:
                return res
            last = res
        return last or FitResult(ok=False, placement=None, rejections=())
