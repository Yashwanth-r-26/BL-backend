"""Quantity takeoff -- materials derived from scene geometry.

Nobody types "42 m² of tiles" into this system. The scene knows the floor
polygon, so the tile quantity is a calculation, and it updates automatically
when the room does. A typed quantity is a second source of truth that goes
stale the first time someone moves a wall.

The two calculations that carry real domain content:

**Flooring.** Floor area, plus 8% wastage for cuts and breakage, plus adhesive
and grout proportional to the *actual laid* area. Skipping wastage is the
classic underquote -- tiles are cut at every edge and a few always break.

**Paint.** Net wall area = gross (perimeter x ceiling height) *minus openings*.
Quoting paint for the gross area bills the client for painting their windows.
Then putty, primer, and topcoat each have their own coverage rate and coat
count, because they are different materials and share nothing but the surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from ..core.enums import Unit
from ..core.geometry import floor_area_mm2, gross_wall_area_mm2, net_wall_area_mm2
from ..core.scene import Room
from ..core.units import mm2_to_m2

import os

# Basis strings use the proper m² glyph by default. Some legacy clients and
# misconfigured Windows consoles mangle UTF-8 into "mÂ²" -- the data is correct
# but the display is not. BASIS_ASCII=1 renders units as plain "m2" for those
# clients. Default stays UTF-8 because it is correct and real frontends handle it.
_SQM = "m2" if os.getenv("BASIS_ASCII") == "1" else "m²"

TILE_WASTAGE_PCT = 0.08
ADHESIVE_KG_PER_M2 = 4.0
GROUT_KG_PER_M2 = 0.5

PUTTY_COVERAGE_M2_PER_KG = 1.1
PUTTY_COATS = 2
PRIMER_COVERAGE_M2_PER_L = 10.0
PRIMER_COATS = 1
PAINT_COVERAGE_M2_PER_L = 9.0
PAINT_COATS = 2


@dataclass(frozen=True)
class TakeoffLine:
    """One derived material quantity, with the arithmetic that produced it.

    ``basis`` exists so a quantity can be argued with. "18.4 litres" invites
    "says who"; "18.4 litres = 82.7 m² net wall x 2 coats / 9.0 m²/L" does not.
    """

    sku: str
    description: str
    quantity: Decimal
    unit: Unit
    basis: str
    room_id: str | None = None

    def __str__(self) -> str:
        return f"{self.description}: {self.quantity} {self.unit} ({self.basis})"


def _dec(value: float, places: int = 2) -> Decimal:
    q = Decimal(10) ** -places
    return Decimal(str(round(value, places))).quantize(q)


def flooring_takeoff(room: Room, *, tile_sku: str = "TILE-STD") -> list[TakeoffLine]:
    """Tiles, adhesive, and grout from the room's floor polygon."""
    area_m2 = mm2_to_m2(floor_area_mm2(room))
    with_wastage = area_m2 * (1.0 + TILE_WASTAGE_PCT)

    return [
        TakeoffLine(
            sku=tile_sku,
            description="Floor tiles",
            quantity=_dec(with_wastage),
            unit=Unit.SQM,
            basis=(
                f"{_dec(area_m2)} {_SQM} floor area + "
                f"{int(TILE_WASTAGE_PCT * 100)}% cutting wastage"
            ),
            room_id=room.id,
        ),
        TakeoffLine(
            sku="ADHESIVE-STD",
            description="Tile adhesive",
            quantity=_dec(with_wastage * ADHESIVE_KG_PER_M2),
            unit=Unit.KG,
            basis=f"{_dec(with_wastage)} {_SQM} laid x {ADHESIVE_KG_PER_M2} kg/{_SQM}",
            room_id=room.id,
        ),
        TakeoffLine(
            sku="GROUT-STD",
            description="Tile grout",
            quantity=_dec(with_wastage * GROUT_KG_PER_M2),
            unit=Unit.KG,
            basis=f"{_dec(with_wastage)} {_SQM} laid x {GROUT_KG_PER_M2} kg/{_SQM}",
            room_id=room.id,
        ),
    ]


def paint_takeoff(room: Room) -> list[TakeoffLine]:
    """Putty, primer, and topcoat from net wall area.

    Net, not gross -- the openings come out first, or the client is billed for
    painting the windows.
    """
    net_m2 = mm2_to_m2(net_wall_area_mm2(room))
    gross_m2 = mm2_to_m2(gross_wall_area_mm2(room))
    deducted = gross_m2 - net_m2

    basis_suffix = (
        f"net wall {_dec(net_m2)} {_SQM} (gross {_dec(gross_m2)} {_SQM} "
        f"less {_dec(deducted)} {_SQM} of openings)"
    )

    return [
        TakeoffLine(
            sku="PUTTY-STD",
            description="Wall putty",
            quantity=_dec(net_m2 * PUTTY_COATS / PUTTY_COVERAGE_M2_PER_KG),
            unit=Unit.KG,
            basis=(
                f"{basis_suffix} x {PUTTY_COATS} coats / "
                f"{PUTTY_COVERAGE_M2_PER_KG} {_SQM}/kg"
            ),
            room_id=room.id,
        ),
        TakeoffLine(
            sku="PRIMER-STD",
            description="Wall primer",
            quantity=_dec(net_m2 * PRIMER_COATS / PRIMER_COVERAGE_M2_PER_L),
            unit=Unit.LITRE,
            basis=(
                f"{basis_suffix} x {PRIMER_COATS} coat / "
                f"{PRIMER_COVERAGE_M2_PER_L} {_SQM}/L"
            ),
            room_id=room.id,
        ),
        TakeoffLine(
            sku="PAINT-STD",
            description="Emulsion paint",
            quantity=_dec(net_m2 * PAINT_COATS / PAINT_COVERAGE_M2_PER_L),
            unit=Unit.LITRE,
            basis=(
                f"{basis_suffix} x {PAINT_COATS} coats / "
                f"{PAINT_COVERAGE_M2_PER_L} {_SQM}/L"
            ),
            room_id=room.id,
        ),
    ]


def furniture_takeoff(room: Room) -> list[TakeoffLine]:
    """One line per placed object. Obstacles are excluded -- they are existing
    building fabric, not something the client is buying."""
    counts: dict[tuple[str, str], int] = {}
    for p in room.placements:
        key = (p.sku, p.object_class.value)
        counts[key] = counts.get(key, 0) + 1

    return [
        TakeoffLine(
            sku=sku,
            description=cls.replace("_", " ").title(),
            quantity=Decimal(n),
            unit=Unit.PIECE,
            basis=f"{n} placed in {room.name}",
            room_id=room.id,
        )
        for (sku, cls), n in sorted(counts.items())
    ]


def room_takeoff(
    room: Room,
    *,
    include_flooring: bool = True,
    include_paint: bool = True,
    include_furniture: bool = True,
) -> list[TakeoffLine]:
    """Full material take-off for one room."""
    lines: list[TakeoffLine] = []
    if include_flooring:
        lines.extend(flooring_takeoff(room))
    if include_paint:
        lines.extend(paint_takeoff(room))
    if include_furniture:
        lines.extend(furniture_takeoff(room))
    return lines