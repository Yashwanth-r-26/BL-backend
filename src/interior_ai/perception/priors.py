"""Typical-dimension priors -- estimated geometry, honestly labelled.

The problem this solves: the pipeline needs a room polygon, but real
measurement (LiDAR / vendor survey) is not wired in yet. The wrong fix is to
have a vision model guess dimensions from a photo -- a single image has no
scale, so that produces confident numbers that are frequently off by 2-3x, and
once a hallucinated "4.2 m x 3.8 m" sits in the scene graph nothing downstream
can tell it from a measurement.

The right fix is a lookup of *typical* dimensions by (region, housing type,
room type, size bucket). These are researchable, defensible figures -- Indian
builder layouts and RERA carpet-area norms -- not per-photo guesses. A vision
model chooses the *category and coarse size class* (something it can do
reliably); this table supplies the *numbers*; and every result is tagged
``estimated_prior`` so a guess can never masquerade as a fact.

Every dimension here is a documented typical, and every polygon this produces
carries a source flag and a confidence. When a real measurement arrives it
overwrites the prior and the flag flips to ``measured``.

Sources for the seed values: standard Indian residential room dimensions as
published in builder floor plans and carpet-area conventions (living rooms
typically 11-18 m^2 in 2-3BHK flats; kitchens 5-9 m^2; ceiling heights 2.9-3.1 m
which is the common Indian slab-to-slab finish, taller than the 2.4 m Western
default). These are starting defaults, meant to be tuned against real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..core.enums import StrEnum


class Region(StrEnum):
    """Coarse geographic context. Drives which typical-size table applies.

    Kept coarse on purpose -- the point is to capture that an Indian metro flat
    and a US suburban house have systematically different room sizes and
    ceiling heights, not to model every city.
    """

    IN_METRO = "IN_METRO"        # Indian metro flats (Bangalore, Mumbai, ...)
    IN_NONMETRO = "IN_NONMETRO"  # Indian smaller-city / independent houses
    GENERIC = "GENERIC"          # fallback when region is unknown


class HousingType(StrEnum):
    """Housing context within a region."""

    FLAT_1BHK = "FLAT_1BHK"
    FLAT_2BHK = "FLAT_2BHK"
    FLAT_3BHK = "FLAT_3BHK"
    INDEPENDENT = "INDEPENDENT"   # independent house / villa
    UNKNOWN = "UNKNOWN"


class RoomType(StrEnum):
    """Room category. This is what the vision model classifies."""

    LIVING = "living"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    DINING = "dining"
    BALCONY = "balcony"
    UNKNOWN = "unknown"


class SizeBucket(StrEnum):
    """Coarse size class within a room type.

    A vision model cannot say '4.2 metres' from a photo, but it can fairly
    reliably say 'this looks small / average / large for a living room'. That
    coarse judgement is what selects between the three columns below.
    """

    SMALL = "small"
    AVERAGE = "average"
    LARGE = "large"


@dataclass(frozen=True)
class RoomDimensions:
    """A typical room footprint. Integer millimetres, matching the scene graph.

    Rectangular here -- typical-room data is only ever quoted as width x depth,
    and an estimate pretending to know an L-shape would be false precision. The
    scene graph accepts any polygon, so a real scan can be any shape; the prior
    is honestly a box.
    """

    width_mm: int
    depth_mm: int
    ceiling_mm: int

    @property
    def area_m2(self) -> float:
        return (self.width_mm * self.depth_mm) / 1_000_000


# Default ceiling heights by region. Indian slab-to-slab finish runs taller
# than the Western 2.4 m default -- getting this wrong throws off every paint
# quantity, since wall area is perimeter x height.
_CEILING_MM = {
    Region.IN_METRO: 3000,
    Region.IN_NONMETRO: 3050,
    Region.GENERIC: 2700,
}


# Typical width x depth in mm, by (region, room type, size bucket).
# Only the region/room combinations with meaningfully different norms are
# enumerated; everything else resolves through _fallback_dimensions.
_TYPICAL: dict[tuple[Region, RoomType], dict[SizeBucket, tuple[int, int]]] = {
    (Region.IN_METRO, RoomType.LIVING): {
        SizeBucket.SMALL: (3300, 3600),
        SizeBucket.AVERAGE: (3700, 4300),
        SizeBucket.LARGE: (4300, 5500),
    },
    (Region.IN_METRO, RoomType.KITCHEN): {
        SizeBucket.SMALL: (2100, 2400),
        SizeBucket.AVERAGE: (2400, 3000),
        SizeBucket.LARGE: (3000, 3600),
    },
    (Region.IN_METRO, RoomType.BEDROOM): {
        SizeBucket.SMALL: (3000, 3000),
        SizeBucket.AVERAGE: (3300, 3700),
        SizeBucket.LARGE: (3700, 4600),
    },
    (Region.IN_METRO, RoomType.BATHROOM): {
        SizeBucket.SMALL: (1200, 1800),
        SizeBucket.AVERAGE: (1500, 2100),
        SizeBucket.LARGE: (2100, 2700),
    },
    (Region.IN_METRO, RoomType.DINING): {
        SizeBucket.SMALL: (2700, 3000),
        SizeBucket.AVERAGE: (3000, 3600),
        SizeBucket.LARGE: (3600, 4300),
    },
    (Region.IN_METRO, RoomType.BALCONY): {
        SizeBucket.SMALL: (1200, 2400),
        SizeBucket.AVERAGE: (1500, 3000),
        SizeBucket.LARGE: (1800, 4000),
    },
}

# Independent Indian houses run larger than metro flats across the board.
_INDEPENDENT_SCALE = 1.15

# Generic fallback footprints when region/room is unknown -- deliberately
# middle-of-the-road and flagged with the lowest confidence downstream.
_GENERIC_FALLBACK: dict[SizeBucket, tuple[int, int]] = {
    SizeBucket.SMALL: (3000, 3000),
    SizeBucket.AVERAGE: (3600, 4000),
    SizeBucket.LARGE: (4300, 5000),
}


def _fallback_dimensions(room: RoomType, bucket: SizeBucket) -> tuple[int, int]:
    return _GENERIC_FALLBACK[bucket]


def lookup(
    *,
    region: Region,
    housing: HousingType,
    room: RoomType,
    bucket: SizeBucket,
) -> tuple[RoomDimensions, float, str]:
    """Resolve a typical footprint.

    Returns (dimensions, confidence, basis). Confidence reflects how specific
    the match was: an exact region+room hit is more trustworthy than a generic
    fallback, though *all* of these are estimates and none should be mistaken
    for a measurement.
    """
    # Resolve region for non-metro / independent housing.
    effective_region = region
    if region is Region.IN_NONMETRO:
        # Non-metro uses metro tables as the closest available proxy, scaled.
        effective_region = Region.IN_METRO

    table = _TYPICAL.get((effective_region, room))
    if table is not None and bucket in table:
        w, d = table[bucket]
        confidence = 0.55
        basis = f"typical {room.value} for {region.value}, {bucket.value} size"
    else:
        w, d = _fallback_dimensions(room, bucket)
        confidence = 0.30
        basis = f"generic {bucket.value} fallback ({room.value} not tabulated for {region.value})"

    # Independent houses and non-metro run larger.
    if housing is HousingType.INDEPENDENT or region is Region.IN_NONMETRO:
        w = int(round(w * _INDEPENDENT_SCALE))
        d = int(round(d * _INDEPENDENT_SCALE))
        basis += ", scaled up for independent/non-metro housing"

    ceiling = _CEILING_MM.get(region, _CEILING_MM[Region.GENERIC])

    return RoomDimensions(width_mm=w, depth_mm=d, ceiling_mm=ceiling), confidence, basis


@dataclass(frozen=True)
class OpeningSpec:
    """A typical opening for an estimated room.

    Estimated rooms otherwise come out as sealed boxes, which makes the paint
    take-off quote solid walls -- billing the client for painting over the
    door and window. A typical room has at least a door and usually a window,
    so the estimate includes them. Like the dimensions, these are typical
    values, replaced the moment a real scan supplies actual openings.

    ``kind`` is "door" or "window"; positions are chosen at build time to sit
    on sensible walls. Sizes here are standard Indian residential defaults
    (900x2100 single door, 1200x1500 window).
    """

    kind: str
    width_mm: int
    height_mm: int


# Typical openings by room type. Bathrooms get a narrower door and a small
# high window; living/dining get a main door plus a large window; balconies are
# mostly opening and are left with just an access door.
_TYPICAL_OPENINGS: dict[RoomType, tuple[OpeningSpec, ...]] = {
    RoomType.LIVING: (
        OpeningSpec("door", 900, 2100),
        OpeningSpec("window", 1500, 1200),
    ),
    RoomType.DINING: (
        OpeningSpec("door", 900, 2100),
        OpeningSpec("window", 1200, 1200),
    ),
    RoomType.KITCHEN: (
        OpeningSpec("door", 800, 2100),
        OpeningSpec("window", 1200, 900),
    ),
    RoomType.BEDROOM: (
        OpeningSpec("door", 900, 2100),
        OpeningSpec("window", 1200, 1200),
    ),
    RoomType.BATHROOM: (
        OpeningSpec("door", 750, 2100),
        OpeningSpec("window", 600, 600),
    ),
    RoomType.BALCONY: (
        OpeningSpec("door", 1500, 2100),
    ),
    RoomType.UNKNOWN: (
        OpeningSpec("door", 900, 2100),
    ),
}


def typical_openings(room: RoomType) -> tuple[OpeningSpec, ...]:
    """Typical openings for a room type. Defaults to a single door."""
    return _TYPICAL_OPENINGS.get(room, (OpeningSpec("door", 900, 2100),))