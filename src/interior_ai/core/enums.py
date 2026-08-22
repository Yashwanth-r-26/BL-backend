"""Shared enumerations.

Every enum here is persisted to Postgres as a string, so values are stable
identifiers -- renaming one is a migration, not a refactor.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that serialises to its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ExecutionPath(StrEnum):
    """Where inference actually runs for a given request."""

    LOCAL_FULL = "LOCAL_FULL"
    LOCAL_LIGHT = "LOCAL_LIGHT"
    CLOUD_API = "CLOUD_API"
    MOCK = "MOCK"


class Tri(StrEnum):
    """Tri-valued (plus unknown) perception signal.

    Perception is allowed to be uncertain. PARTIAL and UNKNOWN are first-class
    answers, not error states -- a half-painted room is the common case, and
    the phase rules treat it as blocking rather than guessing.
    """

    YES = "yes"
    NO = "no"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Phase(StrEnum):
    """Renovation phase a room is currently in."""

    SURFACE_FINISHING = "SURFACE_FINISHING"
    FIXTURES_CARPENTRY = "FIXTURES_CARPENTRY"
    STYLING_RESTRUCTURE = "STYLING_RESTRUCTURE"


class ObjectClass(StrEnum):
    """Catalogue object classes the fit engine and solver understand."""

    SOFA = "sofa"
    COFFEE_TABLE = "coffee_table"
    TV_UNIT = "tv_unit"
    ARMCHAIR = "armchair"
    BED = "bed"
    WARDROBE = "wardrobe"
    SIDE_TABLE = "side_table"
    DINING_TABLE = "dining_table"
    BOOKSHELF = "bookshelf"
    RUG = "rug"


class OpeningKind(StrEnum):
    """Wall openings. Doors carry swing arcs; windows only remove wall area."""

    DOOR = "door"
    WINDOW = "window"


class SwingDirection(StrEnum):
    INWARD = "inward"
    OUTWARD = "outward"


class RejectionCode(StrEnum):
    """Why a placement failed.

    Ordered cheapest-first in the fit engine; each code always arrives with a
    measured overage so the caller can say "too wide by 800 mm".
    """

    TOO_WIDE = "TOO_WIDE"
    TOO_DEEP = "TOO_DEEP"
    TOO_TALL = "TOO_TALL"
    WALL_REQUIRED = "WALL_REQUIRED"
    OUT_OF_ROOM = "OUT_OF_ROOM"
    COLLISION = "COLLISION"
    DOOR_SWING_BLOCKED = "DOOR_SWING_BLOCKED"
    FRONT_CLEARANCE = "FRONT_CLEARANCE"
    CIRCULATION_BLOCKED = "CIRCULATION_BLOCKED"


class PriceStatus(StrEnum):
    """Freshness of a price at snapshot time."""

    FRESH = "fresh"
    STALE = "stale"
    UNPRICED = "unpriced"


class Unit(StrEnum):
    """Billing units for BOQ lines."""

    SQM = "sqm"
    LITRE = "litre"
    KG = "kg"
    PIECE = "piece"
    METRE = "metre"
    BAG = "bag"
