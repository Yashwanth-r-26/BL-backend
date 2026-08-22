"""Room dimension estimation -- classify with a model, size with priors.

This module produces an *estimated* room polygon when no real measurement
exists yet. It is built to make the estimate impossible to mistake for a fact:

* A vision model does the part it is reliable at -- naming the room type and
  judging a coarse size class (small / average / large). It never emits a
  number.
* The prior table (:mod:`interior_ai.perception.priors`) supplies the actual
  dimensions for that (region, housing, type, size) combination.
* The result carries ``source="estimated_prior"`` and a confidence, and the
  scene it produces is tagged the same way, so every downstream quote can say
  "these figures are estimated, not measured".

When a real measurement arrives later, it overwrites the prior and the source
flag flips to ``measured``. Nothing about this module blocks that upgrade.

The classification prompt is deliberately narrow and, like the construction
prompt, forbids the model from inventing dimensions -- size is a bucket, not a
measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.scene import Room, Vec2
from .priors import (
    HousingType,
    Region,
    RoomDimensions,
    RoomType,
    SizeBucket,
    lookup,
)

DIMENSION_SOURCE_ESTIMATED = "estimated_prior"
DIMENSION_SOURCE_MEASURED = "measured"

CLASSIFY_PROMPT = """You are classifying a single room from a photograph for an
interior-design system. Answer TWO questions only.

1. room_type: which kind of room is this? One of:
   living, kitchen, bedroom, bathroom, dining, balcony, unknown

2. size_class: how large does this room look FOR ITS TYPE? One of:
   small, average, large, unknown

CRITICAL RULES:
- You do NOT measure. Do NOT output dimensions, areas, or distances in any
  unit. "size_class" is a coarse visual impression relative to a typical room
  of that type -- nothing more. Another system converts it to numbers.
- Use "unknown" for either field when the photo does not show enough to judge.
  An honest "unknown" is correct and useful; a confident wrong answer is not.
- Judge size_class relative to the TYPE: a large bathroom is still smaller than
  a small living room. "large" means large for a bathroom, etc.

Respond with ONLY a JSON object with exactly these two keys and no other text:
{"room_type": "...", "size_class": "..."}"""


@dataclass(frozen=True)
class RoomClassification:
    """What the model decided about a room's category and coarse size."""

    room_type: RoomType
    size_bucket: SizeBucket
    confidence: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DimensionEstimate:
    """An estimated room footprint with its full provenance.

    ``source`` is the load-bearing field. As long as it reads
    ``estimated_prior``, every consumer knows these numbers came from a typical
    and not a tape measure.
    """

    dimensions: RoomDimensions
    room_type: RoomType
    size_bucket: SizeBucket
    region: Region
    housing: HousingType
    source: str
    confidence: float
    basis: str
    caveat: str = (
        "Dimensions are ESTIMATED from typical room sizes, not measured. "
        "Quantities and costs derived from them are indicative only and must "
        "be confirmed against a real measurement before ordering."
    )

    @property
    def is_measured(self) -> bool:
        return self.source == DIMENSION_SOURCE_MEASURED


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace:
            text = brace.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("classification response was not a JSON object")
    return parsed


def _coerce_room_type(value: Any) -> tuple[RoomType, bool]:
    """Map a model answer to RoomType. Returns (type, was_recognised)."""
    if not isinstance(value, str):
        return RoomType.UNKNOWN, False
    v = value.strip().lower()
    try:
        return RoomType(v), True
    except ValueError:
        return RoomType.UNKNOWN, False


def _coerce_size_bucket(value: Any) -> tuple[SizeBucket, bool]:
    if not isinstance(value, str):
        return SizeBucket.AVERAGE, False
    v = value.strip().lower()
    if v == "unknown":
        # Unknown size defaults to AVERAGE -- the safest middle estimate -- but
        # is reported as unrecognised so confidence drops.
        return SizeBucket.AVERAGE, False
    try:
        return SizeBucket(v), True
    except ValueError:
        return SizeBucket.AVERAGE, False


def parse_classification(text: str) -> RoomClassification:
    """Parse a model reply into a room classification.

    Anything unparseable degrades to (unknown type, average size, low
    confidence) rather than raising -- an estimate built on a failed
    classification is still honestly an estimate, just a less confident one.
    """
    notes: list[str] = []
    try:
        raw = _extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return RoomClassification(
            room_type=RoomType.UNKNOWN,
            size_bucket=SizeBucket.AVERAGE,
            confidence=0.2,
            notes=(f"could not parse classification ({exc}); defaulting to unknown/average",),
        )

    room_type, type_ok = _coerce_room_type(raw.get("room_type"))
    size_bucket, size_ok = _coerce_size_bucket(raw.get("size_class"))

    if not type_ok:
        notes.append(f"room_type {raw.get('room_type')!r} not recognised; treated as unknown")
    if not size_ok:
        notes.append(f"size_class {raw.get('size_class')!r} not recognised; defaulted to average")

    # Confidence: both recognised -> high-ish; each miss knocks it down.
    confidence = 0.8
    if not type_ok:
        confidence -= 0.4
    if not size_ok:
        confidence -= 0.2

    return RoomClassification(
        room_type=room_type,
        size_bucket=size_bucket,
        confidence=max(0.1, confidence),
        notes=tuple(notes),
    )


def estimate_dimensions(
    classification: RoomClassification,
    *,
    region: Region = Region.GENERIC,
    housing: HousingType = HousingType.UNKNOWN,
) -> DimensionEstimate:
    """Combine a classification with priors into an estimated footprint.

    The final confidence multiplies the classification confidence by the
    prior's own confidence -- an uncertain room type applied to a generic
    fallback should not produce a falsely-confident number.
    """
    dims, prior_conf, basis = lookup(
        region=region,
        housing=housing,
        room=classification.room_type,
        bucket=classification.size_bucket,
    )
    combined_conf = round(classification.confidence * prior_conf, 3)

    return DimensionEstimate(
        dimensions=dims,
        room_type=classification.room_type,
        size_bucket=classification.size_bucket,
        region=region,
        housing=housing,
        source=DIMENSION_SOURCE_ESTIMATED,
        confidence=combined_conf,
        basis=(
            f"{basis}; classified as {classification.room_type.value}/"
            f"{classification.size_bucket.value} (class conf "
            f"{classification.confidence}, prior conf {prior_conf})"
        ),
    )


def estimate_to_polygon(estimate: DimensionEstimate) -> tuple[Vec2, ...]:
    """Rectangular polygon from an estimate, origin at (0,0)."""
    w, d = estimate.dimensions.width_mm, estimate.dimensions.depth_mm
    return (Vec2(x=0, y=0), Vec2(x=w, y=0), Vec2(x=w, y=d), Vec2(x=0, y=d))


def build_estimated_room(
    estimate: DimensionEstimate,
    *,
    name: str | None = None,
    include_openings: bool = True,
) -> Room:
    """Construct a scene Room from an estimate.

    The estimate's provenance is stamped into the room name so it is visible
    even in a bare scene dump; structured provenance travels alongside in the
    API/orchestrator layer.

    Typical openings (a door, usually a window) are added by default. Without
    them the room is a sealed box and the paint take-off quotes solid walls --
    billing for painting over the door and window. Like the dimensions, the
    openings are typical values, replaced when a real scan supplies actual ones.
    """
    from ..core.enums import OpeningKind, SwingDirection
    from ..core.scene import Opening
    from .priors import typical_openings

    label = name or f"{estimate.room_type.value.title()} (estimated)"
    polygon = estimate_to_polygon(estimate)
    w = estimate.dimensions.width_mm
    d = estimate.dimensions.depth_mm

    openings: list = []
    if include_openings:
        specs = typical_openings(estimate.room_type)
        for i, spec in enumerate(specs):
            if spec.kind == "door":
                # Door on the front wall (wall 0, y=0), offset from the corner
                # so its swing has room. Clamp so a narrow room still fits it.
                cx = min(max(spec.width_mm, w // 4), w - spec.width_mm)
                openings.append(
                    Opening(
                        kind=OpeningKind.DOOR,
                        centre=Vec2(x=cx, y=0),
                        width_mm=spec.width_mm,
                        height_mm=spec.height_mm,
                        wall_index=0,
                        swing=SwingDirection.INWARD,
                    )
                )
            else:
                # Window on the opposite wall (wall 2, y=d), centred.
                openings.append(
                    Opening(
                        kind=OpeningKind.WINDOW,
                        centre=Vec2(x=w // 2, y=d),
                        width_mm=min(spec.width_mm, w - 200),
                        height_mm=spec.height_mm,
                        sill_height_mm=900,
                        wall_index=2,
                    )
                )

    return Room(
        name=label,
        polygon=polygon,
        ceiling_height_mm=estimate.dimensions.ceiling_mm,
        openings=tuple(openings),
    )