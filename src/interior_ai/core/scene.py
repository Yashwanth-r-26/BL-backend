"""The scene graph -- single source of truth.

Everything else in this service is derived from a :class:`Scene`:

* a render is a *view* of the scene,
* a price is a *calculation* over the scene,
* "that sofa doesn't fit" is a *fact* proved against the scene.

Nothing downstream is allowed to hold state the scene does not have, because
the moment a renderer or a quote remembers something the scene forgot, the two
disagree and there is no way to tell which is right.

Versioning is immutable. :meth:`Scene.next_version` returns a *successor*
carrying ``parent_version_id``; it never mutates in place. That gives an audit
chain -- a quote from March can be re-derived from the exact scene it was
priced against, even after the room has been redesigned five times since.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import ObjectClass, OpeningKind, Phase, SwingDirection

Yaw = Literal[0, 90, 180, 270]


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    """Base for immutable value objects."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Vec2(Frozen):
    """A point in room coordinates, integer millimetres."""

    x: int
    y: int


class Footprint(Frozen):
    """Axis-aligned extent of an object in its own local frame.

    ``width`` runs along local +x, ``depth`` along local +y, ``height`` is up.
    Rotation is applied by the placement, not baked in here.
    """

    width_mm: int = Field(gt=0)
    depth_mm: int = Field(gt=0)
    height_mm: int = Field(gt=0)

    def rotated(self, yaw: Yaw) -> tuple[int, int]:
        """Return (effective_width, effective_depth) after a yaw rotation.

        Only 0/90/180/270 are permitted, so this is a swap rather than a
        trigonometric transform -- which is exactly why the solver is
        restricted to those four angles.
        """
        if yaw in (90, 270):
            return self.depth_mm, self.width_mm
        return self.width_mm, self.depth_mm


class Opening(Frozen):
    """A door or window in a wall.

    Position is the opening's centre, in room coordinates. Doors additionally
    carry a swing arc that the fit engine and solver treat as unusable floor.
    """

    id: str = Field(default_factory=_uid)
    kind: OpeningKind
    centre: Vec2
    width_mm: int = Field(gt=0)
    height_mm: int = Field(gt=0)
    sill_height_mm: int = Field(default=0, ge=0)
    wall_index: int = Field(ge=0)
    swing: SwingDirection | None = None
    swing_radius_mm: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _doors_swing(self) -> "Opening":
        if self.kind is OpeningKind.DOOR:
            if self.swing is None:
                raise ValueError("door openings require a swing direction")
            if self.swing_radius_mm is None:
                object.__setattr__(self, "swing_radius_mm", self.width_mm)
        return self


class CatalogueItem(Frozen):
    """A purchasable object, independent of any placement.

    ``clearance_front_mm`` is the walking/using space the object needs in front
    of it -- the space you stand in to open a wardrobe, or the legroom in front
    of a sofa. It is a property of the product, so it lives here rather than
    in the placement.
    """

    sku: str
    name: str
    object_class: ObjectClass
    footprint: Footprint
    requires_wall: bool = False
    clearance_front_mm: int = Field(default=0, ge=0)
    vendor: str | None = None

    @property
    def anchor_axis_is_width(self) -> bool:
        """Whether the object's 'front' faces along its depth axis."""
        return True


class Placement(Frozen):
    """A catalogue item positioned in a room.

    ``origin`` is the min-corner of the object's axis-aligned box *after*
    rotation, which keeps every consumer (Shapely, CP-SAT, the renderer) using
    the same convention. Storing a centre instead invites off-by-half-a-width
    bugs at every boundary.
    """

    id: str = Field(default_factory=_uid)
    sku: str
    object_class: ObjectClass
    origin: Vec2
    footprint: Footprint
    yaw: Yaw = 0
    fixed: bool = False

    @property
    def effective_size(self) -> tuple[int, int]:
        return self.footprint.rotated(self.yaw)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """(minx, miny, maxx, maxy) in room coordinates."""
        w, d = self.effective_size
        return (self.origin.x, self.origin.y, self.origin.x + w, self.origin.y + d)

    @property
    def centre(self) -> Vec2:
        minx, miny, maxx, maxy = self.bounds
        return Vec2(x=(minx + maxx) // 2, y=(miny + maxy) // 2)

    def facing_vector(self) -> tuple[int, int]:
        """Unit-ish direction the object's front faces, given its yaw."""
        return {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}[self.yaw]


class Obstacle(Frozen):
    """Immovable geometry -- a column, a duct, a radiator.

    Distinct from a Placement because the solver may never move it and the
    takeoff must not price it.
    """

    id: str = Field(default_factory=_uid)
    label: str
    origin: Vec2
    width_mm: int = Field(gt=0)
    depth_mm: int = Field(gt=0)
    height_mm: int | None = Field(default=None, gt=0)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (
            self.origin.x,
            self.origin.y,
            self.origin.x + self.width_mm,
            self.origin.y + self.depth_mm,
        )


class SurfaceState(Frozen):
    """What perception believes about the room's surfaces.

    Every field is Tri-valued. The phase rules consume this; nothing else
    should, because interpreting raw signals in two places is how two parts of
    a system end up disagreeing about what phase a room is in.
    """

    walls_painted: str = "unknown"
    flooring_installed: str = "unknown"
    ceiling_finished: str = "unknown"
    electrical_terminated: str = "unknown"
    plumbing_terminated: str = "unknown"
    carpentry_installed: str = "unknown"
    furniture_present: str = "unknown"

    @field_validator("*", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if hasattr(v, "value"):
            return v.value
        return v


class Room(Frozen):
    """A single room. Polygon in room coordinates, integer millimetres.

    The polygon is stored rather than a width/height pair because L-shaped and
    bay-windowed rooms are common enough that assuming rectangles would force a
    rewrite later. Rectangular rooms are just four-vertex polygons.
    """

    id: str = Field(default_factory=_uid)
    name: str
    polygon: tuple[Vec2, ...]
    ceiling_height_mm: int = Field(gt=0)
    openings: tuple[Opening, ...] = ()
    obstacles: tuple[Obstacle, ...] = ()
    placements: tuple[Placement, ...] = ()
    surfaces: SurfaceState = SurfaceState()
    phase: Phase | None = None

    @field_validator("polygon")
    @classmethod
    def _min_vertices(cls, v: tuple[Vec2, ...]) -> tuple[Vec2, ...]:
        if len(v) < 3:
            raise ValueError("room polygon needs at least 3 vertices")
        return v

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p.x for p in self.polygon]
        ys = [p.y for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def with_placements(self, placements: Sequence[Placement]) -> "Room":
        return self.model_copy(update={"placements": tuple(placements)})


class Scene(Frozen):
    """Root of the scene graph, and the unit of versioning.

    A Scene is never edited. Every change produces a successor via
    :meth:`next_version`, linked by ``parent_version_id``. The chain is what
    makes a six-month-old quote reproducible: you re-read the scene version the
    quote names, and it is byte-for-byte what was priced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_uid)
    version_id: str = Field(default_factory=_uid)
    parent_version_id: str | None = None
    version: int = 1
    project_id: str | None = None
    rooms: tuple[Room, ...] = ()
    created_at: datetime = Field(default_factory=_now)
    notes: str | None = None

    def next_version(
        self,
        *,
        rooms: Sequence[Room] | None = None,
        notes: str | None = None,
    ) -> "Scene":
        """Return a successor scene. Never mutates ``self``.

        The successor keeps the same ``id`` (it is the same scene) but gets a
        fresh ``version_id`` and points back at this one.
        """
        return Scene(
            id=self.id,
            version_id=_uid(),
            parent_version_id=self.version_id,
            version=self.version + 1,
            project_id=self.project_id,
            rooms=tuple(rooms) if rooms is not None else self.rooms,
            notes=notes,
        )

    def room(self, room_id: str) -> Room:
        for r in self.rooms:
            if r.id == room_id:
                return r
        raise KeyError(f"no room {room_id!r} in scene {self.id!r}")

    def replace_room(self, room: Room, *, notes: str | None = None) -> "Scene":
        """Successor scene with one room swapped out."""
        found = False
        new_rooms = []
        for r in self.rooms:
            if r.id == room.id:
                new_rooms.append(room)
                found = True
            else:
                new_rooms.append(r)
        if not found:
            raise KeyError(f"no room {room.id!r} in scene {self.id!r}")
        return self.next_version(rooms=new_rooms, notes=notes)

    def lineage(self) -> list[str]:
        """Version ids from this scene back toward the root.

        Only returns what this object knows -- the full chain lives in the
        repository. Useful for assertions and debugging.
        """
        chain = [self.version_id]
        if self.parent_version_id:
            chain.append(self.parent_version_id)
        return chain
