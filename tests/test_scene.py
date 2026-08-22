"""Scene graph, unit discipline, and geometry."""

from __future__ import annotations

import math

import pytest

from interior_ai.core.enums import ObjectClass, OpeningKind, SwingDirection
from interior_ai.core.geometry import (
    door_swing_polygon,
    floor_area_mm2,
    gross_wall_area_mm2,
    net_wall_area_mm2,
    room_polygon,
)
from interior_ai.core.scene import (
    Footprint,
    Opening,
    Placement,
    Room,
    Scene,
    Vec2,
)
from interior_ai.core.units import (
    apply_tolerance,
    m_to_mm,
    mm2_to_m2,
    mm_to_m,
    snap_down,
    snap_to_grid,
)


class TestUnits:
    def test_metre_round_trip_is_exact(self):
        for metres in (0.0, 1.0, 2.4, 3.075, 12.5):
            assert mm_to_m(m_to_mm(metres)) == pytest.approx(metres)

    def test_snapping(self):
        assert snap_to_grid(1234, 50) == 1250
        assert snap_to_grid(1224, 50) == 1200
        assert snap_down(1249, 50) == 1200
        assert snap_down(1250, 50) == 1250

    def test_snap_rejects_bad_grid(self):
        with pytest.raises(ValueError):
            snap_to_grid(100, 0)

    def test_tolerance_expands(self):
        assert apply_tolerance(1000, 0.08) == 1080
        assert apply_tolerance(1000, 0.0) == 1000
        with pytest.raises(ValueError):
            apply_tolerance(1000, -0.1)

    def test_area_conversion(self):
        assert mm2_to_m2(1_000_000) == pytest.approx(1.0)


class TestSceneVersioning:
    def test_next_version_does_not_mutate(self, scene: Scene):
        original_version = scene.version
        original_id = scene.version_id
        successor = scene.next_version(notes="changed")

        assert scene.version == original_version
        assert scene.version_id == original_id
        assert successor.version == original_version + 1
        assert successor.parent_version_id == original_id

    def test_successor_keeps_scene_identity(self, scene: Scene):
        successor = scene.next_version()
        assert successor.id == scene.id
        assert successor.version_id != scene.version_id

    def test_version_chain_links_backwards(self, scene: Scene):
        v2 = scene.next_version()
        v3 = v2.next_version()
        assert v3.parent_version_id == v2.version_id
        assert v2.parent_version_id == scene.version_id
        assert scene.parent_version_id is None

    def test_replace_room_produces_successor(self, scene: Scene):
        room = scene.rooms[0]
        renamed = room.model_copy(update={"name": "Renamed"})
        successor = scene.replace_room(renamed)

        assert successor.version == scene.version + 1
        assert successor.room(room.id).name == "Renamed"
        # Original untouched -- this is what makes old quotes reproducible.
        assert scene.room(room.id).name == room.name

    def test_replace_unknown_room_raises(self, scene: Scene, bare_room: Room):
        stranger = Room(
            name="Stranger",
            polygon=bare_room.polygon,
            ceiling_height_mm=2400,
        )
        with pytest.raises(KeyError):
            scene.replace_room(stranger)

    def test_scene_is_frozen(self, scene: Scene):
        with pytest.raises(Exception):
            scene.version = 99  # type: ignore[misc]


class TestPlacementGeometry:
    def test_rotation_swaps_extent(self):
        fp = Footprint(width_mm=2200, depth_mm=900, height_mm=800)
        assert fp.rotated(0) == (2200, 900)
        assert fp.rotated(180) == (2200, 900)
        assert fp.rotated(90) == (900, 2200)
        assert fp.rotated(270) == (900, 2200)

    def test_bounds_follow_rotation(self):
        fp = Footprint(width_mm=2000, depth_mm=800, height_mm=800)
        p = Placement(
            sku="X", object_class=ObjectClass.SOFA,
            origin=Vec2(x=100, y=200), footprint=fp, yaw=90,
        )
        assert p.bounds == (100, 200, 100 + 800, 200 + 2000)

    def test_centre_is_midpoint(self):
        fp = Footprint(width_mm=1000, depth_mm=500, height_mm=400)
        p = Placement(
            sku="X", object_class=ObjectClass.COFFEE_TABLE,
            origin=Vec2(x=0, y=0), footprint=fp,
        )
        assert (p.centre.x, p.centre.y) == (500, 250)

    def test_facing_vectors_are_distinct(self):
        fp = Footprint(width_mm=100, depth_mm=100, height_mm=100)
        seen = {
            Placement(
                sku="X", object_class=ObjectClass.SOFA,
                origin=Vec2(x=0, y=0), footprint=fp, yaw=yaw,
            ).facing_vector()
            for yaw in (0, 90, 180, 270)
        }
        assert len(seen) == 4


class TestOpenings:
    def test_door_requires_swing(self):
        with pytest.raises(ValueError):
            Opening(
                kind=OpeningKind.DOOR, centre=Vec2(x=0, y=0),
                width_mm=900, height_mm=2100, wall_index=0,
            )

    def test_door_swing_radius_defaults_to_width(self):
        d = Opening(
            kind=OpeningKind.DOOR, centre=Vec2(x=500, y=0),
            width_mm=900, height_mm=2100, wall_index=0,
            swing=SwingDirection.INWARD,
        )
        assert d.swing_radius_mm == 900

    def test_window_needs_no_swing(self):
        w = Opening(
            kind=OpeningKind.WINDOW, centre=Vec2(x=500, y=0),
            width_mm=1200, height_mm=1500, wall_index=0,
        )
        assert w.swing is None


class TestDoorSwingGeometry:
    def test_swing_approximates_quarter_circle(self, door_room: Room):
        swing = door_swing_polygon(door_room, door_room.openings[0])
        expected = math.pi * 900**2 / 4
        # Polygon approximation of an arc; within 2% is the discretisation error.
        assert swing.area == pytest.approx(expected, rel=0.02)

    def test_swing_lies_inside_the_room(self, door_room: Room):
        swing = door_swing_polygon(door_room, door_room.openings[0])
        assert room_polygon(door_room).contains(swing.buffer(-1))

    def test_outward_door_blocks_no_floor(self, bare_room: Room):
        room = bare_room.model_copy(
            update={
                "openings": (
                    Opening(
                        kind=OpeningKind.DOOR, centre=Vec2(x=1000, y=0),
                        width_mm=900, height_mm=2100, wall_index=0,
                        swing=SwingDirection.OUTWARD,
                    ),
                )
            }
        )
        assert door_swing_polygon(room, room.openings[0]) is None

    def test_window_has_no_swing(self, windowed_room: Room):
        window = next(o for o in windowed_room.openings if o.kind is OpeningKind.WINDOW)
        assert door_swing_polygon(windowed_room, window) is None


class TestAreaCalculations:
    def test_floor_area(self, windowed_room: Room):
        assert floor_area_mm2(windowed_room) == 4000 * 3000

    def test_gross_wall_area_is_perimeter_times_height(self, windowed_room: Room):
        assert gross_wall_area_mm2(windowed_room) == 14000 * 2700

    def test_net_wall_area_subtracts_openings(self, windowed_room: Room):
        gross = gross_wall_area_mm2(windowed_room)
        door = 900 * 2100
        window = 1500 * 1200
        assert net_wall_area_mm2(windowed_room) == gross - door - window

    def test_net_wall_area_never_negative(self):
        """A room whose openings exceed its walls is bad data, but a negative
        area would silently produce a negative paint quantity downstream."""
        room = Room(
            name="Absurd",
            polygon=(Vec2(x=0, y=0), Vec2(x=1000, y=0), Vec2(x=1000, y=1000), Vec2(x=0, y=1000)),
            ceiling_height_mm=100,
            openings=(
                Opening(
                    kind=OpeningKind.WINDOW, centre=Vec2(x=500, y=0),
                    width_mm=5000, height_mm=5000, wall_index=0,
                ),
            ),
        )
        assert net_wall_area_mm2(room) == 0
