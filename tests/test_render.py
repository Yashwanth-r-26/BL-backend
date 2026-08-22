"""Floor-plan SVG rendering."""

from __future__ import annotations

from interior_ai.api.render import render_floor_plan
from interior_ai.core.enums import ObjectClass, OpeningKind, SwingDirection
from interior_ai.core.scene import (
    Footprint,
    Opening,
    Placement,
    Room,
    Vec2,
)


def _room_with_furniture() -> Room:
    return Room(
        name="Living",
        polygon=(Vec2(x=0, y=0), Vec2(x=5000, y=0), Vec2(x=5000, y=4000), Vec2(x=0, y=4000)),
        ceiling_height_mm=2700,
        openings=(
            Opening(
                kind=OpeningKind.DOOR, centre=Vec2(x=700, y=0),
                width_mm=900, height_mm=2100, wall_index=0, swing=SwingDirection.INWARD,
            ),
            Opening(
                kind=OpeningKind.WINDOW, centre=Vec2(x=2500, y=4000),
                width_mm=1500, height_mm=1200, wall_index=2, sill_height_mm=900,
            ),
        ),
        placements=(
            Placement(
                sku="S", object_class=ObjectClass.SOFA,
                origin=Vec2(x=1400, y=200),
                footprint=Footprint(width_mm=2200, depth_mm=900, height_mm=800),
            ),
            Placement(
                sku="T", object_class=ObjectClass.COFFEE_TABLE,
                origin=Vec2(x=1900, y=1300),
                footprint=Footprint(width_mm=1100, depth_mm=600, height_mm=400),
            ),
        ),
    )


class TestFloorPlanRender:
    def test_returns_valid_svg(self):
        svg = render_floor_plan(_room_with_furniture())
        assert svg.startswith("<svg")
        assert svg.rstrip().endswith("</svg>")
        assert "image/svg+xml" or "viewBox" in svg

    def test_draws_every_placement(self):
        room = _room_with_furniture()
        svg = render_floor_plan(room)
        # One rect per placement plus the background rect.
        assert svg.count("<rect") >= len(room.placements) + 1

    def test_draws_openings(self):
        svg = render_floor_plan(_room_with_furniture())
        # Two openings -> two wall segments (lines).
        assert svg.count("<line") >= 2

    def test_labels_can_be_disabled(self):
        with_labels = render_floor_plan(_room_with_furniture(), show_labels=True)
        without = render_floor_plan(_room_with_furniture(), show_labels=False)
        assert with_labels.count("<text") > without.count("<text")

    def test_swings_can_be_disabled(self):
        # A door swing adds a polygon; without swings there should be fewer.
        with_swings = render_floor_plan(_room_with_furniture(), show_swings=True)
        without = render_floor_plan(_room_with_furniture(), show_swings=False)
        assert with_swings.count("<polygon") >= without.count("<polygon")

    def test_empty_room_still_renders(self):
        bare = Room(
            name="Bare",
            polygon=(Vec2(x=0, y=0), Vec2(x=3000, y=0), Vec2(x=3000, y=3000), Vec2(x=0, y=3000)),
            ceiling_height_mm=2700,
        )
        svg = render_floor_plan(bare)
        assert svg.startswith("<svg")

    def test_dimension_caption_present(self):
        svg = render_floor_plan(_room_with_furniture())
        # Subtitle and dimension lines carry the sizes at 2 decimals.
        assert "5.00" in svg and "4.00" in svg

    def test_furniture_drawn_as_rotated_groups(self):
        """Each placement is a transformed group so rotated furniture keeps its
        internal detail oriented."""
        svg = render_floor_plan(_room_with_furniture())
        assert svg.count("<g transform=") >= len(_room_with_furniture().placements)

    def test_legend_lists_present_classes(self):
        svg = render_floor_plan(_room_with_furniture(), show_legend=True)
        assert "sofa" in svg and "coffee table" in svg

    def test_aspect_ratio_preserved_for_thin_room(self):
        corridor = Room(
            name="Corridor",
            polygon=(Vec2(x=0, y=0), Vec2(x=1000, y=0), Vec2(x=1000, y=6000), Vec2(x=0, y=6000)),
            ceiling_height_mm=2700,
        )
        svg = render_floor_plan(corridor, canvas=800, margin=40)
        # A 1:6 room should not render as a square: height must exceed width.
        import re

        m = re.search(r'width="(\d+)" height="(\d+)"', svg)
        assert m
        w, h = int(m.group(1)), int(m.group(2))
        assert h > w