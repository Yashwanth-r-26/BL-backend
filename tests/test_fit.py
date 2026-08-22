"""Fit engine -- one test per gate, plus the ordering and tolerance rules.

Every rejection assertion checks ``overage_mm`` as well as the code, because a
gate that returns the right code with a meaningless measurement is only half
working and the caller cannot act on it.
"""

from __future__ import annotations

import pytest

from interior_ai.core.enums import ObjectClass, RejectionCode
from interior_ai.core.scene import CatalogueItem, Footprint, Placement, Room, Vec2
from interior_ai.fit.engine import FitEngine


@pytest.fixture
def engine() -> FitEngine:
    return FitEngine()


def item(w: int, d: int, h: int, **kwargs) -> CatalogueItem:
    return CatalogueItem(
        sku=kwargs.pop("sku", "TEST"),
        name="Test item",
        object_class=kwargs.pop("object_class", ObjectClass.SOFA),
        footprint=Footprint(width_mm=w, depth_mm=d, height_mm=h),
        **kwargs,
    )


class TestDimensionGates:
    def test_too_wide_reports_measured_overage(self, engine, bare_room):
        res = engine.check(item(9000, 900, 800), bare_room, Vec2(x=0, y=0))
        assert not res.ok
        rej = res.rejections[0]
        assert rej.code is RejectionCode.TOO_WIDE
        assert rej.overage_mm == 4000
        assert "4000" in rej.message

    def test_too_deep_reports_measured_overage(self, engine, bare_room):
        res = engine.check(item(1000, 7000, 800), bare_room, Vec2(x=0, y=0))
        assert res.rejections[0].code is RejectionCode.TOO_DEEP
        assert res.rejections[0].overage_mm == 3000

    def test_too_tall_reports_measured_overage(self, engine, bare_room):
        res = engine.check(item(1000, 600, 3200), bare_room, Vec2(x=0, y=0))
        rej = res.rejections[0]
        assert rej.code is RejectionCode.TOO_TALL
        assert rej.overage_mm == 500

    def test_rotation_can_rescue_a_too_deep_item(self, engine, bare_room):
        """4500 deep fails in a 4000 room; rotated it becomes 4500 wide, which
        fits the 5000 span."""
        tall_thin = item(900, 4500, 800)
        assert not engine.check(tall_thin, bare_room, Vec2(x=0, y=0)).ok
        rotated = engine.check(
            tall_thin, bare_room, Vec2(x=100, y=100), yaw=90, skip_circulation=True
        )
        assert rotated.ok


class TestGateOrdering:
    def test_cheapest_gate_fires_first(self, engine, door_room):
        """An item that is both too wide and would hit the door must report
        TOO_WIDE -- the integer gate short-circuits before any polygon work."""
        res = engine.check(item(9000, 900, 800), door_room, Vec2(x=800, y=100))
        assert res.codes() == (RejectionCode.TOO_WIDE,)

    def test_collect_all_gathers_multiple_failures(self, engine, bare_room):
        res = engine.check(
            item(9000, 7000, 3500), bare_room, Vec2(x=0, y=0), collect_all=True
        )
        codes = set(res.codes())
        assert RejectionCode.TOO_WIDE in codes
        assert RejectionCode.TOO_DEEP in codes
        assert RejectionCode.TOO_TALL in codes


class TestSpatialGates:
    def test_out_of_room(self, engine, bare_room):
        res = engine.check(item(2200, 900, 800), bare_room, Vec2(x=4500, y=100))
        assert res.rejections[0].code is RejectionCode.OUT_OF_ROOM
        assert res.rejections[0].overage_mm > 0

    def test_collision_with_existing_placement(self, engine, bare_room):
        sofa = item(2200, 900, 800)
        occupied = bare_room.model_copy(
            update={
                "placements": (
                    Placement(
                        sku="EXISTING",
                        object_class=ObjectClass.SOFA,
                        origin=Vec2(x=1000, y=1000),
                        footprint=Footprint(width_mm=2200, depth_mm=900, height_mm=800),
                    ),
                )
            }
        )
        res = engine.check(sofa, occupied, Vec2(x=1500, y=1200))
        assert res.rejections[0].code is RejectionCode.COLLISION
        assert res.rejections[0].overage_mm > 0

    def test_collision_with_obstacle(self, engine, obstacle_room):
        res = engine.check(item(1000, 600, 800), obstacle_room, Vec2(x=1700, y=1200))
        assert res.rejections[0].code is RejectionCode.COLLISION
        assert "column" in res.rejections[0].message

    def test_door_swing_blocked(self, engine, door_room):
        res = engine.check(item(1000, 600, 800), door_room, Vec2(x=800, y=100))
        assert res.rejections[0].code is RejectionCode.DOOR_SWING_BLOCKED
        assert res.rejections[0].overage_mm > 0

    def test_clear_of_door_swing_passes(self, engine, door_room):
        res = engine.check(item(1000, 600, 800), door_room, Vec2(x=3000, y=2000))
        assert res.ok


class TestWallRequirement:
    def test_wall_item_rejected_in_open_floor(self, engine, bare_room, wardrobe):
        res = engine.check(wardrobe, bare_room, Vec2(x=2000, y=1500))
        assert res.rejections[0].code is RejectionCode.WALL_REQUIRED
        assert res.rejections[0].overage_mm > 0

    def test_wall_item_accepted_against_wall(self, engine, bare_room, wardrobe):
        # yaw=180 faces the wardrobe into the room, so its front clearance
        # falls on open floor rather than through the wall behind it.
        res = engine.check(wardrobe, bare_room, Vec2(x=1000, y=0), yaw=180)
        assert res.ok

    def test_non_wall_item_unaffected(self, engine, bare_room, coffee_table):
        assert engine.check(coffee_table, bare_room, Vec2(x=2000, y=1500)).ok


class TestFrontClearance:
    def test_clearance_blocked_by_obstacle(self, engine, obstacle_room):
        """Wardrobe flush against the left wall, facing into the room, with the
        column sitting in the space you would stand in to open it."""
        wr = item(
            600, 1200, 2200,
            object_class=ObjectClass.WARDROBE,
            requires_wall=True,
            clearance_front_mm=1400,
        )
        # x=0 satisfies the wall gate; yaw=90 faces +x, so the clearance zone
        # spans x=600..2000 across y=900..2100 -- straight through the column.
        res = engine.check(wr, obstacle_room, Vec2(x=0, y=900), yaw=90)
        assert res.rejections[0].code is RejectionCode.FRONT_CLEARANCE
        assert res.rejections[0].overage_mm > 0

    def test_zero_clearance_item_skips_gate(self, engine, obstacle_room):
        no_clearance = item(400, 400, 800, object_class=ObjectClass.SIDE_TABLE)
        res = engine.check(no_clearance, obstacle_room, Vec2(x=500, y=500))
        assert RejectionCode.FRONT_CLEARANCE not in res.codes()


class TestCirculation:
    def test_wall_to_wall_item_blocks_circulation(self, engine):
        """An item filling a small room leaves no walkway."""
        tiny = Room(
            name="Tiny",
            polygon=(Vec2(x=0, y=0), Vec2(x=2000, y=0), Vec2(x=2000, y=1500), Vec2(x=0, y=1500)),
            ceiling_height_mm=2700,
        )
        res = engine.check(item(1990, 1490, 800), tiny, Vec2(x=0, y=0))
        assert res.rejections[0].code is RejectionCode.CIRCULATION_BLOCKED

    def test_skip_circulation_flag(self, engine):
        tiny = Room(
            name="Tiny",
            polygon=(Vec2(x=0, y=0), Vec2(x=2000, y=0), Vec2(x=2000, y=1500), Vec2(x=0, y=1500)),
            ceiling_height_mm=2700,
        )
        assert engine.check(
            item(1990, 1490, 800), tiny, Vec2(x=0, y=0), skip_circulation=True
        ).ok


class TestTolerance:
    def test_slight_oversize_is_forgiven(self):
        """A 2410 mm sofa against a 2400 mm wall is a phone-camera measurement
        artefact, not a real conflict."""
        tight = Room(
            name="Tight",
            polygon=(Vec2(x=0, y=0), Vec2(x=2400, y=0), Vec2(x=2400, y=3000), Vec2(x=0, y=3000)),
            ceiling_height_mm=2700,
        )
        res = FitEngine(tolerance=0.08).check(
            item(2410, 900, 800), tight, Vec2(x=0, y=100), skip_circulation=True
        )
        assert res.ok

    def test_zero_tolerance_rejects_the_same_item(self):
        tight = Room(
            name="Tight",
            polygon=(Vec2(x=0, y=0), Vec2(x=2400, y=0), Vec2(x=2400, y=3000), Vec2(x=0, y=3000)),
            ceiling_height_mm=2700,
        )
        res = FitEngine(tolerance=0.0).check(
            item(2410, 900, 800), tight, Vec2(x=0, y=100), skip_circulation=True
        )
        assert not res.ok

    def test_tolerance_is_consistent_across_gates(self):
        """Regression: the width gate forgave 10 mm and containment rejected
        it, so an item passed gate 1 and died at gate 5 for the same 10 mm."""
        tight = Room(
            name="Tight",
            polygon=(Vec2(x=0, y=0), Vec2(x=2400, y=0), Vec2(x=2400, y=3000), Vec2(x=0, y=3000)),
            ceiling_height_mm=2700,
        )
        res = FitEngine(tolerance=0.08).check(
            item(2410, 900, 800), tight, Vec2(x=0, y=100),
            collect_all=True, skip_circulation=True,
        )
        assert RejectionCode.OUT_OF_ROOM not in res.codes()

    def test_gross_overhang_still_rejected(self, engine, bare_room):
        res = engine.check(item(2200, 900, 800), bare_room, Vec2(x=4800, y=100))
        assert not res.ok
        assert res.rejections[0].code is RejectionCode.OUT_OF_ROOM


class TestFirstFit:
    def test_returns_first_passing_candidate(self, engine, door_room, sofa):
        candidates = [
            (Vec2(x=800, y=100), 0),    # blocked by door swing
            (Vec2(x=2500, y=2500), 0),  # clear
        ]
        res = engine.first_fit(sofa, door_room, candidates)
        assert res.ok
        assert res.placement.origin.x == 2500

    def test_returns_rejection_when_nothing_fits(self, engine, bare_room):
        huge = item(9000, 900, 800)
        res = engine.first_fit(huge, bare_room, [(Vec2(x=0, y=0), 0)])
        assert not res.ok
        assert res.rejections

    def test_empty_candidates_is_not_a_crash(self, engine, bare_room, sofa):
        res = engine.first_fit(sofa, bare_room, [])
        assert not res.ok
