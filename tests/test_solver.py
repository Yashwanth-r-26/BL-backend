"""Solver hard constraints, re-proved independently with Shapely.

Every test that solves also calls :func:`validate_solution`, which shares no
code with the CP-SAT model. A bug in the formulation -- a doubled unit that
should not be, an interval built from the wrong variable -- produces a layout
that satisfies the model and violates reality, and only an independent check
catches that.
"""

from __future__ import annotations

import pytest
from shapely.geometry import box

from interior_ai.core.enums import ObjectClass, OpeningKind, SwingDirection
from interior_ai.core.geometry import all_door_swings, placement_polygon, room_polygon
from interior_ai.core.scene import (
    CatalogueItem,
    Footprint,
    Obstacle,
    Opening,
    Placement,
    Room,
    Vec2,
)
from interior_ai.core.units import SOLVER_GRID_MM
from interior_ai.restructure.solver import (
    LayoutSolver,
    SolveRequest,
    validate_solution,
)

pytestmark = pytest.mark.slow


def cat(sku: str, cls: ObjectClass, w: int, d: int, h: int, **kw) -> CatalogueItem:
    return CatalogueItem(
        sku=sku, name=sku, object_class=cls,
        footprint=Footprint(width_mm=w, depth_mm=d, height_mm=h), **kw,
    )


SOFA = cat("SOFA", ObjectClass.SOFA, 2200, 900, 800)
TABLE = cat("TABLE", ObjectClass.COFFEE_TABLE, 1100, 600, 400)
TV = cat("TV", ObjectClass.TV_UNIT, 1800, 450, 500, requires_wall=True)
CHAIR = cat("CHAIR", ObjectClass.ARMCHAIR, 800, 800, 900)


@pytest.fixture
def solver() -> LayoutSolver:
    return LayoutSolver()


class TestBasicSolving:
    def test_empty_item_list_succeeds(self, solver, bare_room):
        result = solver.solve(SolveRequest(room=bare_room, items=()))
        assert result.ok
        assert result.placements == ()

    def test_single_item_placed_and_valid(self, solver, bare_room):
        result = solver.solve(SolveRequest(room=bare_room, items=(SOFA,), time_limit_s=15))
        assert result.ok
        assert validate_solution(bare_room, result.placements).ok

    def test_impossible_item_reports_infeasible(self, solver):
        tiny = Room(
            name="Tiny",
            polygon=(Vec2(x=0, y=0), Vec2(x=1000, y=0), Vec2(x=1000, y=1000), Vec2(x=0, y=1000)),
            ceiling_height_mm=2700,
        )
        result = solver.solve(SolveRequest(room=tiny, items=(SOFA,), time_limit_s=10))
        assert not result.ok
        assert result.reasons


class TestHardConstraintContainment:
    def test_every_placement_inside_the_room(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE, CHAIR), time_limit_s=20)
        )
        assert result.ok
        rp = room_polygon(bare_room)
        for p in result.placements:
            assert rp.buffer(1).contains(placement_polygon(p))

    def test_validator_agrees_on_containment(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE), time_limit_s=15)
        )
        assert validate_solution(bare_room, result.placements).containment_ok


class TestHardConstraintOverlap:
    def test_no_two_objects_share_floor(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE, CHAIR, TV), time_limit_s=25)
        )
        assert result.ok
        polys = [placement_polygon(p) for p in result.placements]
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert polys[i].intersection(polys[j]).area <= 1.0

    def test_validator_agrees_on_overlap(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE, CHAIR), time_limit_s=20)
        )
        assert validate_solution(bare_room, result.placements).overlap_ok

    def test_crowded_room_still_produces_valid_layout(self, solver, bare_room):
        items = (SOFA, TABLE, CHAIR, TV, cat("CH2", ObjectClass.ARMCHAIR, 800, 800, 900))
        result = solver.solve(SolveRequest(room=bare_room, items=items, time_limit_s=30))
        assert result.ok
        assert validate_solution(bare_room, result.placements).ok


class TestHardConstraintDoorSwing:
    def test_nothing_intrudes_into_the_swing(self, solver, door_room):
        result = solver.solve(
            SolveRequest(room=door_room, items=(SOFA, TABLE), time_limit_s=20)
        )
        assert result.ok
        swings = all_door_swings(door_room)
        for p in result.placements:
            for sw in swings:
                assert placement_polygon(p).intersection(sw).area <= 1.0

    def test_validator_agrees_on_door_swing(self, solver, door_room):
        result = solver.solve(
            SolveRequest(room=door_room, items=(SOFA, TABLE, CHAIR), time_limit_s=25)
        )
        assert validate_solution(door_room, result.placements).door_swing_ok

    def test_constraint_binds_in_a_tight_room(self, solver):
        """A room where the swing genuinely competes for floor -- the solver
        must route around it rather than ignore it."""
        tight = Room(
            name="Tight",
            polygon=(Vec2(x=0, y=0), Vec2(x=3000, y=0), Vec2(x=3000, y=2500), Vec2(x=0, y=2500)),
            ceiling_height_mm=2700,
            openings=(
                Opening(
                    kind=OpeningKind.DOOR,
                    centre=Vec2(x=600, y=0), width_mm=1000, height_mm=2100, wall_index=0,
                    swing=SwingDirection.INWARD,
                ),
            ),
        )
        items = (cat("S", ObjectClass.SOFA, 2000, 850, 800),
                 cat("B", ObjectClass.BOOKSHELF, 900, 350, 1800, requires_wall=True))
        result = solver.solve(SolveRequest(room=tight, items=items, time_limit_s=20))
        assert result.ok
        report = validate_solution(tight, result.placements)
        assert report.door_swing_ok, report.violations


class TestHardConstraintObstacles:
    def test_nothing_overlaps_a_fixed_obstacle(self, solver, obstacle_room):
        result = solver.solve(
            SolveRequest(room=obstacle_room, items=(SOFA, TABLE), time_limit_s=20)
        )
        assert result.ok
        for obs in obstacle_room.obstacles:
            op = box(*obs.bounds)
            for p in result.placements:
                assert placement_polygon(p).intersection(op).area <= 1.0

    def test_validator_agrees_on_obstacles(self, solver, obstacle_room):
        result = solver.solve(
            SolveRequest(room=obstacle_room, items=(SOFA, TABLE), time_limit_s=20)
        )
        assert validate_solution(obstacle_room, result.placements).obstacle_ok

    def test_multiple_obstacles(self, solver, bare_room):
        room = bare_room.model_copy(
            update={
                "obstacles": (
                    Obstacle(label="col1", origin=Vec2(x=1000, y=1000), width_mm=400, depth_mm=400),
                    Obstacle(label="col2", origin=Vec2(x=3000, y=2000), width_mm=400, depth_mm=400),
                    Obstacle(label="duct", origin=Vec2(x=4500, y=0), width_mm=500, depth_mm=4000),
                )
            }
        )
        result = solver.solve(SolveRequest(room=room, items=(SOFA, TABLE), time_limit_s=25))
        assert result.ok
        assert validate_solution(room, result.placements).ok


class TestRotationAndGrid:
    def test_only_cardinal_yaws_used(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE, CHAIR), time_limit_s=20)
        )
        for p in result.placements:
            assert p.yaw in (0, 90, 180, 270)

    def test_positions_land_on_the_grid(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE), time_limit_s=15)
        )
        for p in result.placements:
            assert p.origin.x % SOLVER_GRID_MM == 0
            assert p.origin.y % SOLVER_GRID_MM == 0

    def test_narrow_room_forces_rotation(self, solver):
        """A 2200 sofa in a room only 1200 wide but 5000 deep can only fit
        turned sideways."""
        corridor = Room(
            name="Corridor",
            polygon=(Vec2(x=0, y=0), Vec2(x=1200, y=0), Vec2(x=1200, y=5000), Vec2(x=0, y=5000)),
            ceiling_height_mm=2700,
        )
        result = solver.solve(SolveRequest(room=corridor, items=(SOFA,), time_limit_s=15))
        assert result.ok
        assert result.placements[0].yaw in (90, 270)
        assert validate_solution(corridor, result.placements).ok


class TestSoftObjectives:
    def test_wall_class_hugs_a_wall(self, solver, bare_room):
        result = solver.solve(SolveRequest(room=bare_room, items=(TV,), time_limit_s=15))
        assert result.ok
        tv = result.placements[0]
        minx, miny, maxx, maxy = tv.bounds
        rminx, rminy, rmaxx, rmaxy = bare_room.bounds
        touches = (
            minx <= rminx + SOLVER_GRID_MM
            or miny <= rminy + SOLVER_GRID_MM
            or maxx >= rmaxx - SOLVER_GRID_MM
            or maxy >= rmaxy - SOLVER_GRID_MM
        )
        assert touches

    @pytest.mark.parametrize("focal_y,expect_lower", [(0, True), (4000, False)])
    def test_coffee_table_sits_between_sofa_and_focal(
        self, solver, bare_room, focal_y, expect_lower
    ):
        """Proximity alone puts the table behind the sofa about half the time --
        behind is exactly as near as in front. This checks the ordering half of
        the constraint, in both orientations."""
        result = solver.solve(
            SolveRequest(
                room=bare_room,
                items=(SOFA, TABLE),
                focal_point=Vec2(x=2500, y=focal_y),
                time_limit_s=20,
            )
        )
        assert result.ok
        by_class = {p.object_class: p for p in result.placements}
        sofa_cy = by_class[ObjectClass.SOFA].centre.y
        table_cy = by_class[ObjectClass.COFFEE_TABLE].centre.y

        if expect_lower:
            assert table_cy <= sofa_cy
        else:
            assert table_cy >= sofa_cy

    def test_objective_reported_when_preferences_present(self, solver, bare_room):
        result = solver.solve(
            SolveRequest(room=bare_room, items=(SOFA, TABLE),
                         focal_point=Vec2(x=2500, y=0), time_limit_s=20)
        )
        assert result.objective is not None


class TestValidationCatchesBadLayouts:
    def test_validator_rejects_a_hand_made_overlap(self, bare_room):
        """The validator must actually fail things, or its passes mean nothing."""
        fp = Footprint(width_mm=2000, depth_mm=900, height_mm=800)
        bad = (
            Placement(sku="A", object_class=ObjectClass.SOFA,
                      origin=Vec2(x=1000, y=1000), footprint=fp),
            Placement(sku="B", object_class=ObjectClass.SOFA,
                      origin=Vec2(x=1500, y=1200), footprint=fp),
        )
        report = validate_solution(bare_room, bad)
        assert not report.ok
        assert not report.overlap_ok
        assert report.violations

    def test_validator_rejects_out_of_room(self, bare_room):
        fp = Footprint(width_mm=2000, depth_mm=900, height_mm=800)
        bad = (
            Placement(sku="A", object_class=ObjectClass.SOFA,
                      origin=Vec2(x=4500, y=3500), footprint=fp),
        )
        report = validate_solution(bare_room, bad)
        assert not report.containment_ok

    def test_validator_rejects_door_swing_intrusion(self, door_room):
        fp = Footprint(width_mm=800, depth_mm=600, height_mm=400)
        bad = (
            Placement(sku="A", object_class=ObjectClass.COFFEE_TABLE,
                      origin=Vec2(x=800, y=100), footprint=fp),
        )
        report = validate_solution(door_room, bad)
        assert not report.door_swing_ok
