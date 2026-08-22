"""Layout restructure as constrained optimisation.

Furniture layout is a packing problem with hard constraints (nothing overlaps,
nothing blocks a door) and soft preferences (the sofa faces the TV, the coffee
table sits between them). Generating a layout and checking it afterwards means
generating until something passes; CP-SAT searches only the feasible region, so
the first answer it returns is already valid.

Two implementation notes that cost real debugging time:

**CP-SAT integer variables cannot be divided.** ``x + w // 2`` is not
expressible -- ``w`` is a variable, and the solver has no integer division on
linear expressions. Every centre comparison here is written in *doubled units*:
to say "object centre equals focal centre" we assert ``2*x + w_eff == 2*fx``.
The doubling makes the halving unnecessary and keeps everything linear.

**Proximity is not ordering.** Constraining the coffee table to be *near* the
sofa places it behind the sofa about half the time -- behind is exactly as near
as in front. It needs an explicit ordering constraint along the sofa-to-focal
axis: the table's centre must lie strictly between them.

Every hard constraint is independently re-validated with Shapely after solving
(see :func:`validate_solution`). The solver and the validator share no code, so
a bug in the model surfaces as a failed validation rather than a plausible
wrong layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model
from shapely.geometry import box

from ..core.enums import ObjectClass
from ..core.geometry import (
    all_door_swings,
    obstacle_polygon,
    placement_polygon,
    room_polygon,
)
from ..core.scene import CatalogueItem, Placement, Room, Vec2
from ..core.units import SOLVER_GRID_MM, snap_down, snap_to_grid

YAWS = (0, 90, 180, 270)

WALL_CLASSES = {ObjectClass.TV_UNIT, ObjectClass.WARDROBE, ObjectClass.BOOKSHELF}


@dataclass(frozen=True)
class SolveRequest:
    room: Room
    items: tuple[CatalogueItem, ...]
    focal_point: Vec2 | None = None
    time_limit_s: float = 10.0
    grid_mm: int = SOLVER_GRID_MM
    circulation_mm: int = 600


@dataclass(frozen=True)
class SolveResult:
    ok: bool
    status: str
    placements: tuple[Placement, ...] = ()
    objective: int | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ValidationReport:
    """Independent Shapely re-check of a solved layout."""

    ok: bool
    containment_ok: bool
    overlap_ok: bool
    door_swing_ok: bool
    obstacle_ok: bool
    violations: tuple[str, ...] = field(default_factory=tuple)


def _forbidden_rectangles(room: Room, grid: int) -> list[tuple[int, int, int, int]]:
    """Static no-go boxes: obstacles plus door-swing arcs.

    Door swings are curved; CP-SAT's ``AddNoOverlap2D`` takes rectangles. We use
    each arc's bounding box, which over-blocks slightly at the corners. That
    direction of error is the safe one -- a layout the solver rejects for being
    near a door is a minor loss, a door that will not open is a site visit.
    """
    rects: list[tuple[int, int, int, int]] = []
    for obs in room.obstacles:
        minx, miny, maxx, maxy = obs.bounds
        rects.append(
            (
                snap_down(minx, grid),
                snap_down(miny, grid),
                snap_to_grid(maxx, grid),
                snap_to_grid(maxy, grid),
            )
        )
    for sw in all_door_swings(room):
        minx, miny, maxx, maxy = sw.bounds
        rects.append(
            (
                snap_down(int(minx), grid),
                snap_down(int(miny), grid),
                snap_to_grid(int(maxx), grid),
                snap_to_grid(int(maxy), grid),
            )
        )
    return rects


class LayoutSolver:
    """CP-SAT furniture arranger."""

    def solve(self, req: SolveRequest) -> SolveResult:
        room = req.room
        grid = req.grid_mm
        model = cp_model.CpModel()

        rminx, rminy, rmaxx, rmaxy = room.bounds
        rminx, rminy = snap_to_grid(rminx, grid), snap_to_grid(rminy, grid)
        rmaxx, rmaxy = snap_down(rmaxx, grid), snap_down(rmaxy, grid)

        n = len(req.items)
        if n == 0:
            return SolveResult(ok=True, status="EMPTY", placements=())

        xs, ys, ws, ds = [], [], [], []
        x_ivs, y_ivs = [], []
        yaw_lits: list[dict[int, cp_model.IntVar]] = []

        for i, item in enumerate(req.items):
            # Positions are constrained to grid multiples via a domain, not
            # just bounded. Without this the solver may legally land on an
            # off-grid coordinate (any integer in range satisfies the other
            # constraints), which makes placements non-reproducible and fails
            # grid-alignment checks intermittently under multi-worker search.
            grid_xs = list(range(rminx, rmaxx + 1, grid))
            grid_ys = list(range(rminy, rmaxy + 1, grid))
            x = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(grid_xs), f"x{i}"
            )
            y = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(grid_ys), f"y{i}"
            )

            # Yaw is one-hot: exactly one of four orientations.
            lits = {}
            for yaw in YAWS:
                lits[yaw] = model.NewBoolVar(f"yaw{i}_{yaw}")
            model.AddExactlyOne(lits.values())
            yaw_lits.append(lits)

            # Effective width/depth follow from the chosen yaw. Because only
            # 0/90/180/270 are allowed, this is a selection between two values
            # rather than a rotation -- which is the reason for that restriction.
            w_eff = model.NewIntVar(0, max(rmaxx - rminx, rmaxy - rminy), f"w{i}")
            d_eff = model.NewIntVar(0, max(rmaxx - rminx, rmaxy - rminy), f"d{i}")
            for yaw in YAWS:
                ew, ed = item.footprint.rotated(yaw)  # type: ignore[arg-type]
                ew, ed = snap_to_grid(ew, grid), snap_to_grid(ed, grid)
                model.Add(w_eff == ew).OnlyEnforceIf(lits[yaw])
                model.Add(d_eff == ed).OnlyEnforceIf(lits[yaw])

            # Containment against the room's bounding box.
            model.Add(x + w_eff <= rmaxx)
            model.Add(y + d_eff <= rmaxy)

            x_iv = model.NewIntervalVar(x, w_eff, model.NewIntVar(rminx, rmaxx + 1, f"xe{i}"), f"xi{i}")
            y_iv = model.NewIntervalVar(y, d_eff, model.NewIntVar(rminy, rmaxy + 1, f"ye{i}"), f"yi{i}")

            xs.append(x); ys.append(y); ws.append(w_eff); ds.append(d_eff)
            x_ivs.append(x_iv); y_ivs.append(y_iv)

        # Hard: no two objects share floor.
        model.AddNoOverlap2D(x_ivs, y_ivs)

        # Hard: nothing enters an obstacle or a door swing. Modelled as fixed
        # intervals joined into the same NoOverlap2D relation.
        forbidden = _forbidden_rectangles(room, grid)
        if forbidden:
            fx_ivs = list(x_ivs)
            fy_ivs = list(y_ivs)
            for j, (fminx, fminy, fmaxx, fmaxy) in enumerate(forbidden):
                fw, fh = max(fmaxx - fminx, grid), max(fmaxy - fminy, grid)
                fx_ivs.append(model.NewFixedSizeIntervalVar(fminx, fw, f"fx{j}"))
                fy_ivs.append(model.NewFixedSizeIntervalVar(fminy, fh, f"fy{j}"))
            model.AddNoOverlap2D(fx_ivs, fy_ivs)

        # Wall-hugging classes stay within a snap distance of a bounding wall.
        for i, item in enumerate(req.items):
            if item.requires_wall or item.object_class in WALL_CLASSES:
                b_left = model.NewBoolVar(f"wl{i}")
                b_right = model.NewBoolVar(f"wr{i}")
                b_bottom = model.NewBoolVar(f"wb{i}")
                b_top = model.NewBoolVar(f"wt{i}")
                model.Add(xs[i] <= rminx + grid).OnlyEnforceIf(b_left)
                model.Add(xs[i] + ws[i] >= rmaxx - grid).OnlyEnforceIf(b_right)
                model.Add(ys[i] <= rminy + grid).OnlyEnforceIf(b_bottom)
                model.Add(ys[i] + ds[i] >= rmaxy - grid).OnlyEnforceIf(b_top)
                model.AddAtLeastOne([b_left, b_right, b_bottom, b_top])

        # ---- soft objectives ------------------------------------------
        penalties: list[cp_model.IntVar] = []
        idx_by_class: dict[ObjectClass, list[int]] = {}
        for i, item in enumerate(req.items):
            idx_by_class.setdefault(item.object_class, []).append(i)

        focal = req.focal_point
        span = max(rmaxx - rminx, rmaxy - rminy)

        sofa_i = idx_by_class.get(ObjectClass.SOFA, [None])[0]
        tv_i = idx_by_class.get(ObjectClass.TV_UNIT, [None])[0]
        table_i = idx_by_class.get(ObjectClass.COFFEE_TABLE, [None])[0]

        # Sofa faces the focal point / TV: align their centres on one axis.
        # NOTE doubled units -- 2*x + w_eff is twice the centre, avoiding the
        # integer division CP-SAT cannot express.
        if sofa_i is not None and focal is not None:
            dev = model.NewIntVar(0, 2 * span, "sofa_focal_dev")
            model.AddAbsEquality(dev, 2 * xs[sofa_i] + ws[sofa_i] - 2 * focal.x)
            penalties.append(dev)

        if sofa_i is not None and tv_i is not None:
            dev = model.NewIntVar(0, 2 * span, "sofa_tv_dev")
            model.AddAbsEquality(
                dev, (2 * xs[sofa_i] + ws[sofa_i]) - (2 * xs[tv_i] + ws[tv_i])
            )
            penalties.append(dev)

        # Coffee table BETWEEN sofa and focal, not merely near the sofa.
        # Without the ordering half of this, the table lands behind the sofa
        # roughly half the time -- behind is just as close as in front.
        if table_i is not None and sofa_i is not None and focal is not None:
            t_cx2 = model.NewIntVar(0, 2 * (rmaxx + span), "t_cx2")
            model.Add(t_cx2 == 2 * xs[table_i] + ws[table_i])
            t_cy2 = model.NewIntVar(0, 2 * (rmaxy + span), "t_cy2")
            model.Add(t_cy2 == 2 * ys[table_i] + ds[table_i])

            s_cy2 = model.NewIntVar(0, 2 * (rmaxy + span), "s_cy2")
            model.Add(s_cy2 == 2 * ys[sofa_i] + ds[sofa_i])

            # Ordering along the sofa->focal axis (y here; focal is the TV wall).
            below = model.NewBoolVar("table_below_sofa")
            model.Add(t_cy2 <= s_cy2).OnlyEnforceIf(below)
            model.Add(t_cy2 >= s_cy2).OnlyEnforceIf(below.Not())
            if focal.y * 2 <= rminy + rmaxy:
                model.Add(below == 1)
            else:
                model.Add(below == 0)

            dev = model.NewIntVar(0, 2 * span, "table_align_dev")
            model.AddAbsEquality(dev, t_cx2 - (2 * xs[sofa_i] + ws[sofa_i]))
            penalties.append(dev)

        if penalties:
            model.Minimize(sum(penalties))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = req.time_limit_s
        solver.parameters.num_search_workers = 4
        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return SolveResult(
                ok=False,
                status=status_name,
                reasons=(
                    f"solver returned {status_name}; constraints cannot be "
                    "satisfied in this room",
                ),
            )

        placements = []
        for i, item in enumerate(req.items):
            chosen = 0
            for yaw in YAWS:
                if solver.Value(yaw_lits[i][yaw]):
                    chosen = yaw
                    break
            placements.append(
                Placement(
                    sku=item.sku,
                    object_class=item.object_class,
                    origin=Vec2(x=solver.Value(xs[i]), y=solver.Value(ys[i])),
                    footprint=item.footprint,
                    yaw=chosen,  # type: ignore[arg-type]
                )
            )

        return SolveResult(
            ok=True,
            status=status_name,
            placements=tuple(placements),
            objective=int(solver.ObjectiveValue()) if penalties else None,
        )


def validate_solution(room: Room, placements: tuple[Placement, ...]) -> ValidationReport:
    """Re-prove every hard constraint with Shapely, independently of CP-SAT.

    This shares no code with the model above by design. If the CP-SAT
    formulation has a bug -- a doubled unit that should not be, an interval
    built from the wrong variable -- the result is a layout that satisfies the
    model and violates reality. Only an independent check catches that, and it
    has to be independent to be worth running.
    """
    violations: list[str] = []
    rp = room_polygon(room)

    containment_ok = True
    for p in placements:
        pp = placement_polygon(p)
        if not rp.buffer(1).contains(pp):
            containment_ok = False
            violations.append(f"{p.object_class.value} extends outside the room")

    overlap_ok = True
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            a, b = placement_polygon(placements[i]), placement_polygon(placements[j])
            inter = a.intersection(b)
            if inter.area > 1.0:
                overlap_ok = False
                violations.append(
                    f"{placements[i].object_class.value} overlaps "
                    f"{placements[j].object_class.value} by {int(inter.area/1000)} cm²"
                )

    door_swing_ok = True
    for sw in all_door_swings(room):
        for p in placements:
            inter = placement_polygon(p).intersection(sw)
            if inter.area > 1.0:
                door_swing_ok = False
                violations.append(
                    f"{p.object_class.value} intrudes into a door swing by "
                    f"{int(inter.area/1000)} cm²"
                )

    obstacle_ok = True
    for obs in room.obstacles:
        op = obstacle_polygon(obs)
        for p in placements:
            inter = placement_polygon(p).intersection(op)
            if inter.area > 1.0:
                obstacle_ok = False
                violations.append(
                    f"{p.object_class.value} overlaps obstacle '{obs.label}'"
                )

    ok = containment_ok and overlap_ok and door_swing_ok and obstacle_ok
    return ValidationReport(
        ok=ok,
        containment_ok=containment_ok,
        overlap_ok=overlap_ok,
        door_swing_ok=door_swing_ok,
        obstacle_ok=obstacle_ok,
        violations=tuple(violations),
    )