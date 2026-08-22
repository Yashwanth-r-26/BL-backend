"""Integer millimetre discipline.

Everything geometric in this service is an ``int`` count of millimetres.

The reason is drift. A room dimension makes the round trip
JSON -> Pydantic -> Postgres -> CP-SAT -> Postgres -> JSON many times in a
single quote, and CP-SAT in particular only accepts integers. If any leg of
that trip is a float, 3000.0 becomes 2999.9999999999995 and a wall that used
to fit stops fitting. Storing integers makes the round trip exact.

Conversions to human units (m^2, litres) happen only at the reporting edge --
see :mod:`interior_ai.pricing.takeoff`.
"""

from __future__ import annotations

MM_PER_M = 1000
MM2_PER_M2 = MM_PER_M * MM_PER_M

# Grid the CP-SAT solver snaps to. 50 mm is fine enough that furniture looks
# deliberately placed and coarse enough that the model stays solvable.
SOLVER_GRID_MM = 50


def m_to_mm(metres: float) -> int:
    """Convert metres to integer millimetres, rounding half away from zero."""
    return int(round(metres * MM_PER_M))


def mm_to_m(mm: int) -> float:
    """Convert millimetres to metres. Reporting edge only."""
    return mm / MM_PER_M


def mm2_to_m2(mm2: int) -> float:
    """Convert square millimetres to square metres. Reporting edge only."""
    return mm2 / MM2_PER_M2


def snap_to_grid(mm: int, grid: int = SOLVER_GRID_MM) -> int:
    """Snap a millimetre value to the nearest grid multiple."""
    if grid <= 0:
        raise ValueError("grid must be positive")
    return int(round(mm / grid)) * grid


def snap_down(mm: int, grid: int = SOLVER_GRID_MM) -> int:
    """Snap down to a grid multiple. Used for upper bounds so the solver
    never proposes a position that overruns the room by a partial cell."""
    if grid <= 0:
        raise ValueError("grid must be positive")
    return (mm // grid) * grid


def apply_tolerance(mm: int, tolerance_pct: float) -> int:
    """Expand a dimension by a tolerance percentage.

    Used by the fit engine so a 2410 mm sofa is not rejected from a 2400 mm
    wall on a measurement that came from a phone camera.
    """
    if tolerance_pct < 0:
        raise ValueError("tolerance must be non-negative")
    return int(round(mm * (1.0 + tolerance_pct)))
