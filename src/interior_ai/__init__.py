"""Interior Design AI.

Scene-graph-driven interior design: geometric fit checking, rules-based phase
classification, CP-SAT layout optimisation, and reproducible quantity-based
pricing.

The scene graph is the single source of truth. Renders are views of it, prices
are calculations over it, and "that won't fit" is a fact proved against it.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Load .env before anything reads os.getenv. Doing this at package import
# means uvicorn, Alembic, the seed script and a bare `python -c` all pick up
# the same configuration without each having to remember. Real environment
# variables still win; INTERIOR_AI_SKIP_DOTENV=1 disables the file entirely.
from .config import describe_env, find_dotenv, load_env

load_env()

from .core.enums import (
    ExecutionPath,
    ObjectClass,
    OpeningKind,
    Phase,
    PriceStatus,
    RejectionCode,
    SwingDirection,
    Tri,
    Unit,
)
from .core.scene import (
    CatalogueItem,
    Footprint,
    Obstacle,
    Opening,
    Placement,
    Room,
    Scene,
    SurfaceState,
    Vec2,
)
from .fit.engine import FitEngine, FitResult, Rejection
from .orchestrator import Orchestrator, PipelineReport
from .perception.probe import Capabilities, CapabilityProbe
from .phase.rules import PhaseVerdict, classify, restart_from_empty
from .pricing.engine import BOQLine, PricingEngine, Quote
from .pricing.prices import PriceBook, PriceObservation, PriceSnapshot
from .restructure.solver import LayoutSolver, SolveRequest, validate_solution

__all__ = [
    "__version__",
    # configuration
    "load_env", "find_dotenv", "describe_env",
    # enums
    "ExecutionPath", "ObjectClass", "OpeningKind", "Phase", "PriceStatus",
    "RejectionCode", "SwingDirection", "Tri", "Unit",
    # scene graph
    "CatalogueItem", "Footprint", "Obstacle", "Opening", "Placement",
    "Room", "Scene", "SurfaceState", "Vec2",
    # engines
    "FitEngine", "FitResult", "Rejection",
    "CapabilityProbe", "Capabilities",
    "classify", "restart_from_empty", "PhaseVerdict",
    "LayoutSolver", "SolveRequest", "validate_solution",
    "PriceBook", "PriceObservation", "PriceSnapshot",
    "PricingEngine", "Quote", "BOQLine",
    "Orchestrator", "PipelineReport",
]