"""Orchestrator -- the pipeline that turns a photo into a priced layout.

Sequence:

1. probe capabilities -> pick a provider
2. perceive surfaces -> Tri signals
3. classify phase (rules table)
4. **gate on phase**: only STYLING_RESTRUCTURE rooms get furniture
5. solve layout (CP-SAT) and independently re-validate with Shapely
6. commit a new immutable scene version
7. price it

Step 4 is the part worth being explicit about. Running the solver on a room
with no floor produces a beautiful arrangement of furniture standing on bare
screed, and then quotes it. The phase gate is what stops the pipeline from
producing confident nonsense for a room that is not ready.

Every stage records into :class:`PipelineReport` rather than raising, because
"this room is not ready and here is why" is a *result*, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .core.enums import ExecutionPath, Phase, Tri
from .core.scene import CatalogueItem, Room, Scene, Vec2
from .fit.engine import FitEngine, FitResult
from .perception.probe import Capabilities, CapabilityProbe, get_probe
from .phase.rules import PhaseVerdict, classify
from .pricing.engine import PricingEngine, Quote
from .pricing.prices import PriceBook
from .providers.base import PerceptionProvider, PerceptionResult, ProviderError
from .providers.mock import MockPerceptionProvider
from .restructure.solver import (
    LayoutSolver,
    SolveRequest,
    SolveResult,
    ValidationReport,
    validate_solution,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PipelineReport:
    """Everything that happened, including the parts that did not.

    Deliberately not an exception-based flow. A room that cannot be furnished
    yet is a normal, expected outcome that the caller must be able to render in
    a UI, and turning it into a stack trace loses the reasons.
    """

    scene_id: str
    scene_version_id: str
    room_id: str
    capabilities: Capabilities | None = None
    perception: PerceptionResult | None = None
    phase: PhaseVerdict | None = None
    # The phase the pipeline actually acted on -- equals the classified phase
    # unless force_phase overrode it. Kept separate so the report never claims
    # a room was "in STYLING_RESTRUCTURE" when perception said otherwise.
    effective_phase: "Phase | None" = None
    solve: SolveResult | None = None
    validation: ValidationReport | None = None
    quote: Quote | None = None
    new_scene: Scene | None = None
    stages: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    generated_at: datetime = field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.blocked_reason is None

    def log(self, msg: str) -> None:
        self.stages.append(msg)


class Orchestrator:
    """End-to-end pipeline."""

    def __init__(
        self,
        *,
        probe: CapabilityProbe | None = None,
        perception: PerceptionProvider | None = None,
        price_book: PriceBook | None = None,
        fit_engine: FitEngine | None = None,
        solver: LayoutSolver | None = None,
    ) -> None:
        self.probe = probe or get_probe()
        self._perception_override = perception
        self.price_book = price_book or PriceBook()
        self.fit = fit_engine or FitEngine()
        self.solver = solver or LayoutSolver()

    @staticmethod
    def _surfaces_known(room: Room) -> bool:
        """Whether the scene's stored surface state is complete enough to trust.

        Any UNKNOWN means the record is incomplete and a fresh look is worth
        taking. All-known means someone (or something) already assessed this
        room, and that assessment wins.
        """
        s = room.surfaces
        values = [
            s.walls_painted,
            s.flooring_installed,
            s.ceiling_finished,
            s.electrical_terminated,
            s.plumbing_terminated,
            s.carpentry_installed,
            s.furniture_present,
        ]
        return all(v != Tri.UNKNOWN.value for v in values)

    def _select_perception(self, caps: Capabilities) -> PerceptionProvider:
        """Choose a provider for the detected path, degrading to MOCK.

        A cloud provider that cannot be constructed (missing key) falls back
        rather than failing the request -- the probe already decided cloud was
        viable, but construction is the last place to find out otherwise.
        """
        if self._perception_override is not None:
            return self._perception_override

        if caps.path is ExecutionPath.CLOUD_API:
            try:
                from .providers.gemini import GeminiPerceptionProvider

                return GeminiPerceptionProvider()
            except Exception:
                return MockPerceptionProvider()

        # LOCAL_FULL / LOCAL_LIGHT would load weights here. Not implemented in
        # this build; MOCK is the honest answer rather than a silent stub that
        # returns plausible-looking signals.
        return MockPerceptionProvider()

    def run(
        self,
        scene: Scene,
        room_id: str,
        *,
        image_ref: str = "mock://room.jpg",
        catalogue: tuple[CatalogueItem, ...] = (),
        focal_point: Vec2 | None = None,
        solve_time_limit_s: float = 10.0,
        force_phase: Phase | None = None,
        reperceive: bool = False,
    ) -> PipelineReport:
        room = scene.room(room_id)
        report = PipelineReport(
            scene_id=scene.id, scene_version_id=scene.version_id, room_id=room_id
        )

        # 1. capabilities
        caps = self.probe.detect()
        report.capabilities = caps
        report.log(f"probe -> {caps.path.value}")

        # 2. perception
        #
        # If the scene already carries fully-determined surface state, that is
        # better evidence than anything re-derived from a photo -- it came from
        # an earlier assessment that a human may have corrected. Re-perceiving
        # would silently overwrite that correction on every run.
        if not reperceive and self._surfaces_known(room):
            perception = PerceptionResult(
                surfaces=room.surfaces,
                confidence=1.0,
                path=caps.path,
                provider="scene-graph",
                notes=("surface state already recorded on the scene; not re-perceived",),
            )
            report.log("perception skipped -- scene already has known surfaces")
        else:
            provider = self._select_perception(caps)
            try:
                perception = provider.analyse(image_ref, room_id=room_id)
            except ProviderError as exc:
                report.log(f"perception failed ({exc}); falling back to mock")
                perception = MockPerceptionProvider().analyse(image_ref, room_id=room_id)
            report.log(f"perception via {perception.provider}")
        report.perception = perception

        # 3. phase
        verdict = classify(perception.surfaces)
        report.phase = verdict
        report.log(f"phase -> {verdict.phase.value} (confidence {verdict.confidence})")

        effective_phase = force_phase or verdict.phase
        report.effective_phase = effective_phase
        if force_phase is not None and force_phase is not verdict.phase:
            report.log(
                f"phase overridden: classified {verdict.phase.value}, "
                f"forced to {force_phase.value}"
            )

        # 4. phase gate
        if effective_phase is not Phase.STYLING_RESTRUCTURE:
            blockers = ", ".join(verdict.blocking_signals) or "incomplete shell"
            report.blocked_reason = (
                f"room is in {effective_phase.value}, not ready for furniture "
                f"layout -- outstanding: {blockers}"
            )
            report.log("blocked at phase gate")
            room_with_phase = room.model_copy(
                update={"surfaces": perception.surfaces, "phase": effective_phase}
            )
            report.new_scene = scene.replace_room(
                room_with_phase, notes=f"perception + phase: {effective_phase.value}"
            )
            report.scene_version_id = report.new_scene.version_id
            # Still price the surface work -- that is exactly what this room needs.
            report.quote = PricingEngine(self.price_book).quote_room(
                report.new_scene,
                room_with_phase,
                include_furniture=False,
            )
            return report

        if not catalogue:
            report.blocked_reason = "no catalogue items supplied for layout"
            report.log("nothing to place")
            return report

        # 5. solve
        solve_req = SolveRequest(
            room=room,
            items=catalogue,
            focal_point=focal_point,
            time_limit_s=solve_time_limit_s,
        )
        result = self.solver.solve(solve_req)
        report.solve = result
        report.log(f"solver -> {result.status}")

        if not result.ok:
            report.blocked_reason = (
                f"layout could not be solved: {'; '.join(result.reasons)}"
            )
            return report

        # 6. independent validation -- never trust the solver's own word
        validation = validate_solution(room, result.placements)
        report.validation = validation
        report.log(f"validation -> {'passed' if validation.ok else 'FAILED'}")

        if not validation.ok:
            report.blocked_reason = (
                "solved layout failed independent geometric validation: "
                + "; ".join(validation.violations)
            )
            return report

        # 7. commit a new immutable version
        new_room = room.model_copy(
            update={
                "placements": result.placements,
                "surfaces": perception.surfaces,
                "phase": effective_phase,
            }
        )
        new_scene = scene.replace_room(new_room, notes="restructure layout applied")
        report.new_scene = new_scene
        report.scene_version_id = new_scene.version_id
        report.log(f"committed scene version {new_scene.version} ({new_scene.version_id[:8]})")

        # 8. price
        report.quote = PricingEngine(self.price_book).quote_scene(new_scene)
        report.log(f"quote total {report.quote.total}")
        return report

    def check_item(
        self, room: Room, item: CatalogueItem, origin: Vec2, yaw: int = 0
    ) -> FitResult:
        """Single-placement feasibility, exposed for interactive drag-and-drop."""
        return self.fit.check(item, room, origin, yaw)