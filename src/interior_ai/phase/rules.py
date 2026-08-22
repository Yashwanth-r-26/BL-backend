"""Phase classification -- a rules table, not a classifier.

Which phase a room is in determines what work gets quoted. Getting it wrong
does not produce a slightly-off render; it produces a quote for painting a room
that is already painted, or for styling a room with bare screed floors.

That is a decision that has to be *inspectable* and *arguable*. A learned
classifier gives a label and a confidence, and when a site supervisor disputes
it there is nothing to point at. This table gives a label, a confidence, and
the specific signals that produced it.

Two rules the table exists to enforce:

**PARTIAL blocks progression.** A half-painted room is the common case, not the
edge case. Rounding PARTIAL up to YES quotes the room as finished and omits the
remaining work; rounding it down to NO re-quotes work already paid for. Neither
is acceptable, so PARTIAL holds the room in the phase that owns that work.

**UNKNOWN yields low confidence, never a wrong answer.** When perception cannot
see the floor, the honest output is "SURFACE_FINISHING, confidence 0.35,
because I could not assess the flooring", which routes to human review. Guessing
produces a confident wrong answer that nobody checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.enums import Phase, Tri
from ..core.scene import SurfaceState

# Signals that must be settled before the room leaves each phase.
SURFACE_SIGNALS = ("walls_painted", "flooring_installed", "ceiling_finished")
FIXTURE_SIGNALS = ("electrical_terminated", "plumbing_terminated", "carpentry_installed")

CONFIDENCE_CERTAIN = 0.95
CONFIDENCE_PARTIAL = 0.75
CONFIDENCE_UNKNOWN = 0.35
CONFIDENCE_MIXED = 0.55

REVIEW_THRESHOLD = 0.5


@dataclass(frozen=True)
class PhaseVerdict:
    """Classification result with its full justification.

    ``blocking_signals`` is the list a site report can print verbatim: these are
    the specific things standing between this room and the next phase.
    """

    phase: Phase
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    blocking_signals: tuple[str, ...] = field(default_factory=tuple)
    unknown_signals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_review(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD

    def explain(self) -> str:
        return "; ".join(self.reasons)


def _tri(value: str | Tri) -> Tri:
    if isinstance(value, Tri):
        return value
    try:
        return Tri(value)
    except ValueError:
        return Tri.UNKNOWN


def _group(state: SurfaceState, names: tuple[str, ...]) -> dict[str, Tri]:
    return {n: _tri(getattr(state, n)) for n in names}


def classify(state: SurfaceState) -> PhaseVerdict:
    """Map perception signals to a renovation phase.

    Evaluated in build order -- surfaces, then fixtures, then styling -- because
    that is the order the work actually happens in. A room cannot be in
    FIXTURES_CARPENTRY while its floor is missing, regardless of how many
    cabinets are already hanging.
    """
    reasons: list[str] = []
    blocking: list[str] = []
    unknown: list[str] = []

    surface = _group(state, SURFACE_SIGNALS)
    fixture = _group(state, FIXTURE_SIGNALS)

    unknown.extend(n for n, v in surface.items() if v is Tri.UNKNOWN)
    unknown.extend(n for n, v in fixture.items() if v is Tri.UNKNOWN)

    # --- Phase 1: surface finishing -------------------------------------
    surface_partial = [n for n, v in surface.items() if v is Tri.PARTIAL]
    surface_no = [n for n, v in surface.items() if v is Tri.NO]
    surface_unknown = [n for n, v in surface.items() if v is Tri.UNKNOWN]

    if surface_no:
        blocking.extend(surface_no)
        reasons.append(f"surfaces not started: {', '.join(sorted(surface_no))}")
        conf = CONFIDENCE_CERTAIN if not surface_unknown else CONFIDENCE_MIXED
        if surface_unknown:
            reasons.append(f"unassessed: {', '.join(sorted(surface_unknown))}")
        return PhaseVerdict(
            phase=Phase.SURFACE_FINISHING,
            confidence=conf,
            reasons=tuple(reasons),
            blocking_signals=tuple(sorted(blocking)),
            unknown_signals=tuple(sorted(unknown)),
        )

    if surface_partial:
        # The load-bearing rule. PARTIAL holds the room here.
        blocking.extend(surface_partial)
        reasons.append(
            f"surfaces only partially complete: {', '.join(sorted(surface_partial))} "
            "-- partial work blocks progression"
        )
        return PhaseVerdict(
            phase=Phase.SURFACE_FINISHING,
            confidence=CONFIDENCE_PARTIAL,
            reasons=tuple(reasons),
            blocking_signals=tuple(sorted(blocking)),
            unknown_signals=tuple(sorted(unknown)),
        )

    if surface_unknown:
        reasons.append(
            f"cannot assess surfaces: {', '.join(sorted(surface_unknown))} "
            "-- defaulting to earliest phase at low confidence"
        )
        return PhaseVerdict(
            phase=Phase.SURFACE_FINISHING,
            confidence=CONFIDENCE_UNKNOWN,
            reasons=tuple(reasons),
            blocking_signals=(),
            unknown_signals=tuple(sorted(unknown)),
        )

    reasons.append("all surfaces complete")

    # --- Phase 2: fixtures and carpentry --------------------------------
    fixture_partial = [n for n, v in fixture.items() if v is Tri.PARTIAL]
    fixture_no = [n for n, v in fixture.items() if v is Tri.NO]
    fixture_unknown = [n for n, v in fixture.items() if v is Tri.UNKNOWN]

    if fixture_no:
        blocking.extend(fixture_no)
        reasons.append(f"fixtures outstanding: {', '.join(sorted(fixture_no))}")
        conf = CONFIDENCE_CERTAIN if not fixture_unknown else CONFIDENCE_MIXED
        if fixture_unknown:
            reasons.append(f"unassessed: {', '.join(sorted(fixture_unknown))}")
        return PhaseVerdict(
            phase=Phase.FIXTURES_CARPENTRY,
            confidence=conf,
            reasons=tuple(reasons),
            blocking_signals=tuple(sorted(blocking)),
            unknown_signals=tuple(sorted(unknown)),
        )

    if fixture_partial:
        blocking.extend(fixture_partial)
        reasons.append(
            f"fixtures partially installed: {', '.join(sorted(fixture_partial))} "
            "-- partial work blocks progression"
        )
        return PhaseVerdict(
            phase=Phase.FIXTURES_CARPENTRY,
            confidence=CONFIDENCE_PARTIAL,
            reasons=tuple(reasons),
            blocking_signals=tuple(sorted(blocking)),
            unknown_signals=tuple(sorted(unknown)),
        )

    if fixture_unknown:
        reasons.append(
            f"cannot assess fixtures: {', '.join(sorted(fixture_unknown))} "
            "-- holding at fixtures phase, low confidence"
        )
        return PhaseVerdict(
            phase=Phase.FIXTURES_CARPENTRY,
            confidence=CONFIDENCE_UNKNOWN,
            reasons=tuple(reasons),
            blocking_signals=(),
            unknown_signals=tuple(sorted(unknown)),
        )

    reasons.append("all fixtures installed")

    # --- Phase 3: styling and restructure -------------------------------
    furniture = _tri(state.furniture_present)
    if furniture is Tri.UNKNOWN:
        reasons.append("furniture state unknown but shell is complete")
        return PhaseVerdict(
            phase=Phase.STYLING_RESTRUCTURE,
            confidence=CONFIDENCE_MIXED,
            reasons=tuple(reasons),
            blocking_signals=(),
            unknown_signals=tuple(sorted(unknown)),
        )

    reasons.append("shell complete -- room is ready for styling and restructure")
    return PhaseVerdict(
        phase=Phase.STYLING_RESTRUCTURE,
        confidence=CONFIDENCE_CERTAIN,
        reasons=tuple(reasons),
        blocking_signals=(),
        unknown_signals=tuple(sorted(unknown)),
    )


def can_progress(state: SurfaceState) -> tuple[bool, tuple[str, ...]]:
    """Whether the room may advance past its current phase, and what blocks it."""
    verdict = classify(state)
    if verdict.phase is Phase.STYLING_RESTRUCTURE:
        return (True, ())
    return (not verdict.blocking_signals, verdict.blocking_signals)


def restart_from_empty() -> SurfaceState:
    """A first-class transition: strip the room back to bare shell.

    This is not a reset button on a form. "Gut it and start over" is a real
    decision a client makes mid-project, and it has to produce a scene state
    the rest of the pipeline treats as legitimate -- a bare room in
    SURFACE_FINISHING with everything known to be absent, not a room full of
    UNKNOWNs that routes to human review.
    """
    return SurfaceState(
        walls_painted=Tri.NO.value,
        flooring_installed=Tri.NO.value,
        ceiling_finished=Tri.NO.value,
        electrical_terminated=Tri.NO.value,
        plumbing_terminated=Tri.NO.value,
        carpentry_installed=Tri.NO.value,
        furniture_present=Tri.NO.value,
    )
