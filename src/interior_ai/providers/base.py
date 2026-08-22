"""Provider protocol -- the seam between the pipeline and any vision model.

Everything downstream of perception (fit, phase, solver, pricing) is
deterministic geometry and rules. This protocol is the only place a model is
allowed to inject uncertainty, and it must express that uncertainty in the
vocabulary the rules already speak: :class:`~interior_ai.core.enums.Tri`
signals, never free text.

A provider that returned "the walls look mostly done" would push interpretation
into the phase rules, which is exactly the coupling the rules table exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.enums import ExecutionPath, Tri
from ..core.scene import SurfaceState


@dataclass(frozen=True)
class PerceptionResult:
    """What a provider saw, plus how sure it is and who said so."""

    surfaces: SurfaceState
    confidence: float
    path: ExecutionPath
    provider: str
    raw: dict | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RenderRequest:
    prompt: str
    room_id: str
    style: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class RenderResult:
    image_ref: str
    path: ExecutionPath
    provider: str
    seed: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class PerceptionProvider(Protocol):
    """Reads an image, returns Tri-valued surface signals."""

    name: str
    path: ExecutionPath

    def analyse(self, image_ref: str, *, room_id: str | None = None) -> PerceptionResult:
        ...


@runtime_checkable
class RenderProvider(Protocol):
    """Produces a visualisation. A render is a *view* of the scene; it never
    feeds back into scene state."""

    name: str
    path: ExecutionPath

    def render(self, req: RenderRequest) -> RenderResult:
        ...


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way the orchestrator should handle by
    falling back rather than propagating.

    Carries the HTTP status where there was one, and whether the failure looks
    transient. A 503 "high demand" and a 400 "bad request" both surface here,
    but only one is worth retrying -- collapsing that distinction is how a
    momentary overload turns into a hundred products stored without images.
    """

    #: HTTP statuses that mean "try again", not "you asked wrongly".
    RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        if retryable is None:
            retryable = (
                status_code in self.RETRYABLE_STATUSES
                if status_code is not None
                else False
            )
        self.retryable = retryable