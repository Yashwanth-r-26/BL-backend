"""Mock provider -- deterministic, offline, and honest about being a mock.

This is the floor the capability probe falls back to, so it has to be genuinely
usable: the whole pipeline runs against it in CI, in local development without
a GPU, and in tests. That means it cannot return random noise.

Determinism comes from hashing the image reference. The same input always
yields the same signals, so a test asserting "this room routes to
FIXTURES_CARPENTRY" stays true across runs and machines.

It also deliberately emits PARTIAL and UNKNOWN for some inputs. A mock that
always returns clean YES/NO would let the PARTIAL-blocks-progression path go
untested, which is the exact path most likely to be wrong in production.
"""

from __future__ import annotations

import hashlib

from ..core.enums import ExecutionPath, Tri
from ..core.scene import SurfaceState
from .base import PerceptionResult, RenderRequest, RenderResult

_SIGNALS = (
    "walls_painted",
    "flooring_installed",
    "ceiling_finished",
    "electrical_terminated",
    "plumbing_terminated",
    "carpentry_installed",
    "furniture_present",
)

# Weighted so PARTIAL and UNKNOWN appear often enough to exercise those paths.
_TRI_CYCLE = (Tri.YES, Tri.NO, Tri.PARTIAL, Tri.YES, Tri.UNKNOWN, Tri.NO, Tri.PARTIAL, Tri.YES)


def _digest(ref: str) -> bytes:
    return hashlib.sha256(ref.encode("utf-8")).digest()


class MockPerceptionProvider:
    """Hash-seeded surface classifier."""

    name = "mock-perception"
    path = ExecutionPath.MOCK

    def __init__(self, *, forced: SurfaceState | None = None) -> None:
        # ``forced`` lets a test pin exact signals without caring about hashing.
        self._forced = forced

    def analyse(self, image_ref: str, *, room_id: str | None = None) -> PerceptionResult:
        if self._forced is not None:
            return PerceptionResult(
                surfaces=self._forced,
                confidence=1.0,
                path=self.path,
                provider=self.name,
                notes=("forced surface state -- test fixture",),
            )

        d = _digest(image_ref)
        values = {}
        for i, sig in enumerate(_SIGNALS):
            values[sig] = _TRI_CYCLE[d[i] % len(_TRI_CYCLE)].value

        return PerceptionResult(
            surfaces=SurfaceState(**values),
            confidence=0.5,
            path=self.path,
            provider=self.name,
            raw={"digest": d.hex()[:16]},
            notes=(
                "MOCK perception -- signals derived from a hash of the image "
                "reference, not from the image. Not suitable for quoting.",
            ),
        )


class MockRenderProvider:
    """Returns a stable synthetic image reference."""

    name = "mock-render"
    path = ExecutionPath.MOCK

    def render(self, req: RenderRequest) -> RenderResult:
        seed = req.seed if req.seed is not None else int.from_bytes(
            _digest(req.prompt)[:4], "big"
        )
        ref = f"mock://render/{req.room_id}/{seed}"
        return RenderResult(
            image_ref=ref,
            path=self.path,
            provider=self.name,
            seed=seed,
            notes=("MOCK render -- no image was generated.",),
        )
