"""Capability probe -- decides where inference actually runs.

The rule this module exists to enforce: **detect capabilities, not hardware.**

A box with a 48 GB A6000 and no model weights on disk cannot run locally. It
must fall through to CLOUD_API. Probing `nvidia-smi`, seeing plenty of VRAM,
and routing to LOCAL_FULL is the failure mode this design is built to prevent
-- it produces a confident router and a stack trace thirty seconds later, at
which point the request has already been accepted.

So each path declares everything it needs, and a path is only eligible when
every requirement is satisfied *right now*:

  LOCAL_FULL   GPU with enough VRAM  AND  full weights present
  LOCAL_LIGHT  weights present (CPU-tolerable)      -- no GPU needed
  CLOUD_API    API key present AND health ping OK
  MOCK         always eligible; the floor

Results are cached for 5 minutes. Probing shells out and hits the network, so
doing it per-request would add latency to every call to answer a question whose
answer changes on the order of deploys, not seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..core.enums import ExecutionPath

CACHE_TTL_SECONDS = 300

# Substring match rather than exact filename: weights ship as
# `sdxl-turbo-fp16.safetensors`, `sdxl_turbo.q4.gguf`, and a dozen other names
# depending on who quantised them. Requiring an exact name means a working
# install gets ignored because someone renamed a file.
FULL_WEIGHT_MARKERS = ("sdxl", "flux", "stable-diffusion")
LIGHT_WEIGHT_MARKERS = ("sd15", "sd-1.5", "tiny-sd", "lcm")

MIN_VRAM_MB_FULL = 8000


@dataclass(frozen=True)
class GpuInfo:
    present: bool
    name: str | None = None
    vram_mb: int | None = None
    torch_cuda: bool = False


@dataclass(frozen=True)
class Capabilities:
    """Everything the router needs, plus why it decided that way.

    ``reasons`` is not decoration. When a deployment routes to MOCK in
    production the first question is always "what did it think was missing",
    and without recorded reasons that question needs a debugger on a live box.
    """

    path: ExecutionPath
    gpu: GpuInfo
    full_weights: bool
    light_weights: bool
    api_key_present: bool
    api_healthy: bool
    forced: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)
    probed_at: float = field(default_factory=time.time)

    def explain(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no reasons recorded"


def _detect_gpu() -> GpuInfo:
    """Look for a usable GPU via nvidia-smi, then confirm through torch.

    nvidia-smi alone is not sufficient: a driver can be present while the
    CUDA runtime torch needs is missing or mismatched, and that combination
    reports a healthy GPU that torch refuses to use.
    """
    name: str | None = None
    vram: int | None = None
    present = False

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    vram = int(float(parts[1]))
                    present = True
        except (subprocess.SubprocessError, ValueError, OSError):
            present = False

    torch_cuda = False
    try:  # pragma: no cover - torch absent in test env
        import torch  # type: ignore

        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda and not present:
            present = True
            if name is None:
                name = torch.cuda.get_device_name(0)
    except Exception:
        torch_cuda = False

    return GpuInfo(present=present, name=name, vram_mb=vram, torch_cuda=torch_cuda)


def _scan_weights(model_dir: Path) -> tuple[bool, bool]:
    """Return (has_full_weights, has_light_weights) by filename substring."""
    if not model_dir.exists() or not model_dir.is_dir():
        return (False, False)

    names = []
    try:
        for p in model_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".safetensors", ".ckpt", ".gguf", ".bin", ".pt"):
                names.append(p.name.lower())
    except OSError:
        return (False, False)

    full = any(any(m in n for m in FULL_WEIGHT_MARKERS) for n in names)
    light = any(any(m in n for m in LIGHT_WEIGHT_MARKERS) for n in names)
    return (full, light)


def _default_health_check(api_key: str) -> bool:
    """Cheap liveness ping against the cloud provider.

    Skippable via PROBE_SKIP_HEALTHCHECK=1. On serverless, a cold start pays
    this latency on the first request of every container, which is exactly
    where the latency budget is tightest -- and the key being present is
    usually enough signal there.
    """
    if os.getenv("PROBE_SKIP_HEALTHCHECK") == "1":
        return True
    try:
        import httpx

        url = os.getenv("GEMINI_HEALTH_URL", "https://generativelanguage.googleapis.com/v1beta/models")
        resp = httpx.get(url, params={"key": api_key}, timeout=3.0)
        return resp.status_code < 500
    except Exception:
        return False


class CapabilityProbe:
    """Caching capability detector."""

    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        health_check: Callable[[str], bool] | None = None,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        gpu_detector: Callable[[], GpuInfo] | None = None,
    ) -> None:
        self._model_dir = Path(model_dir or os.getenv("MODEL_DIR", "./models"))
        self._health_check = health_check or _default_health_check
        self._ttl = ttl_seconds
        self._gpu_detector = gpu_detector or _detect_gpu
        self._cached: Capabilities | None = None

    def invalidate(self) -> None:
        self._cached = None

    def detect(self, *, force_refresh: bool = False) -> Capabilities:
        if not force_refresh and self._cached is not None:
            if time.time() - self._cached.probed_at < self._ttl:
                return self._cached

        caps = self._probe()
        self._cached = caps
        return caps

    def _probe(self) -> Capabilities:
        reasons: list[str] = []

        override = os.getenv("FORCE_EXECUTION_PATH")
        gpu = self._gpu_detector()
        full_w, light_w = _scan_weights(self._model_dir)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY") or ""
        api_key_present = bool(api_key.strip())
        api_healthy = False
        if api_key_present:
            api_healthy = bool(self._health_check(api_key))
            reasons.append(f"api key present, health={'ok' if api_healthy else 'failed'}")
        else:
            reasons.append("no api key")

        if override:
            try:
                forced_path = ExecutionPath(override.strip().upper())
            except ValueError:
                reasons.append(f"FORCE_EXECUTION_PATH={override!r} invalid, ignored")
            else:
                reasons.insert(0, f"forced to {forced_path.value} by FORCE_EXECUTION_PATH")
                return Capabilities(
                    path=forced_path,
                    gpu=gpu,
                    full_weights=full_w,
                    light_weights=light_w,
                    api_key_present=api_key_present,
                    api_healthy=api_healthy,
                    forced=True,
                    reasons=tuple(reasons),
                )

        if gpu.present:
            reasons.append(
                f"gpu={gpu.name or 'unknown'} vram={gpu.vram_mb or '?'}MB torch_cuda={gpu.torch_cuda}"
            )
        else:
            reasons.append("no gpu detected")

        reasons.append(
            f"weights in {self._model_dir}: full={full_w} light={light_w}"
        )

        enough_vram = gpu.present and (gpu.vram_mb or 0) >= MIN_VRAM_MB_FULL
        if enough_vram and full_w:
            reasons.append("-> LOCAL_FULL (gpu + full weights)")
            path = ExecutionPath.LOCAL_FULL
        elif light_w:
            reasons.append("-> LOCAL_LIGHT (light weights present)")
            path = ExecutionPath.LOCAL_LIGHT
        elif api_key_present and api_healthy:
            reasons.append("-> CLOUD_API (no usable local weights, cloud reachable)")
            path = ExecutionPath.CLOUD_API
        else:
            if gpu.present and not (full_w or light_w):
                reasons.append("gpu present but NO weights on disk -- cannot run locally")
            reasons.append("-> MOCK (nothing else eligible)")
            path = ExecutionPath.MOCK

        return Capabilities(
            path=path,
            gpu=gpu,
            full_weights=full_w,
            light_weights=light_w,
            api_key_present=api_key_present,
            api_healthy=api_healthy,
            forced=False,
            reasons=tuple(reasons),
        )


_default_probe: CapabilityProbe | None = None


def get_probe() -> CapabilityProbe:
    global _default_probe
    if _default_probe is None:
        _default_probe = CapabilityProbe()
    return _default_probe


def reset_probe() -> None:
    """Test hook -- drops the module-level singleton."""
    global _default_probe
    _default_probe = None
