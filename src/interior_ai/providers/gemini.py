"""Gemini cloud provider.

The job here is narrow: get a vision model to answer seven questions in the
Tri vocabulary, and refuse to let anything else through.

The parsing is strict on purpose. A model asked for JSON will occasionally
return prose, fenced code, or JSON with an unexpected extra key, and the
tempting response is to fuzzy-match it into shape. That is how a hedge like
"mostly painted" silently becomes YES, which then quotes a half-painted room as
finished. Anything this parser does not recognise becomes UNKNOWN, which the
phase rules already handle correctly by dropping confidence and routing to
review.

Network failure is likewise not an emergency -- it raises
:class:`~interior_ai.providers.base.ProviderError`, and the orchestrator falls
back to MOCK rather than failing the request.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..core.enums import ExecutionPath, Tri
from ..core.scene import SurfaceState
from .base import (
    PerceptionResult,
    ProviderError,
    RenderRequest,
    RenderResult,
)

# Current GA text model as of mid-2026. Google retires these on a schedule
# (gemini-2.0-flash was shut down June 2026), so this WILL go stale -- it is an
# env-overridable default (GEMINI_MODEL), not a hard dependency. For a fixed
# 7-question image classification, flash-lite is the right cost/latency point.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

_SIGNALS = (
    "walls_painted",
    "flooring_installed",
    "ceiling_finished",
    "electrical_terminated",
    "plumbing_terminated",
    "carpentry_installed",
    "furniture_present",
)

PROMPT = """You are assessing the construction state of a single room from a photograph.

For each of the following, answer with exactly one of: yes, no, partial, unknown.

- walls_painted: are the walls fully painted and finished?
- flooring_installed: is the finished floor laid (not bare screed or subfloor)?
- ceiling_finished: is the ceiling finished and painted?
- electrical_terminated: are switches, sockets and fittings terminated (not bare wires)?
- plumbing_terminated: are taps, outlets and sanitary fittings terminated?
- carpentry_installed: is built-in carpentry (wardrobes, cabinets) installed?
- furniture_present: is loose furniture present in the room?

Use "partial" when work is visibly begun but incomplete -- for example, one wall
painted and three bare. Use "unknown" when the photograph does not show enough
to judge. Do not guess.

Respond with ONLY a JSON object with those seven keys and no other text."""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Handles bare JSON and fenced blocks. Everything else raises, and the caller
    turns that into all-UNKNOWN rather than a guess.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace:
            text = brace.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"response was not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("response JSON was not an object")
    return parsed


def _coerce_tri(value: Any) -> Tri:
    """Map a model's answer onto Tri, defaulting to UNKNOWN.

    Only exact vocabulary matches are accepted. "mostly", "looks done", and
    "probably" all become UNKNOWN, because a hedge is not a yes and treating it
    as one is how a half-finished room gets quoted as complete.
    """
    if isinstance(value, bool):
        return Tri.YES if value else Tri.NO
    if not isinstance(value, str):
        return Tri.UNKNOWN
    v = value.strip().lower()
    if v in ("yes", "true", "complete", "completed", "done"):
        return Tri.YES
    if v in ("no", "false", "none", "not started", "absent"):
        return Tri.NO
    if v == "partial":
        return Tri.PARTIAL
    return Tri.UNKNOWN


def parse_surface_response(text: str) -> tuple[SurfaceState, list[str]]:
    """Parse a model reply into a SurfaceState plus any notes about what
    could not be read."""
    notes: list[str] = []
    try:
        raw = _extract_json(text)
    except ProviderError as exc:
        return (
            SurfaceState(),
            [f"could not parse model response ({exc}); all signals set to unknown"],
        )

    values: dict[str, str] = {}
    for sig in _SIGNALS:
        if sig not in raw:
            notes.append(f"model omitted {sig}; treated as unknown")
            values[sig] = Tri.UNKNOWN.value
            continue
        tri = _coerce_tri(raw[sig])
        if tri is Tri.UNKNOWN and str(raw[sig]).strip().lower() not in ("unknown", ""):
            notes.append(
                f"model answered {sig}={raw[sig]!r}, which is not a recognised "
                "value; treated as unknown"
            )
        values[sig] = tri.value

    extra = set(raw) - set(_SIGNALS)
    if extra:
        notes.append(f"ignored unexpected keys: {', '.join(sorted(extra))}")

    return SurfaceState(**values), notes


class GeminiPerceptionProvider:
    """Vision perception backed by the Gemini API."""

    name = "gemini-perception"
    path = ExecutionPath.CLOUD_API

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY") or ""
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or os.getenv("GEMINI_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout_s = timeout_s
        # ``transport`` is an injection seam: tests pass a callable returning a
        # canned payload so the parsing logic is testable without a network.
        self._transport = transport

    def _post(self, payload: dict) -> dict:
        if self._transport is not None:
            return self._transport(payload)

        if not self.api_key:
            raise ProviderError("no Gemini API key configured")
        try:
            import httpx

            url = f"{self.endpoint}/{self.model}:generateContent"
            resp = httpx.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_s,
            )
            if resp.status_code >= 400:
                raise ProviderError(
                    f"Gemini returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            return resp.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc

    @staticmethod
    def _text_from_response(data: dict) -> str:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected Gemini response shape: {exc}") from exc

    def analyse(self, image_ref: str, *, room_id: str | None = None) -> PerceptionResult:
        parts: list[dict[str, Any]] = [{"text": PROMPT}]

        # image_ref may be an inline base64 payload or a plain reference. Only
        # the former can actually be sent; a bare reference degrades to a
        # text-only query, which will honestly answer "unknown" for everything.
        if image_ref.startswith("data:"):
            header, _, b64 = image_ref.partition(",")
            mime = header.split(";")[0].removeprefix("data:") or "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        else:
            parts.append({"text": f"(image reference: {image_ref})"})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }

        data = self._post(payload)
        text = self._text_from_response(data)
        surfaces, notes = parse_surface_response(text)

        unknown_count = sum(
            1 for s in _SIGNALS if getattr(surfaces, s) == Tri.UNKNOWN.value
        )
        confidence = max(0.1, 1.0 - (unknown_count / len(_SIGNALS)))

        return PerceptionResult(
            surfaces=surfaces,
            confidence=confidence,
            path=self.path,
            provider=self.name,
            raw={"model": self.model},
            notes=tuple(notes),
        )

    def classify_room(self, image_ref: str) -> str:
        """Ask the model for room type + coarse size class only.

        Returns the raw model text; :func:`interior_ai.perception.estimator.
        parse_classification` turns it into a structured result. Kept as a thin
        method here so the estimator owns all the parsing and never depends on
        the transport, which keeps it unit-testable without a network.

        Deliberately separate from ``analyse``: construction state and room
        category are different questions with different prompts, and coupling
        them into one call would make a parse failure in one poison the other.
        """
        from ..perception.estimator import CLASSIFY_PROMPT

        parts: list[dict[str, Any]] = [{"text": CLASSIFY_PROMPT}]
        if image_ref.startswith("data:"):
            header, _, b64 = image_ref.partition(",")
            mime = header.split(";")[0].removeprefix("data:") or "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        else:
            parts.append({"text": f"(image reference: {image_ref})"})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        data = self._post(payload)
        return self._text_from_response(data)


class GeminiRenderProvider:
    """Image generation via Gemini.

    Kept separate from perception because the two have different failure modes
    and different models -- and because a render is a view, so a failed render
    must never block a quote.
    """

    name = "gemini-render"
    path = ExecutionPath.CLOUD_API

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 60.0,
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY") or ""
        self.model = model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
        self.endpoint = endpoint or os.getenv("GEMINI_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout_s = timeout_s
        self._transport = transport

    def render(self, req: RenderRequest) -> RenderResult:
        prompt = req.prompt
        if req.style:
            prompt = f"{prompt}\n\nStyle: {req.style}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }

        if self._transport is not None:
            data = self._transport(payload)
        else:
            if not self.api_key:
                raise ProviderError("no Gemini API key configured")
            try:
                import httpx

                url = f"{self.endpoint}/{self.model}:generateContent"
                resp = httpx.post(
                    url, params={"key": self.api_key}, json=payload, timeout=self.timeout_s
                )
                if resp.status_code >= 400:
                    raise ProviderError(
                        f"Gemini returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"Gemini render failed: {exc}") from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected Gemini response shape: {exc}") from exc

        for part in parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                return RenderResult(
                    image_ref=f"data:{mime};base64,{inline['data']}",
                    path=self.path,
                    provider=self.name,
                    seed=req.seed,
                )

        raise ProviderError("Gemini response contained no image data")