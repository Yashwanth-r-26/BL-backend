"""Quotation from before/after images plus everything we already know.

Structure adapted from the earlier Foundry quotation service: one master
prompt, three costed options (contractor / DIY / hybrid), a contractor list,
and a JSON parser that repairs a truncated response rather than losing the
whole estimate to a missing brace.

What is different here, and it is the point of this module: **we already know
some of the prices.** A swapped-in product has a SKU, a catalogue price and a
vendor. Those are facts. Labour rates, paint, tiling, a false ceiling and
anything a typed instruction implies are not -- nothing in this system knows
what a painter charges in Mysuru.

So the prompt draws that line explicitly. Known prices are passed as fixed
anchors the model must reproduce verbatim; everything else it estimates for
the region, and every line it returns says which kind it is. The alternative --
letting the model re-price a sofa whose price we hold -- produces a quote that
disagrees with the picker the customer just used, and there is no good way to
explain that to them.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

MASTER_PROMPT = """You are an experienced renovation cost estimator and interior
project planner working in India. You combine a quantity surveyor's discipline,
a contractor's knowledge of site realities, and a DIY educator's clarity. You
are precise, honest, and regionally aware.

YOU ARE GIVEN
- TWO photographs: the FIRST is the room BEFORE, the SECOND is the room AFTER
  the changes the owner has chosen.
- The city and its market tier, and today's date.
- KNOWN ITEMS: products already selected from a catalogue, with real prices.
- REQUESTED CHANGES: work the owner asked for in their own words.
- SCOPE AND PREFERENCES from a short questionnaire.

HOW TO PRICE

1. Compare the two photographs and identify every visible change: walls,
   flooring, ceiling, furniture, cabinetry, fixtures, lighting, decor.

2. KNOWN ITEMS ARE NOT YOURS TO ESTIMATE. Each one comes with a real price the
   customer has already been shown. Use that exact figure. Do not adjust it for
   the region, do not round it, do not substitute a similar product. Mark every
   such line "known". Getting this wrong means the quote contradicts the price
   the customer saw a minute ago.

3. Everything else you estimate: materials, labour, and the trades involved.
   Use realistic {city} rates for {tier_description} as at {date}, in INR. Not
   aspirational, not padded -- what a homeowner there would actually be quoted.
   Mark these lines "estimated".

4. Quantities follow from the room. Estimate its dimensions from the
   photographs if they are not given, and say what you assumed.

PRODUCE THREE OPTIONS

A) CONTRACTOR -- a professional does everything.
   Materials and labour separated, a line-item breakdown, and labour broken
   down by trade.

B) DIY -- the owner does the work.
   Materials only, at retail prices (a homeowner pays more per unit than a
   contractor). A numbered, beginner-usable guide with tools, order of work and
   realistic timings. Then an honest split: what is genuinely DIY-able, and what
   needs a professional -- electrical work, plumbing, false ceilings, anything
   structural -- and why. Do not encourage someone to do unsafe work cheaply.

C) HYBRID -- the sensible middle.
   Which tasks are worth doing yourself and which to hire out, with the
   reasoning, and a total for that mix.

Then suggest five contractors who could do this work in {city}: plausible
business name, specialty, quote range consistent with your contractor total,
rating, years of experience, and one line on why they price as they do.

OUTPUT
Respond with ONLY valid JSON, no markdown or commentary, using exactly this
schema:

{{
  "currency": "INR",
  "currency_symbol": "\\u20b9",
  "room_summary": "string -- the room and your dimension assumption",
  "assumptions": ["string -- anything you had to assume"],
  "changes_detected": [
    {{"category": "walls|flooring|ceiling|furniture|cabinets|fixtures|decor|other",
      "description": "string", "confidence": "high|medium|low"}}
  ],
  "contractor": {{
    "total": number, "materials_total": number, "labor_total": number,
    "line_items": [
      {{"name": "string", "description": "string", "quantity": number,
        "unit": "string", "unit_rate": number, "total": number,
        "category": "string", "type": "material|labor",
        "pricing": "known|estimated"}}
    ],
    "timeline_weeks": number, "notes": "string"
  }},
  "diy": {{
    "materials_total": number, "labor_total": 0,
    "line_items": [
      {{"name": "string", "description": "string", "quantity": number,
        "unit": "string", "unit_rate": number, "total": number,
        "category": "string", "pricing": "known|estimated"}}
    ],
    "steps": [
      {{"step": number, "title": "string", "instructions": "string",
        "tools_needed": ["string"], "estimated_time": "string",
        "difficulty": "easy|medium|hard"}}
    ],
    "diy_achievable": ["string"],
    "needs_professional": ["string -- the task, and why it needs a pro"],
    "notes": "string"
  }},
  "hybrid": {{
    "total": number,
    "diy_tasks": [{{"task": "string", "materials_cost": number, "reason": "string"}}],
    "hire_tasks": [{{"task": "string", "cost": number, "materials_cost": number,
                    "labor_cost": number, "reason": "string"}}],
    "plan_summary": "string", "notes": "string"
  }},
  "contractors": [
    {{"name": "string", "specialty": "string", "quote_min": number,
      "quote_max": number, "rating": number, "years_experience": number,
      "justification": "string"}}
  ]
}}

Return ONLY that JSON object, fully populated."""


TIER_DESCRIPTIONS = {
    "metro": "a major metro, where labour and materials cost more than the "
             "national average",
    "tier2": "a tier-2 city, above the national average but well below metro "
             "rates",
    "tier3": "a smaller city or town, where labour is cheaper and premium "
             "materials often have to be brought in",
}


def repair_json(text: str) -> dict | None:
    """Parse a JSON reply, closing it if the model was cut off mid-answer.

    A long quotation can hit the output limit part-way through a string or an
    array. Losing an otherwise complete estimate to one missing brace is a poor
    trade, so an unterminated response is closed and re-parsed. Anything still
    unreadable returns None -- a half-invented quote would be worse than none.
    """
    if not text:
        return None
    raw = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    try:
        if raw.count('"') % 2 == 1:
            raw += '"'
        stack: list[str] = []
        in_string = False
        escaped = False
        for ch in raw:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
        raw = raw.rstrip().rstrip(",")
        raw += "".join("}" if c == "{" else "]" for c in reversed(stack))
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _downscale(data_uri_or_b64: str, max_dim: int = 1024) -> str:
    """Shrink an image for the prompt; two full-size photos time the call out."""
    b64 = data_uri_or_b64.partition(",")[2] if data_uri_or_b64.startswith("data:") \
        else data_uri_or_b64
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        return base64.b64encode(out.getvalue()).decode()
    except Exception:
        return b64


def build_context(
    *, location: dict, questionnaire: dict, manifest: dict, date_str: str,
) -> str:
    """The factual block that accompanies the two photographs."""
    known = manifest.get("known_products") or []
    instructions = manifest.get("instructions") or []

    lines = [
        "CONTEXT",
        f"- City: {location.get('city') or 'unknown'}, "
        f"{location.get('country_name') or 'India'}",
        f"- Market tier: {location.get('city_tier', 'tier3')}",
        f"- Currency: {location.get('currency', 'INR')}",
        f"- Date: {date_str}",
    ]

    if questionnaire:
        lines.append("")
        lines.append("SCOPE AND PREFERENCES")
        for key, value in questionnaire.items():
            if value in (None, "", [], {}):
                continue
            shown = ", ".join(value) if isinstance(value, list) else value
            lines.append(f"- {key.replace('_', ' ')}: {shown}")

    lines.append("")
    if known:
        lines.append("KNOWN ITEMS -- real prices, use these figures exactly:")
        for item in known:
            lines.append(
                f"- {item['name']} ({item['object_class']}), "
                f"{item['width_mm']}x{item['depth_mm']}x{item['height_mm']} mm, "
                f"{item['currency']} {item['price']} from {item['vendor']}. "
                f"Replaces the {item['replaced']}. "
                f"{item['description']}".rstrip()
            )
    else:
        lines.append("KNOWN ITEMS: none -- every price in this quote is your "
                     "estimate.")

    lines.append("")
    if instructions:
        lines.append("REQUESTED CHANGES -- the owner's own words. Nothing is "
                     "known about their cost; estimate materials and labour:")
        for entry in instructions:
            lines.append(f"- \"{entry['instruction']}\" ({entry['applied_to']})")
    else:
        lines.append("REQUESTED CHANGES: none beyond the products listed above.")

    lines.append("")
    lines.append("The FIRST image is BEFORE, the SECOND is AFTER. Compare them "
                 "and produce the quotation JSON.")
    return "\n".join(lines)


class GeminiQuoter:
    """Quotation via a Gemini text model that accepts images."""

    #: Fast first, then steadier. Image generation and estimation are different
    #: workloads, so this chain is separate from the image model chain.
    FALLBACKS = ("gemini-3.5-flash", "gemini-flash-latest")

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 transport: Any | None = None, timeout_s: float | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY") or ""
        self.model = model or os.getenv("GEMINI_QUOTE_MODEL") \
            or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.endpoint = os.getenv(
            "GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"
        )
        self.timeout_s = timeout_s or float(os.getenv("GEMINI_QUOTE_TIMEOUT_S", "180"))
        self._transport = transport

    @property
    def model_chain(self) -> list[str]:
        chain = [self.model, *self.FALLBACKS]
        seen: set[str] = set()
        return [m for m in chain if not (m in seen or seen.add(m))]

    def quote(
        self, before_b64: str, after_b64: str, *, location: dict,
        questionnaire: dict, manifest: dict, date_str: str,
    ) -> dict:
        """Returns {status, provider, data, notes}. Never raises for a model
        failure -- a quote that could not be produced is reported, not thrown,
        because everything else about the session is still valid."""
        from ..providers.base import ProviderError

        tier = location.get("city_tier", "tier3")
        prompt = MASTER_PROMPT.format(
            city=location.get("city") or "this city",
            tier_description=TIER_DESCRIPTIONS.get(tier, TIER_DESCRIPTIONS["tier3"]),
            date=date_str,
        )
        context = build_context(
            location=location, questionnaire=questionnaire,
            manifest=manifest, date_str=date_str,
        )
        parts = [
            {"text": prompt},
            {"text": context},
            {"text": "BEFORE image:"},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _downscale(before_b64)}},
            {"text": "AFTER image:"},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": _downscale(after_b64)}},
        ]
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
                "maxOutputTokens": 32768,
            },
        }

        last_error = "no attempt made"
        for model in self.model_chain:
            for attempt in range(2):
                try:
                    data = self._post(model, payload)
                except ProviderError as exc:
                    last_error = str(exc)
                    if getattr(exc, "retryable", False):
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break
                text = "".join(
                    part.get("text", "")
                    for candidate in data.get("candidates", [])
                    for part in candidate.get("content", {}).get("parts", [])
                )
                parsed = repair_json(text)
                if parsed is not None:
                    return {"status": "ok", "provider": model, "data": parsed,
                            "notes": []}
                last_error = "response was not usable JSON"
                break

        return {"status": "error", "provider": None, "data": None,
                "notes": [f"quotation failed: {last_error}"]}

    def _post(self, model: str, payload: dict) -> dict:
        from ..providers.base import ProviderError

        if self._transport is not None:
            return self._transport(model, payload)
        if not self.api_key:
            raise ProviderError("no Gemini API key configured")
        try:
            import httpx

            resp = httpx.post(
                f"{self.endpoint}/{model}:generateContent",
                params={"key": self.api_key},
                json=payload,
                timeout=httpx.Timeout(connect=15.0, read=self.timeout_s,
                                      write=60.0, pool=15.0),
            )
            if resp.status_code >= 400:
                raise ProviderError(
                    f"Gemini returned HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )
            return resp.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"quotation request failed: {exc}",
                                retryable=True) from exc


class MockQuoter:
    """Offline stand-in. Produces a clearly-labelled shell so the whole flow
    can be exercised without a key -- and so nobody mistakes it for a quote."""

    def quote(self, before_b64, after_b64, *, location, questionnaire,
              manifest, date_str):
        known = manifest.get("known_products") or []
        known_total = sum(float(k["price"]) for k in known)
        return {
            "status": "mock",
            "provider": "MockQuoter",
            "notes": ["No language model on this path -- only the known "
                      "catalogue prices are real; nothing has been estimated."],
            "data": {
                "currency": location.get("currency", "INR"),
                "currency_symbol": location.get("currency_symbol", "\u20b9"),
                "room_summary": "MOCK quotation -- no model available.",
                "assumptions": ["Generated offline; no estimation performed."],
                "changes_detected": [],
                "contractor": {
                    "total": known_total, "materials_total": known_total,
                    "labor_total": 0.0,
                    "line_items": [
                        {"name": k["name"], "description": k["description"],
                         "quantity": 1, "unit": "piece",
                         "unit_rate": float(k["price"]),
                         "total": float(k["price"]),
                         "category": k["object_class"], "type": "material",
                         "pricing": "known"}
                        for k in known
                    ],
                    "timeline_weeks": 0, "notes": "mock",
                },
                "diy": {"materials_total": known_total, "labor_total": 0,
                        "line_items": [], "steps": [], "diy_achievable": [],
                        "needs_professional": [], "notes": "mock"},
                "hybrid": {"total": known_total, "diy_tasks": [],
                           "hire_tasks": [], "plan_summary": "mock",
                           "notes": "mock"},
                "contractors": [],
            },
        }