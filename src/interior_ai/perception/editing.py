"""Interactive photo editing -- detect, select by click, replace, iterate.

The flow this module powers:

1. **Detect** every furnishable object in the photo once, up front. Gemini
   returns labelled bounding boxes in normalised coordinates. Detection runs
   once per session because it is the expensive call and the objects do not
   move between edits of *other* objects.
2. **Select by click.** The user taps a pixel; the hit-test maps it to the
   detection whose box contains it (smallest box wins, so clicking a lamp
   standing in front of a wall selects the lamp, not the wall).
3. **Replace.** The selected region plus a catalogue product's reference image
   go to the image-edit model with a preserve-first prompt: change only this
   object, keep everything else -- lighting, shadows, the rest of the room --
   exactly as it is.
4. **Iterate.** Every replacement is an append-only step; the user loops until
   satisfied, then the final image's step chain names every sku that was
   swapped in, which is what the quote prices.

Provider note: detection and editing both go through Gemini here (bounding-box
detection + Nano Banana semantic editing). SAM-family models produce sharper
pixel masks and remain the documented upgrade path -- the interfaces below
take a *region*, not a Gemini call, precisely so a SAM mask can replace the
box hit-test without touching anything downstream.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..providers.base import ProviderError

# Objects worth detecting in an interior photo -- things a store could sell a
# replacement for -- plus the three restylable surfaces. Surfaces are detected
# as regions so a click on bare wall/ceiling/floor can select them for
# repainting or refinishing, exactly like an object swap.
DETECT_CLASSES = (
    "sofa", "armchair", "coffee_table", "side_table", "dining_table",
    "tv_unit", "bed", "wardrobe", "bookshelf", "rug", "lamp", "curtain",
    "wall_art", "plant", "television", "ottoman", "ceiling_light", "mirror",
    "blinds", "room_divider", "cabinet", "chair", "cushion", "decor",
)

SURFACE_CLASSES = ("wall", "ceiling", "floor")

DETECT_PROMPT = """You are an exhaustive object detector for interior photographs.
Your detections drive a click-to-select tool, so ANY object a user might click
must have a box. Missing objects are failures; err on the side of including.

PASS 1 -- OBJECTS. Detect EVERY distinct visible furniture, fixture and decor
item. Typical furnished rooms contain 10-25 detectable items. Include:
- seating (sofa, armchair, chair, ottoman), tables (coffee_table, side_table,
  dining_table), storage (tv_unit, wardrobe, bookshelf, cabinet)
- electronics (television), lighting (lamp, ceiling_light -- including track
  and spot fittings), soft furnishing (rug, cushion, curtain, blinds)
- decor (wall_art, mirror, plant, decor for vases/sculptures/objects),
  room_divider for screens and partition panels
- partially visible and background objects too; one entry per physical object
  (two side tables = two entries)

PASS 2 -- SURFACES. Also emit one entry per visible major surface REGION:
- "wall": each distinct visible wall area (a marble feature wall and a painted
  side wall are separate entries; describe each, e.g. "white marble feature
  wall", "beige painted wall")
- "ceiling": the visible ceiling
- "floor": the visible floor (describe the material, e.g. "herringbone wood
  floor")
Surface boxes should cover the visible extent of that surface, even where
objects sit in front of it.

For each entry output:
- "label": short specific description ("walnut slatted room divider")
- "object_class": one of: sofa, armchair, chair, coffee_table, side_table,
  dining_table, tv_unit, bed, wardrobe, bookshelf, cabinet, rug, lamp,
  ceiling_light, curtain, blinds, wall_art, mirror, plant, decor, television,
  ottoman, cushion, room_divider, wall, ceiling, floor, other
- "box_2d": [y_min, x_min, y_max, x_max] in normalised 0-1000 coordinates
- "confidence": 0.0-1.0

Boxes tight for objects; full visible extent for surfaces. Do not merge
distinct objects. Respond with ONLY the JSON array, no other text."""


@dataclass(frozen=True)
class Detection:
    """One detected object, in normalised 0-1000 image coordinates."""

    id: str
    label: str
    object_class: str
    box: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), 0-1000
    confidence: float

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def contains(self, x: int, y: int, *, pad: int = 0) -> bool:
        """Whether a normalised point falls inside (or within ``pad`` of) the
        box. The pad is what makes clicking *near* the sofa still select it."""
        x0, y0, x1, y1 = self.box
        return (x0 - pad) <= x <= (x1 + pad) and (y0 - pad) <= y <= (y1 + pad)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "object_class": self.object_class,
            "box": list(self.box),
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(
            id=d["id"],
            label=d["label"],
            object_class=d["object_class"],
            box=tuple(d["box"]),  # type: ignore[arg-type]
            confidence=float(d.get("confidence", 0.5)),
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply, fenced or bare."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace:
            text = brace.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ProviderError("response was not a JSON object")
    return parsed


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        bracket = re.search(r"(\[.*\])", text, re.DOTALL)
        if bracket:
            text = bracket.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ProviderError("detection response was not a JSON array")
    return parsed


def parse_detections(text: str) -> tuple[list[Detection], list[str]]:
    """Parse a detection reply. Malformed entries are skipped with a note, not
    fatal -- one bad box should not blank the whole photo's detections."""
    notes: list[str] = []
    try:
        raw = _extract_json_array(text)
    except (json.JSONDecodeError, ProviderError) as exc:
        return [], [f"could not parse detections ({exc})"]

    out: list[Detection] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            notes.append(f"entry {i} was not an object; skipped")
            continue
        box = entry.get("box_2d") or entry.get("box")
        label = entry.get("label")
        if not (isinstance(box, list) and len(box) == 4 and label):
            notes.append(f"entry {i} missing box or label; skipped")
            continue
        try:
            # Gemini's convention is [y_min, x_min, y_max, x_max]; we store
            # (x_min, y_min, x_max, y_max). Clamp to 0-1000.
            y0, x0, y1, x1 = (max(0, min(1000, int(v))) for v in box)
        except (TypeError, ValueError):
            notes.append(f"entry {i} box not numeric; skipped")
            continue
        if x1 <= x0 or y1 <= y0:
            notes.append(f"entry {i} box degenerate; skipped")
            continue
        cls = str(entry.get("object_class", "other")).lower()
        if cls not in DETECT_CLASSES and cls not in SURFACE_CLASSES:
            cls = "other"
        out.append(
            Detection(
                id=uuid.uuid4().hex[:12],
                label=str(label),
                object_class=cls,
                box=(x0, y0, x1, y1),
                confidence=float(entry.get("confidence", 0.5)),
            )
        )
    return out, notes


INTENT_PROMPT = """You are interpreting an edit request for an interior
photograph, so it can be carried out precisely.

THE USER TYPED: "{text}"
{selection_line}
OBJECTS AND SURFACES DETECTED IN THE PHOTO (id -- label -- box as
[x_min, y_min, x_max, y_max] in normalised 0-1000 coordinates):
{catalogue}

Decide what the user wants changed, and answer with a JSON object:

- "target_ids": a list of the ids this applies to. Usually one. Empty if the
  request is about the whole scene (e.g. "make the room brighter") or you
  genuinely cannot tell.
- "operation": one of "recolour", "replace", "remove", "restyle", "scene",
  "unclear".
- "selection_matches": true if the request is about the object they selected;
  false if their words clearly refer to something else; null if nothing was
  selected.
- "instruction": one clear sentence describing the change, naming the object
  by its visible appearance rather than its id.
- "confidence": 0.0-1.0.
- "note": a short plain-language remark if something is worth flagging --
  ambiguity, a mismatch between what they clicked and what they wrote, or a
  request that cannot be done by editing this photo. Otherwise "".

Rules:
- HOW MANY: a request that names a surface or object type generally --
  "paint the walls", "paint wall sage green", "change the flooring" -- applies
  to EVERY matching region in the list, so return all of their ids. Only a
  request that points at one -- "this wall", "that shelf", "the marble wall" --
  or one made while an object was selected, applies to a single id. When in
  doubt about a bare plural or an unqualified surface name, include them all:
  painting one wall of four and leaving the rest is never what someone meant
  by "paint the wall".
- Prefer the selected object when the request is consistent with it.
- If their words plainly describe a different object than the one selected,
  set "selection_matches" to false and point "target_id" at what they
  actually described. A misplaced click is common; their words are the better
  evidence of intent.
- If the request names no object and describes a wall, floor or ceiling
  change, target the matching surface.
- Do not invent ids. Use only the ids listed above.

Respond with ONLY the JSON object."""


@dataclass(frozen=True)
class EditIntent:
    """What the user's words were understood to mean.

    ``target_ids`` is a list because "paint the wall" rarely means one wall.
    A room's walls arrive as several detected regions, and resolving such a
    request to a single id paints one panel and leaves the rest -- which is
    never what anybody meant.
    """

    target_ids: tuple[str, ...]
    operation: str
    instruction: str
    confidence: float
    selection_matches: bool | None = None
    note: str = ""

    @property
    def target_id(self) -> str | None:
        """First target, for the common single-object case."""
        return self.target_ids[0] if self.target_ids else None

    @property
    def is_actionable(self) -> bool:
        return self.operation != "unclear" and bool(self.instruction)


def parse_intent(text: str) -> EditIntent:
    """Read an intent reply, degrading to 'unclear' rather than guessing."""
    try:
        raw = _extract_json_object(text)
    except Exception:
        return EditIntent(
            target_ids=(), operation="unclear", instruction="",
            confidence=0.0, note="could not interpret the request",
        )

    operation = str(raw.get("operation", "unclear")).lower()
    if operation not in {"recolour", "recolor", "replace", "remove",
                         "restyle", "scene", "unclear"}:
        operation = "unclear"
    if operation == "recolor":
        operation = "recolour"

    matches = raw.get("selection_matches")
    if matches is not None:
        matches = bool(matches)

    # Accept either key: a single id is still the common answer, and models
    # drift between the two shapes.
    raw_targets = raw.get("target_ids")
    if raw_targets is None:
        single = raw.get("target_id")
        raw_targets = [single] if single else []
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    targets = tuple(str(v) for v in raw_targets if v) if isinstance(raw_targets, list) else ()

    return EditIntent(
        target_ids=targets,
        operation=operation,
        instruction=str(raw.get("instruction") or "").strip(),
        confidence=float(raw.get("confidence") or 0.0),
        selection_matches=matches,
        note=str(raw.get("note") or "").strip(),
    )


INSTRUCT_REGION_TEMPLATE = """Edit this interior photograph.

TASK: {instruction}

The change applies to the {label}, in the region roughly at
[x {x0}-{x1}, y {y0}-{y1}] on a 0-1000 scale.

RULES:
- Change ONLY what the task describes. Everything else -- every other object,
  every other surface, the framing, the lighting -- must remain
  pixel-identical.
- The result must be photorealistic and sharp, matching this image's
  resolution, focus, grain and lighting direction. No blur, no halo, no
  illustration look.
- Keep the object's real geometry and its contact with the floor, wall or
  ceiling exactly as it is, unless the task explicitly asks otherwise.
- Do not add, move or delete anything the task did not mention.
- Output the edited photograph only."""


INSTRUCT_REMOVE_TEMPLATE = """Edit this interior photograph.

TASK: remove the {label} completely, and reconstruct what would be behind it.

RULES:
- Delete the object along with its shadows and any reflection of it.
- Fill the space it occupied with a plausible continuation of the surfaces
  behind and beneath it -- the floor's pattern, the wall's finish, the skirting
  line -- matching perspective, lighting and texture so the result looks like a
  photograph of the room without that object, not a patch.
- Everything else must remain pixel-identical. Do not add anything in its
  place, and do not move the objects around it.
- Output the edited photograph only."""


INSTRUCT_SCENE_TEMPLATE = """Edit this interior photograph.

TASK: {instruction}

SCOPE -- read this before deciding what to change.

Change ONLY the thing the task names. Everything the task does not name stays
exactly as photographed: same colour, same material, same finish, same
texture, same position. If you are unsure whether something is included, it is
NOT included.

The task names a category, and every other category is untouched. In
particular these are all separate things, and naming one never includes
another:

- WALLS mean the flat plastered or painted wall surface itself. Built-in
  units, wardrobes, shelving, cabinetry, panelling, joinery, a media unit or
  a feature cladding fixed against a wall are FURNITURE, not wall -- even when
  they run floor to ceiling, even when they are the same colour as the wall,
  even when they cover most of it. Painting the walls leaves every one of them
  exactly as it is.
- CEILING means the overhead surface only, not the lights or tracks fixed to it.
- FLOOR means the floor covering only, not the skirting, rugs or furniture
  standing on it.
- FURNITURE means the individual piece named, not the surfaces around it.

Also leave untouched, unless the task names them: all decor and objects on
shelves, all soft furnishings, curtains and blinds, lighting and fittings,
the television and electronics, plants, artwork, and the view through any
window.

RENDERING:
- Keep the room's geometry, every object's position, and the camera framing
  exactly as they are.
- Apply the change to the whole of what was named -- every wall in the room if
  the task says walls, not just the most prominent one.
- Match the room's existing lighting, shadows and reflections.
- Photorealistic and sharp, consistent with the original's resolution and
  grain.
- Output the edited photograph only."""


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ox * oy
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def reconcile_detections(
    old: list[Detection], new: list[Detection], *, min_iou: float = 0.25
) -> list[Detection]:
    """Carry stable identities across a re-detection.

    Detection ids are generated fresh on every pass, but the quote's
    supersession logic is keyed on them: it decides that a later swap of the
    *same* object replaces an earlier one by comparing detection ids. Re-detect
    without reconciling and the same sofa, swapped before and after, becomes
    two ids -- and the customer is charged for two sofas.

    Objects are matched by overlap, best pair first, preferring the same class.
    An unmatched new detection keeps its fresh id: it is genuinely something
    the previous pass did not see.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, n in enumerate(new):
        for j, o in enumerate(old):
            score = _iou(n.box, o.box)
            if score >= min_iou:
                # Same class is a stronger signal than raw overlap; a replaced
                # object keeps its class even when its shape changes a lot.
                pairs.append((score + (0.35 if n.object_class == o.object_class else 0.0), i, j))
    pairs.sort(reverse=True)

    taken_new: set[int] = set()
    taken_old: set[int] = set()
    mapping: dict[int, str] = {}
    for _score, i, j in pairs:
        if i in taken_new or j in taken_old:
            continue
        taken_new.add(i)
        taken_old.add(j)
        mapping[i] = old[j].id

    return [
        Detection(
            id=mapping.get(i, n.id),
            label=n.label,
            object_class=n.object_class,
            box=n.box,
            confidence=n.confidence,
        )
        for i, n in enumerate(new)
    ]


def hit_test(
    detections: list[Detection], x: int, y: int, *, near_pad: int = 25
) -> Detection | None:
    """Map a normalised click to a detection.

    Smallest containing box wins: a lamp standing in front of a wardrobe is
    the smaller box, and clicking the lamp must select the lamp. If nothing
    contains the point, retry with a pad so a click just *outside* the sofa's
    box still selects the sofa -- users aim at objects, not at rectangles.
    """
    containing = [d for d in detections if d.contains(x, y)]
    if not containing:
        containing = [d for d in detections if d.contains(x, y, pad=near_pad)]
    if not containing:
        return None
    return min(containing, key=lambda d: d.area)


REPLACE_PROMPT_TEMPLATE = """Edit this interior photograph (the FIRST image).

TASK: replace exactly one object -- the {label} located roughly at normalised
box [x {x0}-{x1}, y {y0}-{y1}] (0-1000 scale) -- with the replacement product.

THE REPLACEMENT PRODUCT is shown in the SECOND image: {product_name}
({product_desc}). That second image is a catalogue reference on a plain
background, and it may show the product from a DIFFERENT viewing angle than
this room needs.

RENDERING RULES:
1. Remove the original {label} completely, including its shadows, reflections
   and any occlusion artefacts.
2. Place the replacement product at the SAME position and the same floor/wall
   contact as the original, scaled to the product's real-world size relative
   to the room.
3. Re-render the product from the CORRECT viewing angle for this room's camera
   perspective. Rotate it in 3D as needed -- NEVER paste the reference image
   flat. Preserve the product's exact identity: silhouette, materials,
   colours, textures, proportions, leg/base style, stitching, hardware and
   every design detail from the reference.
4. Relight the product for THIS room: match the lighting direction, colour
   temperature, exposure and contrast of the first image. Add correct contact
   shadows where it meets floor or wall, and any reflections the scene implies.
5. The inserted product must be SHARP and photorealistic, matching the first
   image's resolution, focus and grain. No blur, no smearing, no halo at the
   edges, no illustration or 3D-render look.
5a. DRAW IT COMPLETE. The product must be rendered in full and FULLY OPAQUE
   across its entire footprint -- every part of it solid, with its real
   material and pattern continuing edge to edge. It must never be
   semi-transparent, faded, partially drawn, or left showing the old object or
   the floor through it. Where another object sits on top of or in front of
   the product, the product still continues UNBROKEN underneath and behind
   that object; only the covering object hides it, and the product's surface
   resumes on the far side of it.
6. SIZE DIFFERENCES: the replacement may be taller, shorter, wider or
   narrower than the original. Keep its FLOOR (or wall/ceiling) contact and
   its horizontal position the same, and let the rest of it extend naturally
   -- a taller sofa grows UPWARD from the same seat position, it is not
   squashed to fit the old object's outline. Draw the product complete and
   correctly proportioned.
7. PARTIAL VISIBILITY: if the original {label} runs off the edge of this crop
   or is occluded by other objects, the replacement stays cut off the same
   way. NEVER move the object, recentre it, duplicate it, or add another
   instance elsewhere.
8. Everything else in the image -- all other objects, all surfaces, the
   framing -- must remain pixel-identical.
9. Do not add, remove, or move anything else. Output the edited image only."""


PRODUCT_IMAGE_PROMPT = """Create a clean e-commerce catalogue photograph of a
single piece of furniture.

PRODUCT: {name} -- a {object_class}.
SPECIFICATION: {description}

REQUIREMENTS:
- Photorealistic studio product photography. Not an illustration, sketch,
  render-style image, or lifestyle scene.
- Follow the specification exactly: the material, colour, silhouette, leg or
  base style and proportions described above must all be visible and correct.
- Pure white background (#FFFFFF). Nothing else in frame -- no room, no props,
  no other furniture, no people, no text, no watermark, no logos.
- Three-quarter front view at eye level, the whole product in frame,
  occupying roughly 85% of the image, centred.
- Soft even studio lighting with a subtle contact shadow directly beneath the
  product. Sharp focus throughout, clean edges.
- Do not invent a brand marking of any kind.

Output the photograph only."""


CUTOUT_PROMPT = """Edit this product photograph.

TASK: isolate the single main product and remove everything else.

RULES:
- Keep the product itself EXACTLY as photographed: same pixels, same angle,
  same colours, materials, textures and details. Do not restyle, recolour,
  straighten or "improve" the product.
- Remove the entire background, all surrounding objects, props, text,
  watermarks and people.
- Place the product on a pure white background (#FFFFFF), centred, occupying
  roughly 90% of the frame, with a very soft neutral contact shadow directly
  beneath it and nothing else.
- Clean, precise edges -- no halo, no leftover background fragments, no
  cropping of the product.
- Output the edited image only."""


SURFACE_PROMPT_TEMPLATE = """Edit this interior photograph.

RESTYLE exactly one surface: the {label} in the region roughly at normalised
box [x {x0}-{x1}, y {y0}-{y1}] (0-1000 scale).

Apply this finish to that ENTIRE surface region: {product_name}
({product_desc}).

STRICT RULES:
- Change ONLY that surface's finish. Every object in front of it -- furniture,
  decor, fittings -- and every other surface must remain pixel-identical.
- The new finish must follow the surface's real perspective and geometry, and
  continue naturally behind objects that partially cover it.
- Preserve the room's lighting: shadows, highlights and reflections on the
  surface must be re-rendered consistently for the new finish.
- Do not add, remove, or move anything.
- Output the edited photograph only."""


def build_replace_prompt(
    detection: Detection, *, product_name: str, product_desc: str
) -> str:
    x0, y0, x1, y1 = detection.box
    template = (
        SURFACE_PROMPT_TEMPLATE
        if detection.object_class in SURFACE_CLASSES
        else REPLACE_PROMPT_TEMPLATE
    )
    return template.format(
        label=detection.label,
        x0=x0, x1=x1, y0=y0, y1=y1,
        product_name=product_name,
        product_desc=product_desc or "as pictured",
    )


# ---- region-locked editing ----------------------------------------------
#
# Image models treat textual box coordinates as a suggestion: told to replace
# a barely-visible sofa "at x 850-1000", they will happily paint a whole sofa
# somewhere more photogenic. The guarantee the product needs -- ONLY the
# selected region changes -- cannot come from a prompt. It comes from code:
# crop the region (with context), let the model edit only the crop, paste the
# result back, and composite so every pixel outside the detection box is the
# original by construction. A side benefit: a 5%-visible object becomes the
# main subject of its crop, which markedly improves edit accuracy.

def overlapping_detections(
    region: tuple[int, int, int, int],
    detections: "list[Detection]",
    *,
    exclude_id: str | None = None,
    min_overlap: float = 0.10,
) -> "list[Detection]":
    """Objects sharing space with the editable region.

    The region is sized for the incoming product, so on a large replacement it
    inevitably covers neighbours -- and every one of those may itself be a
    product the customer already chose. Naming them lets the model be told to
    keep them, instead of silently repainting over an earlier decision.

    Surfaces are excluded: a wall or floor is *supposed* to be behind the
    object, and listing them as things to preserve is noise.
    """
    rx0, ry0, rx1, ry1 = region
    out = []
    for det in detections:
        if det.id == exclude_id or det.object_class in SURFACE_CLASSES:
            continue
        x0, y0, x1, y1 = det.box
        ox = max(0, min(rx1, x1) - max(rx0, x0))
        oy = max(0, min(ry1, y1) - max(ry0, y0))
        if ox <= 0 or oy <= 0:
            continue
        area = max(1, (x1 - x0) * (y1 - y0))
        if (ox * oy) / area >= min_overlap:
            out.append(det)
    return out


def split_by_depth(
    target: "Detection", neighbours: "list[Detection]"
) -> "tuple[list[Detection], list[Detection]]":
    """Split neighbours into (in front of the target, behind it).

    In a photograph of a room, the floor recedes upward: an object whose base
    sits LOWER in the frame is nearer the camera. That single comparison
    settles the layering question the prompt otherwise has to guess at -- and
    guessing it wrongly is what erases a coffee table when a TV console behind
    it is replaced, because the model is told the console is the nearer object
    and dutifully draws it over the table.
    """
    in_front, behind = [], []
    target_base = target.box[3]
    for nb in neighbours:
        (in_front if nb.box[3] > target_base else behind).append(nb)
    return in_front, behind


PRESERVE_IN_FRONT_TEMPLATE = """
IN FRONT OF THE NEW {target}, AND NEARER THE CAMERA: {items}.
These stand between the camera and the {target}, and every one of them is
already visible in the image you were given. Draw the new {target} first,
complete, and then draw these objects OVER it, copied exactly as they already
appear -- same shape, colour, material, position, scale and orientation. They
hide the parts of the {target} directly behind them; the {target} continues
normally around them. Never delete, move, resize, restyle or substitute any of
them, and never draw the {target} on top of them."""


PRESERVE_NOTE_TEMPLATE = """
ALSO PRESENT IN THIS REGION, AND MUST SURVIVE: {items}.
These are separate objects the customer has already chosen. Keep each one
exactly as it appears -- same design, colour, materials and position. Where
the new {target} physically stands in front of one, draw the overlap
naturally, showing the new object nearer the camera and the existing object
partly hidden behind it. Never delete, restyle, move or substitute them."""


#: When the thing being replaced lies on the floor, everything standing on it
#: is IN FRONT, not behind. Telling the model the opposite makes it try to
#: render the new rug "around" the furniture, and the result reads as a
#: see-through mat with gaps where the table legs are.
PRESERVE_ON_TOP_TEMPLATE = """
LAYERING -- READ THIS CAREFULLY. Objects are standing ON this {target}, and
they are all visible in the image you were given: {items}.

Work in two layers, in this order:

LAYER 1 -- the new {target}. Draw it FIRST and draw it WHOLE: one continuous
surface covering its entire area, with its material and pattern unbroken all
the way across. It passes UNDERNEATH every object standing on it. Do not stop
it at their edges, do not cut holes or gaps around their legs or bases, do not
fade, lighten, blur or wash it out anywhere, and never leave bare floor, a
pale patch, or any part of the previous {target} showing through it.

LAYER 2 -- the objects listed above, drawn ON TOP of that surface. Copy each
one from the image exactly as it already appears: same shape, colour,
material, position, scale and orientation. They hide the parts of the
{target} directly behind them, and the {target} continues normally on every
side of them. Give each a soft, natural contact shadow on the new surface.
Do not delete, move, resize, restyle or substitute any of them, and do not
add anything that is not already there.

If in doubt, the {target} is continuous and the objects sit on it."""


CROP_CONTEXT_NOTE = """NOTE: this first image is a CROPPED REGION of a larger room
photograph, supplied so you can see the object and its immediate surroundings
clearly. Edit within this crop exactly as instructed; the edited crop will be
composited back into the full photograph."""


# Where an object gains size when it is replaced. A sofa's floor contact is
# correct in the original box -- a taller replacement grows UPWARD, not
# downward through the floor. A wall-mounted TV grows around its centre. A
# pendant light grows downward from its ceiling fixing. Getting this wrong is
# what clips the top off a replacement sofa.
_FLOOR_STANDING = {
    "sofa", "armchair", "chair", "coffee_table", "side_table", "dining_table",
    "tv_unit", "bed", "wardrobe", "bookshelf", "cabinet", "ottoman", "rug",
    "plant", "room_divider", "lamp",
}
_WALL_MOUNTED = {"television", "wall_art", "mirror", "curtain", "blinds"}
_CEILING_MOUNTED = {"ceiling_light"}

# Minimum breathing room on a growth-permitted side, as a fraction of the box.
_MIN_ALLOWANCE = 0.15
# Never grow a side by more than this fraction -- an allowance is headroom for
# a differently-proportioned product, not a licence to redraw the room.
_MAX_ALLOWANCE = 1.0
# When the object is clipped by the frame, its true width is unknown, so the
# proportion estimate below is unreliable; fall back to a generous fixed
# headroom instead of a wrong computed one.
_CLIPPED_ALLOWANCE = 0.45
#: How far below a floating object to reach when its replacement stands on the
#: floor. Expressed in multiples of the original box height because a floating
#: console is a thin band and the floor is far below it in those terms.
_FLOATING_TO_FLOOR_DROP = 2.5
#: Ceiling on that reach, so a misread label cannot license repainting half
#: the room.
_MAX_DROP = 4.0

#: Typical width in mm for each class, used only to judge whether a chosen
#: product is unusually large and therefore needs more room than the object it
#: replaces occupied. The original's real size is unknowable from an image, so
#: a class typical is the honest reference point.
#: Items that lie flat on the floor. Their apparent extent up the image is
#: their DEPTH, not their height -- a rug's height_mm is its pile thickness,
#: and reading that as "a 15 mm tall object" asks for no room at all, which is
#: exactly how a larger rug ends up clipped along its far edge.
_FLAT_ON_FLOOR = {"rug"}

_TYPICAL_WIDTH_MM = {
    "sofa": 1900, "armchair": 800, "chair": 500, "coffee_table": 1050,
    "side_table": 450, "dining_table": 1600, "tv_unit": 1600, "bed": 1550,
    "wardrobe": 1600, "bookshelf": 900, "cabinet": 1000, "rug": 1700,
    "lamp": 400, "ceiling_light": 500, "ottoman": 600, "room_divider": 1500,
    "mirror": 700, "wall_art": 800, "plant": 500, "television": 1300,
}


#: Words in a detection label that mean the original object does not touch the
#: floor. A "floating tv console" and a floor-standing TV unit share the class
#: ``tv_unit``, so the class alone cannot tell them apart -- but the label can,
#: and the difference decides which way a replacement is allowed to grow.
_FLOATING_HINTS = (
    "floating", "wall-mounted", "wall mounted", "wall hung", "wall-hung",
    "hanging", "suspended", "mounted",
)


def looks_floating(label: str) -> bool:
    """Whether a detection's own description says it is off the floor."""
    lowered = label.lower()
    return any(hint in lowered for hint in _FLOATING_HINTS)


#: A box whose bottom sits at least this fraction of the image above the floor
#: line is not resting on the floor, whatever its label says.
_FLOATING_GAP_FRAC = 0.06


def is_off_the_floor(
    box: tuple[int, int, int, int], label: str, floor_top: int | None
) -> bool:
    """Whether the ORIGINAL object stands on the floor.

    Measured first, read second. If the floor's own detection is available,
    a clear gap between the object's base and the floor line settles it --
    that works for a wall shelf, a mounted cabinet, a hanging light or
    anything else, and does not depend on the vision model having chosen the
    word "floating". The label is a fallback for when no floor was detected.
    """
    if floor_top is not None:
        gap = floor_top - box[3]
        if gap > _FLOATING_GAP_FRAC * 1000:
            return True
        # Base at or below the floor line: it is standing on it.
        if gap <= 0:
            return False
    return looks_floating(label)


def replaced_object_box(
    box: tuple[int, int, int, int],
    object_class: str,
    product_dims: tuple[int, int, int] | None,
    *,
    label: str = "",
    floor_top: int | None = None,
) -> tuple[int, int, int, int]:
    """Where the NEW object now sits, after a replacement.

    The stored detection still describes the object that was replaced. Leave it
    alone and every later interaction uses the wrong geometry: clicking the
    part of a larger sofa that extends beyond the old outline selects nothing,
    and swapping that sofa again sizes its region from the footprint of an
    object no longer in the picture.

    The new extent is derived the same way the editable region was -- from the
    product's own proportions against a class typical -- but tightened to the
    object itself rather than the allowance around it. It is an estimate, like
    the region, and it is far closer than leaving the old box in place.
    """
    if product_dims is None or product_dims[0] <= 0:
        return box

    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    typical_w = _TYPICAL_WIDTH_MM.get(object_class)
    scale = (product_dims[0] / typical_w) if typical_w else 1.0
    scale = max(0.4, min(3.0, scale))

    new_w = bw * scale
    extent_mm = (
        product_dims[1] if object_class in _FLAT_ON_FLOOR else product_dims[2]
    )
    new_h = new_w * (extent_mm / product_dims[0])

    # Anchor the new box where the object is physically fixed, so it grows the
    # way the real thing would.
    cx = (x0 + x1) / 2
    nx0, nx1 = cx - new_w / 2, cx + new_w / 2

    if object_class in _CEILING_MOUNTED:
        ny0, ny1 = y0, y0 + new_h              # hangs from its fixing
    elif is_off_the_floor(box, label, floor_top) and object_class in _FLOOR_STANDING:
        # Was mounted, now stands: its base drops to the floor.
        base = floor_top if floor_top is not None else y1 + new_h
        ny1 = min(1000, base)
        ny0 = ny1 - new_h
    elif object_class in _FLOOR_STANDING or object_class in _FLAT_ON_FLOOR:
        ny1 = y1                                # floor contact is unchanged
        ny0 = ny1 - new_h
    else:
        cy = (y0 + y1) / 2                      # wall-mounted: about its centre
        ny0, ny1 = cy - new_h / 2, cy + new_h / 2

    clamp = lambda v: int(max(0, min(1000, round(v))))
    out = (clamp(nx0), clamp(ny0), clamp(nx1), clamp(ny1))
    if out[2] - out[0] < 2 or out[3] - out[1] < 2:
        return box
    return out


def replacement_region(
    img_w: int,
    img_h: int,
    box: tuple[int, int, int, int],
    object_class: str,
    product_dims: tuple[int, int, int] | None = None,
    *,
    label: str = "",
    expand: float = 1.0,
    floor_top: int | None = None,
) -> tuple[int, int, int, int]:
    """Grow the detection box into the region a replacement may occupy.

    The detection box bounds the ORIGINAL object; a replacement with different
    proportions needs room the original never used. Returns a normalised
    0-1000 box.

    Which way it grows depends on the TRANSITION, not on the replacement's
    class alone. Replacing a floor-standing sofa with a taller sofa means
    growing upward, because the floor contact in the box is already right.
    Replacing a *floating* console with a legged floor unit is the opposite
    case: the box sits high on the wall, the floor is somewhere below it, and
    growing upward leaves the new unit's legs with nowhere to go -- they get
    clipped at the box edge. Assuming the first case for everything is what
    cuts the bottom off a floor-standing replacement.

    ``floor_top`` is the normalised y where the floor region begins, taken
    from the floor's own detection. When a floating object is replaced by a
    standing one, that is the honest answer to "how far down?" -- scaling the
    original box's height is a guess, and a poor one when the box is a thin
    wall-mounted band with the floor far below it.

    ``expand`` multiplies the computed allowance, for when the heuristic
    misjudges and a person can see that it did.
    """
    if object_class in SURFACE_CLASSES:
        return box  # surfaces are already large regions; nothing to grow into

    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)

    # Is the object clipped by a frame edge? Then its true extent is unknown.
    clipped_h = x0 <= 2 or x1 >= 998
    clipped_v = y0 <= 2 or y1 >= 998

    up = side = down = _MIN_ALLOWANCE

    original_floats = is_off_the_floor(box, label, floor_top)

    if object_class in _FLOOR_STANDING and original_floats:
        # The replacement stands on a floor the original never touched, and
        # that floor is below the box. Reach down to it: without this the legs
        # are cut off at the bottom edge of a thin wall-mounted band.
        up = 0.10
        side = 0.15
        if floor_top is not None and floor_top > y1:
            # Detected floor: reach to where it starts, plus enough beyond for
            # the product to sit ON it rather than end at its edge.
            drop_units = (floor_top - y1) / bh
            depth_margin = 0.6
            if product_dims and product_dims[0] > 0 and not clipped_h:
                # How tall the product should read in view, from its own
                # width:height ratio against the width it occupies.
                depth_margin = max(depth_margin,
                                   (bw * (product_dims[2] / product_dims[0])) / bh)
            down = min(_MAX_DROP, drop_units + depth_margin)
        else:
            # No floor detection to aim at; fall back to a generous multiple.
            down = _FLOATING_TO_FLOOR_DROP
            if product_dims and product_dims[0] > 0 and not clipped_h:
                need_h = bw * (product_dims[2] / product_dims[0])
                if need_h > bh:
                    down = max(down, min(_MAX_DROP, (need_h - bh) / bh))
        down = max(down, _MIN_ALLOWANCE)
    elif object_class in _FLOOR_STANDING:
        # Floor contact is trustworthy; height is where the surprise lives.
        down = 0.03
        side = 0.12
        up = _MIN_ALLOWANCE
        if product_dims and not clipped_h:
            pw, _pd, ph = product_dims
            if pw > 0:
                # Box width is the object's real width in view, so the height
                # a correctly-proportioned replacement needs follows from the
                # product's own width:height ratio.
                need_h = bw * (ph / pw)
                if need_h > bh:
                    up = max(up, min(_MAX_ALLOWANCE, (need_h - bh) / bh))
        elif clipped_h:
            # Width is partial, so the ratio would understate the height.
            up = max(up, _CLIPPED_ALLOWANCE)
    elif object_class in _CEILING_MOUNTED:
        up, down, side = 0.05, max(_MIN_ALLOWANCE, 0.4), 0.15
    elif object_class in _WALL_MOUNTED:
        up = down = side = 0.18
    else:
        up = down = side = _MIN_ALLOWANCE

    if clipped_v and object_class in _FLOOR_STANDING:
        up = max(up, _CLIPPED_ALLOWANCE)

    # General guarantee, whatever the class or transition: the region has to
    # hold a product that may simply be BIGGER than what it replaces. The
    # original's real-world size is unknown -- only its pixels are -- so the
    # comparison is against a typical item of its class. "This L-shape is 37%
    # wider than a typical sofa" is a defensible reason to widen the region by
    # 37%; guessing from the box alone is not, and produces the perverse
    # result that a wide, low product asks for *less* room than a compact one.
    if product_dims and product_dims[0] > 0:
        typical_w = _TYPICAL_WIDTH_MM.get(object_class)
        if typical_w:
            scale = product_dims[0] / typical_w
            if scale > 1.0 and not clipped_h:
                side = max(side, min(_MAX_DROP, (scale - 1.0) / 2 + side))
            # Vertical extent in view: for upright furniture that is its
            # height; for something lying flat it is the depth receding from
            # the camera.
            extent_mm = (
                product_dims[1] if object_class in _FLAT_ON_FLOOR
                else product_dims[2]
            )
            need_h_units = bw * scale * (extent_mm / product_dims[0])
            shortfall = (need_h_units - bh * (1 + up + down)) / bh
            if shortfall > 0:
                if object_class in _CEILING_MOUNTED:
                    down += shortfall
                elif object_class in _FLAT_ON_FLOOR:
                    # A rug's near edge stays put; a bigger one reaches further
                    # away from the camera, which is up the image.
                    up += shortfall * 0.85
                    down += shortfall * 0.15
                elif original_floats and object_class in _FLOOR_STANDING:
                    down += shortfall
                elif object_class in _FLOOR_STANDING:
                    up += shortfall
                else:
                    up += shortfall / 2
                    down += shortfall / 2

    up = min(up, _MAX_DROP)
    down = min(down, _MAX_DROP)
    side = min(side, _MAX_DROP)

    expand = max(1.0, float(expand))
    side, up, down = side * expand, up * expand, down * expand

    nx0 = int(round(x0 - bw * side))
    nx1 = int(round(x1 + bw * side))
    ny0 = int(round(y0 - bh * up))
    ny1 = int(round(y1 + bh * down))

    return (max(0, nx0), max(0, ny0), min(1000, nx1), min(1000, ny1))


def _crop_geometry(
    img_w: int, img_h: int, box: tuple[int, int, int, int],
    *, context_frac: float = 0.30, min_crop_px: int = 320,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Compute (crop_rect, inner_rect_in_crop) in pixels.

    crop_rect: the detection box grown by context on every side (so the model
    sees local floor/wall/lighting), clamped to the image, and grown further
    toward min_crop_px so tiny objects still yield a workable canvas.
    inner_rect_in_crop: the detection box expressed in crop coordinates --
    the only area whose pixels are allowed to change.
    """
    bx0 = int(box[0] / 1000 * img_w)
    by0 = int(box[1] / 1000 * img_h)
    bx1 = max(bx0 + 1, int(box[2] / 1000 * img_w))
    by1 = max(by0 + 1, int(box[3] / 1000 * img_h))

    pad_x = max(32, int((bx1 - bx0) * context_frac))
    pad_y = max(32, int((by1 - by0) * context_frac))
    cx0, cy0 = bx0 - pad_x, by0 - pad_y
    cx1, cy1 = bx1 + pad_x, by1 + pad_y

    # Grow toward a minimum crop size so a sliver of sofa at the frame edge
    # still gives the model real context. Growth is applied as a *shift* when
    # it would run off an edge -- clamping alone would leave a corner object
    # with a tiny, useless canvas.
    def _window(lo: int, hi: int, limit: int) -> tuple[int, int]:
        want = min(max(min_crop_px, hi - lo), limit)
        if hi - lo < want:
            grow = want - (hi - lo)
            lo -= grow // 2
            hi += grow - grow // 2
        if lo < 0:
            hi = min(limit, hi - lo)
            lo = 0
        if hi > limit:
            lo = max(0, lo - (hi - limit))
            hi = limit
        return lo, hi

    cx0, cx1 = _window(cx0, cx1, img_w)
    cy0, cy1 = _window(cy0, cy1, img_h)

    inner = (bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0)
    return (cx0, cy0, cx1, cy1), inner


def _decode_image(data_uri: str):
    import base64 as _b64
    import io

    from PIL import Image

    _, _, b64 = data_uri.partition(",")
    return Image.open(io.BytesIO(_b64.b64decode(b64))).convert("RGB")


def _encode_image(img, *, fmt: str = "PNG", quality: int = 92) -> str:
    """Encode to a data URI.

    PNG by default and specifically for composited results: this loop is
    iterative, and re-encoding the whole photo as JPEG on every swap would
    accumulate generational compression damage across 5-10 edits. Crops sent
    *to* the model use JPEG -- smaller upload, and the model re-renders them
    anyway.
    """
    import base64 as _b64
    import io

    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + _b64.b64encode(buf.getvalue()).decode("ascii")


def object_mask(
    original, box_px: tuple[int, int, int, int], *, ring_px: int = 6,
    sensitivity: float = 2.2,
) -> "object":
    """Estimate which pixels inside a detection box are the OBJECT.

    A bounding box is a poor stand-in for an object. A coffee table's box is
    mostly the floor visible between thin legs, so restoring the whole
    rectangle to protect the table also restores that floor -- which is how a
    freshly drawn rug ends up with a rectangle of old herringbone punched
    through the middle of it.

    Without a segmentation model this approximates the object by contrast: the
    ring of pixels just outside the box is sampled as background, and pixels
    inside the box that differ from it are treated as object. Crude, but it
    separates dark legs and solid furniture from an even floor, which is the
    case that matters here. A real mask model would drop straight in.
    """
    from PIL import Image, ImageFilter

    x0, y0, x1, y1 = box_px
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    inner = original.crop(box_px).convert("RGB")

    # Background is sampled from the BAND around the box, never from the box
    # itself. Including the object in its own background statistics is
    # self-defeating: a large solid object drags the mean onto its own colour
    # and then nothing reads as foreground at all.
    ox0, oy0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
    ox1, oy1 = min(original.width, x1 + ring_px), min(original.height, y1 + ring_px)
    outer = original.crop((ox0, oy0, ox1, oy1)).convert("RGB")
    ow, oh = outer.size
    ix0, iy0 = x0 - ox0, y0 - oy0
    ix1, iy1 = ix0 + w, iy0 + h

    px_out = outer.load()
    band: list[tuple[int, int, int]] = []
    step = max(1, (ow + oh) // 160)
    for yy in range(0, oh, step):
        for xx in range(0, ow, step):
            if ix0 <= xx < ix1 and iy0 <= yy < iy1:
                continue  # inside the box: this is the object, not background
            band.append(px_out[xx, yy])
    if not band:
        return Image.new("L", (w, h), 255)

    n = len(band)
    bg = [sum(c[i] for c in band) / n for i in range(3)]
    var = sum(
        sum((c[i] - bg[i]) ** 2 for i in range(3)) for c in band
    ) / (n * 3)
    spread = max(6.0, var ** 0.5)

    mask = Image.new("L", (w, h), 0)
    px_in = inner.load()
    px_m = mask.load()
    threshold = spread * sensitivity
    for yy in range(h):
        for xx in range(w):
            r, g, b = px_in[xx, yy]
            dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
            if dist > threshold:
                px_m[xx, yy] = 255
    # Close small gaps so a leg reads as one shape rather than speckle.
    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.5))


def composite_region(
    original, edited_crop, crop_rect: tuple[int, int, int, int],
    inner_rect: tuple[int, int, int, int], *, feather_px: int = 10,
) -> "object":
    """Paste an edited crop back so ONLY the inner rect (feathered) changes.

    Even if the model redrew the crop's context pixels, everything outside the
    detection box is discarded in favour of the original -- the locality
    guarantee lives here, not in the prompt.

    """
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    cx0, cy0, cx1, cy1 = crop_rect
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    if edited_crop.size != (crop_w, crop_h):
        edited_crop = edited_crop.resize((crop_w, crop_h), Image.LANCZOS)

    mask = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask)
    ix0, iy0, ix1, iy1 = inner_rect
    # Small blend margin so the seam feathers across the box edge.
    m = feather_px
    draw.rectangle((ix0 - m, iy0 - m, ix1 + m, iy1 + m), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather_px))

    result = original.copy()
    original_crop = original.crop(crop_rect)
    blended = Image.composite(edited_crop, original_crop, mask)
    result.paste(blended, (cx0, cy0))
    return result


class GeminiPhotoEditor:
    """Detection + replacement via the Gemini image models.

    Detection uses the standard text model (bounding boxes are a text task);
    replacement uses the image-output model (Nano Banana line). Both accept an
    injected ``transport`` for tests, mirroring the perception provider.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        detect_model: str | None = None,
        edit_model: str | None = None,
        endpoint: str | None = None,
        timeout_s: float | None = None,
        edit_timeout_s: float | None = None,
        transport: Any | None = None,
    ) -> None:
        import os

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("CLOUD_API_KEY") or ""
        # Detection quality scales with model strength. GEMINI_DETECT_MODEL
        # lets detection run on a stronger model (e.g. non-lite flash or pro)
        # than the cheap text default, without touching other calls.
        self.detect_model = detect_model or os.getenv(
            "GEMINI_DETECT_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        )
        self.edit_model = edit_model or os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
        self.endpoint = endpoint or os.getenv(
            "GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"
        )
        # Two timeouts on purpose: detection is a fast text call, but image
        # EDITS legitimately run 1-3 minutes on real photos. One shared 60s
        # ceiling made every slow edit surface as "read operation timed out".
        self.timeout_s = timeout_s or float(os.getenv("GEMINI_TIMEOUT_S", "60"))
        # Per ATTEMPT, not per operation. A hung request is better abandoned
        # and retried than waited on for four minutes: the retry usually lands
        # on a less loaded worker, and the person watching gets an answer
        # sooner than a single long timeout would give them.
        self.edit_timeout_s = edit_timeout_s or float(
            os.getenv("GEMINI_EDIT_TIMEOUT_S", "150")
        )
        self._transport = transport

    # ------------------------------------------------------------ plumbing

    def _post(self, model: str, payload: dict, *, timeout_s: float | None = None) -> dict:
        if self._transport is not None:
            return self._transport(model, payload)
        if not self.api_key:
            raise ProviderError("no Gemini API key configured")
        read_timeout = timeout_s or self.timeout_s
        try:
            import httpx

            url = f"{self.endpoint}/{model}:generateContent"
            resp = httpx.post(
                url,
                params={"key": self.api_key},
                json=payload,
                # Generous read window for slow generations; tight connect so a
                # dead network still fails fast.
                timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=10.0),
            )
            if resp.status_code >= 400:
                retry_after = resp.headers.get("retry-after")
                try:
                    retry_after_s = float(retry_after) if retry_after else None
                except ValueError:
                    retry_after_s = None
                raise ProviderError(
                    f"Gemini returned HTTP {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                    retry_after_s=retry_after_s,
                )
            return resp.json()
        except ProviderError:
            raise
        except Exception as exc:
            msg = str(exc)
            timed_out = "timed out" in msg.lower() or "timeout" in type(exc).__name__.lower()
            if timed_out:
                msg += (
                    f" (waited {read_timeout:.0f}s; image edits can take minutes -- "
                    "raise GEMINI_EDIT_TIMEOUT_S or retry)"
                )
            # Network faults and timeouts are transient by nature: the request
            # never got a verdict, so retrying is legitimate.
            raise ProviderError(
                f"Gemini request failed: {msg}", retryable=True
            ) from exc

    # ---------------------------------------------------------- retrying

    #: Image generation is the call most likely to meet a busy model, so it
    #: gets a fallback chain. Configure with GEMINI_IMAGE_FALLBACKS (comma
    #: separated) to match the models your key can reach.
    @property
    def image_model_chain(self) -> list[str]:
        import os

        raw = os.getenv("GEMINI_IMAGE_FALLBACKS", "")
        fallbacks = [m.strip() for m in raw.split(",") if m.strip()]
        chain = [self.edit_model, *fallbacks]
        seen: set[str] = set()
        return [m for m in chain if not (m in seen or seen.add(m))]

    def _post_with_retry(
        self,
        models: list[str],
        payload: dict,
        *,
        timeout_s: float | None = None,
        max_attempts: int = 5,
        base_delay_s: float = 2.0,
        on_retry=None,
    ) -> dict:
        """POST with exponential backoff, then fall back to the next model.

        A 503 "high demand" is the service asking us to wait, not telling us
        the request was wrong; treating it as fatal wastes the whole run.
        Permanent failures (bad key, malformed request) are raised at once --
        retrying those just multiplies the same error.
        """
        import random
        import time

        last: ProviderError | None = None
        for model_index, model in enumerate(models):
            for attempt in range(1, max_attempts + 1):
                try:
                    return self._post(model, payload, timeout_s=timeout_s)
                except ProviderError as exc:
                    last = exc
                    if not exc.retryable:
                        raise
                    final_attempt = attempt == max_attempts
                    if final_attempt:
                        break
                    # Honour Retry-After when the server sent one; otherwise
                    # exponential backoff with jitter so parallel runs do not
                    # resynchronise onto the same retry instant.
                    delay = exc.retry_after_s or base_delay_s * (2 ** (attempt - 1))
                    delay = min(delay, 60.0) * (0.8 + 0.4 * random.random())
                    if on_retry:
                        on_retry(model, attempt, max_attempts, delay, exc)
                    time.sleep(delay)
            if model_index + 1 < len(models) and on_retry:
                on_retry(model, max_attempts, max_attempts, 0.0, last, )
        assert last is not None
        raise last

    @staticmethod
    def _image_part(image_ref: str) -> dict:
        if image_ref.startswith("data:"):
            header, _, b64 = image_ref.partition(",")
            mime = header.split(";")[0].removeprefix("data:") or "image/jpeg"
            return {"inline_data": {"mime_type": mime, "data": b64}}
        return {"text": f"(image reference: {image_ref})"}

    # ------------------------------------------------------------- detect

    def detect(self, image_ref: str) -> tuple[list[Detection], list[str]]:
        payload = {
            "contents": [
                {"parts": [{"text": DETECT_PROMPT}, self._image_part(image_ref)]}
            ],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        data = self._post(self.detect_model, payload)
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected detection response shape: {exc}") from exc
        return parse_detections(text)

    # ------------------------------------------------------------ replace

    def replace(
        self,
        image_ref: str,
        detection: Detection,
        *,
        product_name: str,
        product_desc: str = "",
        product_image_ref: str | None = None,
        product_dims: tuple[int, int, int] | None = None,
        expand: float = 1.0,
        floor_top: int | None = None,
        neighbours: "list[Detection] | None" = None,
        on_retry=None,
    ) -> str:
        """Replace one detected object; returns the edited full image as a
        data URI.

        Region-locked: the model only ever sees a crop around the detection,
        and the result is composited back so pixels outside the editable
        region are the original by construction. The editable region is the
        detection box GROWN directionally (see :func:`replacement_region`) so
        a taller replacement has somewhere to go instead of being clipped at
        the original object's outline.
        """
        if not image_ref.startswith("data:"):
            raise ProviderError(
                "replace requires the image as a data URI (got a reference)"
            )
        try:
            original = _decode_image(image_ref)
        except Exception as exc:
            raise ProviderError(f"could not decode image for editing: {exc}") from exc

        img_w, img_h = original.size
        editable = replacement_region(
            img_w, img_h, detection.box, detection.object_class, product_dims,
            label=detection.label, expand=expand, floor_top=floor_top,
        )
        crop_rect, inner = _crop_geometry(img_w, img_h, editable)
        crop = original.crop(crop_rect)

        # Re-express the ORIGINAL object's box in crop-local normalised
        # coordinates. The prompt must point at the object being replaced, not
        # at the grown editable region -- the allowance is headroom the model
        # may use, not the outline it should fill.
        cw, ch = crop.size
        ox0 = int(detection.box[0] / 1000 * img_w) - crop_rect[0]
        oy0 = int(detection.box[1] / 1000 * img_h) - crop_rect[1]
        ox1 = int(detection.box[2] / 1000 * img_w) - crop_rect[0]
        oy1 = int(detection.box[3] / 1000 * img_h) - crop_rect[1]
        local_box = (
            max(0, min(1000, int(ox0 / cw * 1000))),
            max(0, min(1000, int(oy0 / ch * 1000))),
            max(0, min(1000, int(ox1 / cw * 1000))),
            max(0, min(1000, int(oy1 / ch * 1000))),
        )
        local_det = Detection(
            id=detection.id, label=detection.label,
            object_class=detection.object_class,
            box=local_box, confidence=detection.confidence,
        )
        prompt = (
            CROP_CONTEXT_NOTE
            + "\n\n"
            + build_replace_prompt(
                local_det, product_name=product_name, product_desc=product_desc
            )
        )
        at_risk = overlapping_detections(
            editable, neighbours or [], exclude_id=detection.id
        )
        if at_risk:
            if detection.object_class in _FLAT_ON_FLOOR:
                # Everything standing on a floor covering is in front of it.
                prompt += PRESERVE_ON_TOP_TEMPLATE.format(
                    items=", ".join(d.label for d in at_risk[:8]),
                    target=detection.label,
                )
            else:
                # Otherwise decide per neighbour: nearer the camera means it
                # must be drawn over the replacement, not under it.
                in_front, behind = split_by_depth(detection, at_risk)
                if in_front:
                    prompt += PRESERVE_IN_FRONT_TEMPLATE.format(
                        items=", ".join(d.label for d in in_front[:8]),
                        target=detection.label,
                    )
                if behind:
                    prompt += PRESERVE_NOTE_TEMPLATE.format(
                        items=", ".join(d.label for d in behind[:8]),
                        target=detection.label,
                    )

        if is_off_the_floor(detection.box, detection.label, floor_top) \
                and detection.object_class in _FLOOR_STANDING:
            # The geometry now allows the drop; the model still has to be told
            # to use it, or it will draw another floating unit.
            prompt += (
                "\n\nIMPORTANT: the object being replaced is mounted off the "
                "floor, but the replacement is a FREE-STANDING piece. Draw it "
                "resting on the floor below, complete with its legs or base, "
                "at the correct height for its real proportions. Do not leave "
                "it floating and do not crop its base."
            )

        parts: list[dict] = [
            {"text": prompt},
            self._image_part(_encode_image(crop, fmt="JPEG")),
        ]
        if product_image_ref and product_image_ref.startswith("data:"):
            parts.append(self._image_part(product_image_ref))

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        # Edits retry too: a timeout means the request never got a verdict,
        # and the overload that returns 503 to one caller returns "slow" to
        # another. Fewer attempts than a batch job, because a person is
        # waiting on this one.
        data = self._post_with_retry(
            self.image_model_chain,
            payload,
            timeout_s=self.edit_timeout_s,
            max_attempts=getattr(self, "default_attempts", 3),
            on_retry=on_retry,
        )
        try:
            out_parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected edit response shape: {exc}") from exc

        edited_uri: str | None = None
        for part in out_parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                edited_uri = f"data:{mime};base64,{inline['data']}"
                break
        if edited_uri is None:
            raise ProviderError("edit response contained no image data")

        try:
            edited_crop = _decode_image(edited_uri)
        except Exception as exc:
            raise ProviderError(f"could not decode edited image: {exc}") from exc

        # Neighbours keep their pixels, except where the replacement itself
        # stands: the target's own box is where the new object goes, so an
        # overlap there is legitimate occlusion rather than destruction.
        # NOTE: earlier versions pasted neighbouring objects back from the
        # pre-edit image to guarantee they survived. That is removed. Pixels
        # copied from before the edit cannot blend with a surface generated
        # after it, so every paste-back showed as a patch -- first a rectangle
        # of old floor, then a pale block under the tables. The crop sent to
        # the model already CONTAINS those objects, so the model can see them;
        # what it needed was to be told what they are and that they sit on top.
        # That is the prompt's job, above.
        result = composite_region(original, edited_crop, crop_rect, inner)
        # Lossless by default so repeated swaps never accumulate compression
        # damage. EDIT_OUTPUT_FORMAT=JPEG trades that for ~30-70% smaller
        # payloads where bandwidth matters more than generational fidelity.
        import os as _os

        fmt = _os.getenv("EDIT_OUTPUT_FORMAT", "PNG")
        return _encode_image(result, fmt=fmt)

    # ------------------------------------------------------------- cutout

    def generate_product_image(
        self, *, name: str, description: str, object_class: str, on_retry=None
    ) -> str:
        """Generate a catalogue photo from a product's own specification.

        Retries transient failures and falls back through
        :attr:`image_model_chain`, because image models are the first thing to
        return "high demand" under load and a single 503 should not cost a
        product its photograph.
        """
        prompt = PRODUCT_IMAGE_PROMPT.format(
            name=name,
            description=description or "as described",
            object_class=object_class.replace("_", " "),
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        data = self._post_with_retry(
            self.image_model_chain,
            payload,
            timeout_s=self.edit_timeout_s,
            max_attempts=getattr(self, "default_attempts", 5),
            on_retry=on_retry,
        )
        try:
            out_parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected generation response shape: {exc}") from exc
        for part in out_parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                return f"data:{mime};base64,{inline['data']}"
        raise ProviderError("generation response contained no image data")

    # ------------------------------------------------------- instructions

    def analyse_instruction(
        self, text: str, detections: "list[Detection]",
        *, selected: "Detection | None" = None,
    ) -> EditIntent:
        """Work out what a typed request means before acting on it.

        A text call, not an image one: it is fast and cheap next to the edit
        it protects. Sending raw user text straight to an image model wastes a
        minute discovering that "make it darker" was about the wall and not
        the sofa that happened to be selected.
        """
        catalogue = "\n".join(
            f"- {d.id} -- {d.label} ({d.object_class}) -- {list(d.box)}"
            for d in detections
        ) or "- (nothing detected)"
        selection_line = (
            f"THEY HAD SELECTED: {selected.id} -- {selected.label}\n"
            if selected is not None
            else "THEY HAD NOTHING SELECTED.\n"
        )
        payload = {
            "contents": [{"parts": [{"text": INTENT_PROMPT.format(
                text=text, selection_line=selection_line, catalogue=catalogue,
            )}]}],
            "generationConfig": {"temperature": 0.0,
                                 "responseMimeType": "application/json"},
        }
        data = self._post_with_retry([self.detect_model], payload,
                                     max_attempts=3)
        try:
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected intent response shape: {exc}") from exc
        return parse_intent(reply)

    def instruct(
        self, image_ref: str, intent: EditIntent,
        *, target: "Detection | None" = None,
        targets: "list[Detection] | None" = None,
        neighbours: "list[Detection] | None" = None,
        floor_top: int | None = None,
        on_retry=None,
    ) -> str:
        """Carry out an interpreted instruction; returns the edited image.

        With a target, the edit is region-locked exactly as a product swap is,
        so a request to recolour one chair cannot repaint the room. Without
        one -- a genuine whole-scene request -- the full image goes, because
        there is no region to lock to.
        """
        if not image_ref.startswith("data:"):
            raise ProviderError("instruct requires the image as a data URI")

        # One target keeps the tight region lock. Several -- "paint the walls"
        # resolving to every wall in the room -- are edited together over the
        # region that spans them, because doing them one at a time produces
        # visible seams where two walls meet and costs an image call each.
        group = [d for d in (targets or []) if d is not None]
        if not group and target is not None:
            group = [target]

        if not group:
            payload = {
                "contents": [{"parts": [
                    {"text": INSTRUCT_SCENE_TEMPLATE.format(
                        instruction=intent.instruction)},
                    self._image_part(image_ref),
                ]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            }
            data = self._post_with_retry(
                self.image_model_chain, payload,
                timeout_s=self.edit_timeout_s,
                max_attempts=getattr(self, "default_attempts", 3),
                on_retry=on_retry,
            )
            return self._image_from(data)

        try:
            original = _decode_image(image_ref)
        except Exception as exc:
            raise ProviderError(f"could not decode image for editing: {exc}") from exc

        img_w, img_h = original.size
        primary = group[0]
        # A removal needs the surface behind the object, so it gets a little
        # more room than a recolour, which stays on the object itself.
        expand = 1.4 if intent.operation == "remove" else 1.0
        editable = replacement_region(
            img_w, img_h, primary.box, primary.object_class, None,
            label=primary.label, expand=expand, floor_top=floor_top,
        )
        for extra in group[1:]:
            region = replacement_region(
                img_w, img_h, extra.box, extra.object_class, None,
                label=extra.label, expand=expand, floor_top=floor_top,
            )
            editable = (
                min(editable[0], region[0]), min(editable[1], region[1]),
                max(editable[2], region[2]), max(editable[3], region[3]),
            )
        crop_rect, inner = _crop_geometry(img_w, img_h, editable)
        crop = original.crop(crop_rect)
        cw, ch = crop.size

        span = (
            min(d.box[0] for d in group), min(d.box[1] for d in group),
            max(d.box[2] for d in group), max(d.box[3] for d in group),
        )
        ox0 = int(span[0] / 1000 * img_w) - crop_rect[0]
        oy0 = int(span[1] / 1000 * img_h) - crop_rect[1]
        ox1 = int(span[2] / 1000 * img_w) - crop_rect[0]
        oy1 = int(span[3] / 1000 * img_h) - crop_rect[1]
        local = tuple(
            max(0, min(1000, int(v)))
            for v in (ox0 / cw * 1000, oy0 / ch * 1000,
                      ox1 / cw * 1000, oy1 / ch * 1000)
        )

        label = (
            primary.label if len(group) == 1
            else " and ".join(d.label for d in group[:4])
        )
        if intent.operation == "remove":
            body = INSTRUCT_REMOVE_TEMPLATE.format(label=label)
        else:
            body = INSTRUCT_REGION_TEMPLATE.format(
                instruction=intent.instruction, label=label,
                x0=local[0], y0=local[1], x1=local[2], y1=local[3],
            )
        if len(group) > 1:
            body += (
                "\n\nAPPLY IT TO ALL OF THEM: this change covers every one of "
                f"these -- {', '.join(d.label for d in group)} -- not just the "
                "largest or the most prominent. Treat them as one continuous "
                "surface where they meet, so no edge between them is left "
                "half-changed."
            )
        prompt = CROP_CONTEXT_NOTE + "\n\n" + body

        group_ids = {d.id for d in group}
        at_risk = [
            d for d in overlapping_detections(
                editable, neighbours or [], exclude_id=primary.id
            )
            if d.id not in group_ids
        ]
        if at_risk:
            in_front, behind = split_by_depth(primary, at_risk)
            if in_front:
                prompt += PRESERVE_IN_FRONT_TEMPLATE.format(
                    items=", ".join(d.label for d in in_front[:8]),
                    target=label,
                )
            if behind:
                prompt += PRESERVE_NOTE_TEMPLATE.format(
                    items=", ".join(d.label for d in behind[:8]),
                    target=label,
                )

        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                self._image_part(_encode_image(crop, fmt="JPEG")),
            ]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        data = self._post_with_retry(
            self.image_model_chain, payload,
            timeout_s=self.edit_timeout_s,
            max_attempts=getattr(self, "default_attempts", 3),
            on_retry=on_retry,
        )
        try:
            edited_crop = _decode_image(self._image_from(data))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"could not decode edited image: {exc}") from exc

        import os as _os

        result = composite_region(original, edited_crop, crop_rect, inner)
        return _encode_image(result, fmt=_os.getenv("EDIT_OUTPUT_FORMAT", "PNG"))

    @staticmethod
    def _image_from(data: dict) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected edit response shape: {exc}") from exc
        for part in parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                return f"data:{mime};base64,{inline['data']}"
        raise ProviderError("edit response contained no image data")

    def cutout(self, image_ref: str) -> str:
        """Strip a product photo to product-on-white; returns a data URI.

        Run once at upload time, not per swap: the operator sees and approves
        the cutout immediately, and every later replacement gets a clean
        two-image call (room + reference) without paying for this edit again.
        """
        payload = {
            "contents": [{"parts": [{"text": CUTOUT_PROMPT}, self._image_part(image_ref)]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        }
        data = self._post_with_retry(
            self.image_model_chain,
            payload,
            timeout_s=self.edit_timeout_s,
            max_attempts=getattr(self, "default_attempts", 3),
        )
        try:
            out_parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected cutout response shape: {exc}") from exc
        for part in out_parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mime_type") or inline.get("mimeType") or "image/png"
                return f"data:{mime};base64,{inline['data']}"
        raise ProviderError("cutout response contained no image data")


class MockPhotoEditor:
    """Deterministic offline editor for tests and keyless development.

    Detection returns a fixed plausible set; replace returns a synthetic ref
    that encodes what was asked, so tests can assert the chain without any
    image bytes existing.
    """

    def detect(self, image_ref: str) -> tuple[list[Detection], list[str]]:
        dets = [
            Detection(id="det-sofa", label="three-seat sofa", object_class="sofa",
                      box=(150, 550, 700, 900), confidence=0.9),
            Detection(id="det-table", label="wooden coffee table", object_class="coffee_table",
                      box=(350, 380, 620, 560), confidence=0.85),
            Detection(id="det-lamp", label="floor lamp", object_class="lamp",
                      box=(720, 200, 800, 620), confidence=0.8),
            # Surfaces: large regions behind/under everything, so the
            # smallest-box hit-test still prefers objects on a direct hit.
            Detection(id="det-wall", label="beige painted wall", object_class="wall",
                      box=(0, 0, 1000, 620), confidence=0.9),
            Detection(id="det-ceiling", label="white ceiling", object_class="ceiling",
                      box=(0, 0, 1000, 120), confidence=0.9),
            Detection(id="det-floor", label="wooden floor", object_class="floor",
                      box=(0, 880, 1000, 1000), confidence=0.9),
        ]
        return dets, ["MOCK detection -- fixed set, image not read"]

    def replace(self, image_ref: str, detection: Detection, *, product_name: str,
                product_desc: str = "", product_image_ref: str | None = None) -> str:
        return f"mock://edited/{detection.id}/{product_name.replace(' ', '_')}"

    def analyse_instruction(self, text, detections, *, selected=None):
        """Offline stand-in: honours the selection, guesses the operation from
        obvious keywords, and is honest about how little it knows."""
        lowered = text.lower()
        if any(w in lowered for w in ("remove", "delete", "get rid")):
            operation = "remove"
        elif any(w in lowered for w in ("colour", "color", "paint", "repaint")):
            operation = "recolour"
        elif any(w in lowered for w in ("replace", "swap", "change to")):
            operation = "replace"
        else:
            operation = "restyle"
        return EditIntent(
            target_ids=(selected.id,) if selected is not None else (),
            operation=operation,
            instruction=text.strip(),
            confidence=0.3,
            selection_matches=True if selected is not None else None,
            note="MOCK interpretation -- no language model on this path",
        )

    def instruct(self, image_ref, intent, *, target=None, targets=None,
                 neighbours=None, floor_top=None, on_retry=None):
        group = [d for d in (targets or []) if d is not None] or (
            [target] if target is not None else []
        )
        label = "+".join(d.id for d in group) if group else "scene"
        return f"mock://instructed/{label}/{intent.operation}"

    def cutout(self, image_ref: str) -> str:
        """No image model offline; signals 'not processed' with a mock ref so
        the console stores the original and says so, rather than lying."""
        return "mock://cutout/unprocessed"

    def generate_product_image(
        self, *, name: str, description: str, object_class: str
    ) -> str:
        """Offline placeholder: a labelled grey card, clearly not a product
        photo, so a mock run can never be mistaken for real imagery."""
        import base64
        import io

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (768, 768), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((96, 200, 672, 568), fill=(216, 216, 212), outline=(120, 120, 118), width=3)
        draw.text((110, 610), f"MOCK {object_class}", fill=(90, 90, 88))
        draw.text((110, 630), name[:48], fill=(90, 90, 88))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()