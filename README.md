# Interior AI

A backend that turns a photograph of a room into a validated furniture layout,
an editable design, and a costed quotation.

Point it at a room photo and it will estimate the room's dimensions, judge how
finished the room is, solve a furniture layout that actually fits, draw a floor
plan, let a customer swap products in and out of their own photograph, and
produce a contractor / DIY / hybrid quotation priced for their city.

---

## The idea this is built around

Most of the design decisions here follow from one rule:

> **A guess must never be able to pass for a fact.**

An AI model will answer any question you ask it, confidently, including
questions it has no way of knowing the answer to. A photograph does not contain
the room's dimensions. A language model does not know what a painter charges in
Mysuru. If those answers are allowed into the system unlabelled, they end up in
a quote, and someone orders the wrong quantity of tile.

So every number carries its provenance:

| Value | Marked as | Meaning |
|---|---|---|
| Room dimensions from a photo | `estimated_prior` | A typical room of this type and size, not a measurement |
| Room dimensions from a scan | `measured` | Reserved; nothing sets this yet |
| A swapped-in product's price | `known` | The catalogue price the customer was shown |
| Labour, materials, regional rates | `estimated` | The model's estimate for that city |
| A construction signal it could not see | `unknown` | Honest ignorance, not a default |

Where a guess cannot be made honest, the work is refused rather than faked. The
floor plan is *drawn* from solved coordinates, not generated, because an image
model cannot draw to coordinates and a pretty plan that lies about the geometry
is worse than none.

---

## The pipeline

```
    PHOTO
      │
 00 ──┼─► capabilities      probe: is there a GPU? weights? an API key?
      │                     routes LOCAL_FULL / LOCAL_LIGHT / CLOUD_API / MOCK
      │
 01 ──┼─► location          device GPS or a typed city
      │                     → currency, market tier, dimension prior
      │
 02 ──┼─► estimate-scene    model classifies room type + size class
      │                     → prior table supplies typical dimensions
      │                     → scene tagged `estimated_prior`
      │
 03 ──┼─► perceive          seven construction-state signals from the photo
      │                     → phase: SURFACE_FINISHING / FIXTURES_CARPENTRY /
      │                              STYLING_RESTRUCTURE
      │
 04 ──┼─► brief             scope, quality, budget, timeline, occupancy
      │                     (detection runs in the background meanwhile)
      │
 05 ──┼─► pipeline          phase gate → CP-SAT layout solve →
      │                     independent geometric validation →
      │                     immutable scene version → bill of quantities
      │
 06 ──┼─► plan.svg          designer-grade floor plan drawn from the
      │                     solved coordinates
      │
 07 ──┼─► edit session      detect objects → click to select → swap products
      │                     → or type a change in your own words → iterate
      │
 08 ──┴─► quotation         before + after photos + known prices + city
                            → contractor / DIY / hybrid, with a DIY guide
```

### 00 · Capabilities

`GET /capabilities`

Detects what the machine can actually do — a 48 GB GPU with no model weights
is not a local inference box — and picks an execution path. `CLOUD_API` when a
Gemini key is present, `MOCK` otherwise. `MOCK` runs the entire flow with
deterministic stand-ins and says so at every step, so the system is fully
explorable without a key and can never be mistaken for the real thing.

### 01 · Location

`POST /edit-sessions/{id}/location`

Takes either a typed city or device coordinates. Coordinates are resolved
against a **local table of 70 Indian cities** — no geocoding API, no key, no
rate limit, and no third party receiving a user's position. The result gives:

- **currency** for the quote
- **market tier** (metro / tier-2 / tier-3), which changes what identical work
  costs
- **which dimension prior applies**, replacing a dropdown the customer would
  have had to interpret

A fix far from any known city still resolves, but reports the distance —
pricing a village at a metro's rates would be wrong in a way nobody could see.

India only. Other markets are listed in `GET /regions` as unsupported *with the
reason*, because a confident quote for a market we have no rate data for is
worse than an honest refusal.

### 02 · Dimension estimation

`POST /estimate-scene`

**A single photograph cannot measure a room.** There is no scale in an
uncalibrated image; a 4 m wall and a 5 m wall photographed from different
distances are identical pixels. Asking a vision model for millimetres gets a
confident number that is routinely off by 2–3×.

So the work is split between what each part is actually good at:

1. The model classifies **room type** and a **coarse size class** — judgements
   it makes reliably.
2. A **prior table** (`perception/priors.py`) supplies the dimensions for that
   combination, from published Indian residential norms.
3. The result is tagged `estimated_prior` with a confidence that compounds both
   steps, so a shaky classification on a generic fallback yields 0.06 rather
   than a confident wrong number.

Estimated rooms get a typical door and window, so paint quantities deduct
openings instead of billing for painting over them.

When real measurement arrives (LiDAR, a vendor survey), it overwrites the prior
and the flag flips to `measured`. That seam is the only thing standing between
this and accurate geometry.

### 03 · Perception

`POST /scenes/{id}/perceive`

Seven yes / no / partial / unknown signals: walls painted, flooring installed,
ceiling finished, electrical terminated, plumbing terminated, carpentry
installed, furniture present.

`partial` and `unknown` are first-class answers. A room with one wall painted
is `partial`, never `yes`. A floor that is out of frame is `unknown`, and the
prompt says plainly that an honest `unknown` beats a confident guess. `PARTIAL`
blocks progression; `UNKNOWN` lowers confidence and flags for review.

A rules table turns those signals into a **phase**. It is a table, not a
classifier, because the mapping is a business rule and should be readable and
arguable rather than learned.

### 04 · The brief

`POST /edit-sessions/{id}/questionnaire`

Scope, quality tier, budget, timeline, whether the home is occupied. Shapes the
quotation only — it never influences the edits.

Object detection starts in the background at this point, because it needs no
input from the person and they were about to spend a minute typing anyway.

### 05 · Layout and quantities

`POST /scenes/{id}/pipeline`

- **Fit engine** — nine geometric gates, cheapest first: width, depth, height,
  wall, containment, collision, door swing, front clearance, circulation. Every
  rejection carries the measured overage ("too deep by 4700 mm"), never just
  "does not fit".
- **CP-SAT solver** — furniture placed on a 50 mm grid with no overlaps, door
  swings as forbidden regions, four rotations, coffee-table ordering along the
  sofa→focal axis.
- **Independent validation** — the solution is re-checked with Shapely. If the
  solver and the validator disagree, the answer is wrong and that is worth
  knowing.
- **Immutable versioning** — every change creates a new scene version with a
  parent pointer. Old quotes stay reproducible.
- **Quantities from geometry** — floor area plus 8 % cutting wastage; net wall
  area is gross minus openings. Every line shows its arithmetic.
- **Prices** — append-only history, frozen into a snapshot at quote time with
  vendor and observation date. Stale prices are flagged, not hidden; unpriced
  items are surfaced, not zeroed.

The phase gate can be overridden with `force_phase`, and the override is
recorded in the committed version rather than quietly applied.

### 06 · Floor plan

`GET /scenes/{id}/rooms/{id}/plan.svg`

Drawn from the solved coordinates: architectural walls, doors with swing arcs,
windows as glazing symbols, furniture as recognisable icons at true footprint
and rotation, dimension lines, legend, north arrow.

Generated rather than drawn was considered and rejected. An image model cannot
honour coordinates — give it "sofa at x=850" and it paints a sofa somewhere
plausible. For a plan whose entire value is that the sofa is where the sofa
will go, that is not a trade worth making.

### 07 · Interactive editing

The part a customer actually touches.

**Detect** (`POST .../edit-session`) finds every object and surface in the
photo, including walls, ceiling and floor as selectable regions.

**Select** (`POST .../select`) maps a click to an object — smallest containing
box wins, so a lamp in front of a wardrobe selects the lamp — and returns
catalogue products of that class. Products that cannot physically fit the room
are shown *last, with the measured reason*, never hidden.

**Apply** (`POST .../apply`) replaces the object. This is where most of the
engineering lives:

- **Region locking.** The model only ever sees a crop around the target, and
  the result is composited back so pixels outside the editable region are the
  original *by construction*. A prompt asking for locality is a request; this
  is a guarantee.
- **Directional allowance.** The editable region grows the way the object
  would: a taller sofa upward from the same floor contact, a pendant downward
  from its fixing, a floor-standing unit replacing a wall-mounted one *down to
  the detected floor line*.
- **Layer awareness.** Objects nearer the camera are drawn over the
  replacement; things standing on a rug are in front of it, not behind. Getting
  this backwards is what makes a new rug render with holes around the table
  legs.
- **Preflight.** A product far too large for the position asks before spending
  a minute generating — with the measured reason, and nothing is spent if the
  answer is no.
- **Re-detection.** The image is re-analysed afterwards so boxes and labels
  match what is now there. Identities are reconciled by overlap, because
  detection ids regenerate and the quote decides supersession by comparing
  them — without that, one sofa swapped twice is billed twice.

**Instruct** (`POST .../instruct`) takes a change in the customer's own words.
With an object selected, the request is interpreted against the detection map
and region-locked. With nothing selected, the photo and the words go to the
model **with no object map at all** — detection carves a room's walls into
separate regions, and any target resolved from that list repaints one panel
rather than the room.

**Undo** moves a pointer along an append-only step chain. Nothing is deleted,
so the quote can always name the exact step it priced.

### 08 · Quotation

`POST /edit-sessions/{id}/quotation`

Before and after photographs, the city, the brief, and every change go to a
vision-capable text model, which returns three costed options and a suggested
contractor list.

The pricing split is the point:

- **Known items are anchors.** A swapped-in product has a SKU, a catalogue
  price and a vendor. The prompt says plainly that these are not the model's to
  estimate and must be reproduced exactly. A quote that contradicts the price
  the customer saw a minute ago in the picker cannot be explained to them.
- **Everything else is estimated** — labour by trade, materials, regional rates
  — and every returned line declares which kind it is.

A truncated response is repaired rather than discarded: losing an otherwise
complete estimate to one missing brace is a poor trade. Genuinely unreadable
output returns nothing, because a half-invented quote is worse than none.

The DIY option is the one place this could hurt someone, so the prompt requires
an explicit list of what needs a professional — electrical, plumbing, false
ceilings, anything structural — and forbids encouraging unsafe savings.
**Review that output before anyone acts on it.**

---

## Running it

```bash
pip install -e .
cp .env.example .env      # then fill in DATABASE_URL and GEMINI_API_KEY
alembic upgrade head
uvicorn interior_ai.api.app:app --reload
```

- **Pipeline console** — <http://localhost:8000/ui>
- **Product console** — <http://localhost:8000/admin>
- **API docs** — <http://localhost:8000/docs>
- **Health** — <http://localhost:8000/health> (reports whether storage is
  durable and the schema applied)
- **Config** — <http://localhost:8000/config> (which settings loaded, and from
  where; reports presence, never values)

Without `DATABASE_URL` everything runs in memory and is lost on restart —
`/health` says so rather than pretending.

Without `GEMINI_API_KEY` everything runs on deterministic stand-ins, and every
response that would have been a model's says which parts were not real.

### Loading a catalogue

```bash
python -m interior_ai.db.build_catalogue                 # 136 items, generated photos
python -m interior_ai.db.build_catalogue --only sofa --limit 3   # try a few first
python -m interior_ai.db.build_catalogue --no-images     # specs and prices only
```

Product photos are **generated from each product's own specification**, not
collected from the web: a retailer's photograph is theirs, and a photo of some
other company's sofa attached to this catalogue's SKU means the customer picks
one product and the edit inserts a different one. Real supplier photography
beats both and can replace any of them through `/admin`.

Each image is an image-generation call, so the full run takes time. It is
resumable — a product that already has an image is skipped — and it stops
rather than storing a hundred products unpriced if the model goes down.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | in-memory | Postgres URL. Without it nothing persists. |
| `GEMINI_API_KEY` | — | Enables the CLOUD_API path. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Text model. |
| `GEMINI_DETECT_MODEL` | `GEMINI_MODEL` | Detection benefits from a stronger model. |
| `GEMINI_IMAGE_MODEL` | `gemini-3.1-flash-image` | Image editing. |
| `GEMINI_QUOTE_MODEL` | `GEMINI_MODEL` | Quotation. |
| `GEMINI_IMAGE_FALLBACKS` | — | Comma-separated models to fall back to when busy. |
| `GEMINI_EDIT_TIMEOUT_S` | `150` | Per attempt, with retries on top. |
| `EDIT_OUTPUT_FORMAT` | `PNG` | Lossless, so repeated edits do not degrade. |
| `AUTO_CREATE_SCHEMA` | `0` | Create tables without Alembic. Throwaway databases only. |
| `BASIS_ASCII` | `0` | Plain `m2` instead of `m²` for clients that mangle UTF-8. |
| `FORCE_EXECUTION_PATH` | — | Override the capability probe. |

`.env` is loaded automatically — from the working directory, or failing that
from the package's own location, so starting the server from a parent directory
does not silently lose every setting.

---

## Testing

```bash
pytest                 # ~580 tests
pytest -k editing      # one area
```

The suite is hermetic: it clears `DATABASE_URL` and `GEMINI_API_KEY` before
collection, so a developer whose `.env` points at a real Neon database will not
have the tests write to it or spend money on API calls.

---

## What is not built

Stated plainly, because a README that only lists strengths is not much use:

- **Real measurement.** Dimensions are estimated priors. Apple RoomPlan is the
  researched path (iPhone Pro / iPad Pro only — Android has no equivalent), and
  the `estimated_prior → measured` flag is where it plugs in. This is the
  single largest gap.
- **Authentication.** The API is open. Fine locally; not fine on the internet.
- **Object masks.** Preservation of neighbouring objects during an edit is
  instruction, not enforcement. Pixel-accurate masks (SAM-family) would make it
  a guarantee; the interfaces take a region rather than a Gemini call so they
  can drop in.
- **Object storage.** Product images are base64 in the database. Fine at this
  scale; at thousands of products they belong in S3/R2 with URLs in the row.
- **Rate data outside India.** The priors, SKUs and typical costs are Indian.

---

## Layout

```
src/interior_ai/
  core/          scene graph, geometry, units, enums
  perception/    probe, priors, estimator, editing, edit_session, quotation
  fit/           the nine-gate fit engine
  phase/         construction-state rules
  restructure/   CP-SAT layout solver
  pricing/       price history, take-off, quote assembly
  providers/     Gemini and mock implementations
  db/            models, repositories, catalogue, regions, seed scripts
  api/           FastAPI app, schemas, SVG renderer, the two consoles
alembic/         migrations
tests/           ~580 tests
```

## Quick start



``` bash
TO RUN 
ACTIVATE VENE : (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& c:\dev\interior-ai\interior_ai\.venv\Scripts\Activate.ps1)
$env:GEMINI_API_KEY="your-key"; uvicorn interior_ai.api.app:app --reload
python -m interior_ai.db.seed_catalogue --api http://localhost:8000
```
```bash
pip install -e ".[dev]"
pytest                      # 199 tests, ~2s
uvicorn interior_ai.api.app:app --reload
```
