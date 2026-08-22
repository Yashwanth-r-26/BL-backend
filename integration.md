# Mobile integration guide

For wiring the Expo / React Native app to this API.

Every endpoint is plain JSON over HTTPS. Images go up as `multipart/form-data`
and come back as data URIs you can hand straight to `<Image source={{uri}} />`.
There is no SDK to install and no socket to keep open.

---

## The shape of a session

One idea carries the whole flow: **an edit session**. It is created when you
upload a room photo, and everything after that — the detected objects, every
swap, every typed change, the location, the brief, the undo history — hangs off
its id.

```
create session ──► detections
                      │
                      ├─► select ──► offers ──► apply ──┐
                      │                                  ├──► iterate
                      ├─► instruct ─────────────────────┘
                      │
                      └─► quotation
```

Keep the `session_id` in screen state. Nothing else needs to be cached; the
server is the source of truth and every response returns the current state.

---

## Screens

### 1 · Location

**What the user sees.** A short screen: "Where is this room?", a **Use my
location** button, and a city field underneath as the fallback.

```js
import * as Location from "expo-location";

async function detectLocation() {
  const { status } = await Location.requestForegroundPermissionsAsync();
  if (status !== "granted") return null;          // fall through to typing
  const pos = await Location.getLastKnownPositionAsync()
           ?? await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Low });
  return { latitude: pos.coords.latitude, longitude: pos.coords.longitude };
}
```

Ask for **low accuracy** and accept a cached fix. You are choosing a pricing
tier, not navigating — a fix good to a few kilometres is exactly as useful as
one good to five metres, and it returns instantly instead of waking the GPS.

Send whichever you have; the server resolves coordinates to a city itself, so
no geocoding SDK is needed:

```js
POST /edit-sessions/{sessionId}/location
{ "country": "IN", "latitude": 12.97, "longitude": 77.59 }
// or
{ "country": "IN", "city": "Bengaluru" }
```

```json
{ "city": "Bengaluru", "city_tier": "metro", "currency": "INR",
  "currency_symbol": "₹", "source": "device",
  "distance_km": 0.4, "confident": true }
```

**Handle `confident: false`.** It means the nearest city we price is
`distance_km` away. Show the city as a suggestion the user can correct rather
than a settled fact — silently pricing a small town at metro rates is the kind
of error nobody notices until the quote feels wrong.

If permission is denied, do not ask twice. Show the city field.

> Location can be captured before the session exists. Hold it in state and post
> it once you have a `session_id`.

### 2 · Capture

**What the user sees.** A camera view with a single instruction: stand back far
enough to get the whole room in frame.

```js
const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
```

`quality: 0.8` is deliberate. The server downscales to 1536 px anyway, and a
12 MP upload over mobile data is a slow, expensive way to send something that
will be resized on arrival.

### 3 · Estimate

Upload the photo. This creates the scene and estimates the room's size.

```js
const form = new FormData();
form.append("image", { uri: photo.uri, name: "room.jpg", type: "image/jpeg" });
form.append("region", location.prior_region);   // from step 1
form.append("housing", "FLAT_2BHK");

const scene = await fetch(`${API}/estimate-scene`, { method: "POST", body: form })
  .then(r => r.json());
// { scene_id, room_id, width_mm, depth_mm, area_m2,
//   dimension_source: "estimated_prior", confidence: 0.44, caveat: "..." }
```

**Show the caveat.** `dimension_source` will be `estimated_prior`, and the
`caveat` field is written to be displayed verbatim:

> Dimensions are ESTIMATED from typical room sizes, not measured. Quantities
> and costs derived from them are indicative only and must be confirmed against
> a real measurement before ordering.

A confidence of 0.44 is normal and correct — it is an estimate. Do not hide it
behind a spinner and present the number as fact.

### 4 · Perceive, then the brief

Perception reads how finished the room is. It takes a few seconds.

```js
const form = new FormData();
form.append("image", { uri: photo.uri, name: "room.jpg", type: "image/jpeg" });
form.append("room_id", scene.room_id);
const state = await post(`/scenes/${scene.scene_id}/perceive`, form);
// { surfaces: {...}, phase: "SURFACE_FINISHING", phase_confidence: 0.35 }
```

**Then start object detection in the background and show the brief screen
immediately.** Detection needs no input and takes several seconds; the brief is
a minute of tapping. Run them together.

```js
// fire and forget — do not await before navigating
startDetection(scene, photo);
navigation.navigate("Brief");
```

The brief itself is a form: scope checkboxes, quality, budget, timeline,
occupancy. All optional, all improve the quote.

```js
POST /edit-sessions/{sessionId}/questionnaire
{ "scope": ["walls", "furniture"], "quality_tier": "mid-range",
  "budget_band": "under 3 lakh", "timeline": "1–3 months",
  "occupied_during_work": "yes" }
```

By the time they finish, detection has usually returned.

### 5 · Design — the main screen

**What the user sees.** Their photo, filling the screen. A collapsible list of
identified objects. Tapping anything selects it and slides up a product sheet.
A text field at the bottom for describing a change in their own words.

#### Starting the session

```js
const form = new FormData();
form.append("image", { uri: photo.uri, name: "room.jpg", type: "image/jpeg" });
const session = await post(
  `/scenes/${sceneId}/rooms/${roomId}/edit-session`, form);
// { session_id, detections: [{ id, label, object_class, box, confidence }] }
```

#### Boxes and taps

Detection boxes are **normalised 0–1000 on both axes**, so they are independent
of the device, the image resolution and the display size. Convert both ways
against the *rendered* image, not the source:

```js
// draw a box
const style = {
  left:   (box[0] / 1000) * displayWidth,
  top:    (box[1] / 1000) * displayHeight,
  width:  ((box[2] - box[0]) / 1000) * displayWidth,
  height: ((box[3] - box[1]) / 1000) * displayHeight,
};

// a tap becomes a selection
function onImagePress(event) {
  const { locationX, locationY } = event.nativeEvent;
  select({
    x: Math.round((locationX / displayWidth) * 1000),
    y: Math.round((locationY / displayHeight) * 1000),
  });
}
```

Use `resizeMode="contain"` and measure the *displayed* box with `onLayout`. If
the image is letterboxed and you measure the container, every tap lands
slightly wrong — and the error grows toward the edges, where the interesting
objects usually are.

#### Selecting

```js
POST /edit-sessions/{sessionId}/select
{ "x": 420, "y": 680 }              // a tap
{ "detection_id": "a1b2c3" }        // or a row in the list
```

```json
{
  "hit": true,
  "detection": { "id": "a1b2c3", "label": "three-seat sofa", "box": [...] },
  "offers": [
    { "sku": "SOFA-MILANO-3S", "name": "Milano 3-Seater",
      "display_price": "52000", "currency": "INR",
      "image_url": "/catalogue/SOFA-MILANO-3S/image",
      "fits_room": true, "suggested": true, "swatch": null }
  ],
  "affects": [{ "id": "d4e5", "label": "wooden coffee table" }]
}
```

Three things to render properly:

- **`fits_room: false`** — show the product, greyed, with `fit_note` ("too deep
  by 4700 mm"). Do not filter it out; a customer who cannot find a product they
  know you sell assumes the app is broken.
- **`suggested: true`** — these sort first; a badge earns its place.
- **`affects`** — objects this swap will cover. Show it before they commit:
  *"This also covers the coffee table and floor lamp — they'll be kept."*

Walls, ceilings and floors are selectable too, and their offers are paints and
finishes with a `swatch` hex to render as a colour chip.

#### Applying — the slow one

```js
POST /edit-sessions/{sessionId}/apply
{ "detection_id": "a1b2c3", "sku": "SOFA-MILANO-3S" }
```

**This takes 30–90 seconds.** Plan the screen around that:

- Disable the offer buttons immediately. A second tap queues a second edit
  behind the first and the user waits twice as long for a result they did not
  ask for.
- Show elapsed seconds, not an indeterminate spinner. "Editing… 47s" reads as
  progress; a spinning circle for a minute reads as broken.
- Past ~150 s, say the model may be busy and that it is retrying. It is true,
  and it stops them killing the app.

Handle **409** — it is not an error:

```js
if (res.status === 409) {
  const { code, message, reasons } = (await res.json()).detail;
  if (code === "oversize_replacement") {
    // "This product is larger than the space it would occupy. Continue anyway?"
    // reasons: ["It does not fit the room: too deep by 4700 mm", ...]
    if (await confirm(message, reasons)) retry({ confirm_oversize: true });
  }
}
```

Nothing has been generated at that point, so cancelling costs nothing.

On success:

```json
{ "step_id": 3, "result_image_ref": "data:image/png;base64,...",
  "swapped_skus": { "a1b2c3": "SOFA-MILANO-3S" },
  "detections": [ ...refreshed boxes and labels... ] }
```

**Replace your detections with the ones in the response.** The swap resized and
renamed the object; keeping the originals means drawing a stale outline around
something that is no longer there.

#### Typing a change

```js
POST /edit-sessions/{sessionId}/instruct
{ "text": "paint the walls sage green",
  "detection_id": selectedId || null }
```

Whether something is selected changes the behaviour meaningfully, and the UI
should say so:

- **Selected** → the change is locked to that object. Show *"Changing: the
  three-seat sofa"* above the field.
- **Nothing selected** → the request goes to the model with the whole photo and
  no object map, so "paint the walls" means every wall. Show *"Applies to the
  whole photo"*.

If `applied` is false with `needs_confirmation`, their words described
something other than what they tapped:

> "That sounds like the marble feature wall, not what you selected. Apply it
> there instead?"

Retry with `confirm_mismatch: true` if they agree. A misplaced tap is common
and their words are the better evidence of intent.

#### Undo

```js
POST /edit-sessions/{sessionId}/undo
// { current_image_ref, swapped_skus, detections }
```

Fast — no model call. Give it a permanent, obvious place on the screen. It is
what makes experimenting feel safe, and everything else on this screen is an
experiment.

### 6 · Quotation

```js
POST /edit-sessions/{sessionId}/quotation
```

Takes 20–60 seconds. Same progress treatment as the edit.

```json
{
  "status": "ok",
  "data": {
    "currency_symbol": "₹",
    "contractor": { "total": 186500, "materials_total": 121000,
                    "labor_total": 65500, "timeline_weeks": 3,
                    "line_items": [ { "name": "Milano 3-Seater",
                                      "total": 52000, "pricing": "known" } ] },
    "diy": { "materials_total": 138000, "steps": [...],
             "needs_professional": ["Electrical work — ..."] },
    "hybrid": { "total": 154000, "plan_summary": "..." },
    "contractors": [ { "name": "...", "quote_min": 170000 } ]
  },
  "known_products": [...], "instructions": [...]
}
```

Render as three cards — Contractor / DIY / Hybrid — with a total each, then
detail on tap.

**Two things must not be flattened:**

`pricing` on every line item is `"known"` or `"estimated"`. Known lines are the
catalogue prices the customer already saw; estimated lines are the model's
figures for their city. Badge them differently. Presenting an estimate with the
same authority as a real price is how someone budgets on a number that was
never a commitment.

`needs_professional` is a safety list — electrical, plumbing, structural. Give
it visual weight in the DIY tab. This is the one screen in the app where
getting it wrong could hurt somebody, and a collapsed accordion is not weight.

Handle **409** here too: `location_required` (send them back to step 1) and
`nothing_to_quote` (they have not changed anything yet).

---

## Practical notes

**Errors.** Every failure returns `detail` — a string, or an object with `code`
and `message` for the ones you are meant to act on. `409` always means "a
decision is needed", never "something broke".

**Timeouts.** Set the client timeout to **180 seconds** for `/apply`,
`/instruct` and `/quotation`. React Native's default will fire long before the
model answers.

**Backgrounding.** iOS suspends network requests when the app is backgrounded.
A user who switches away mid-edit will come back to a dead request; the server
finished the work regardless. `GET /edit-sessions/{id}` returns the current
state, so recover with that on resume rather than re-running the edit.

**Images.** Results are data URIs — usable directly, but large. Do not put them
in navigation params (both platforms cap the size); keep them in state or a
store.

**Product thumbnails.** `image_url` is a path — prefix your API base. They are
served as ordinary images, so `<Image>` caches them normally.

**Offline.** There is no offline mode. Every meaningful action needs the
server. Detect connectivity up front rather than letting each call fail on its
own.

---

## Endpoint summary

| Endpoint | Method | Typical time |
|---|---|---|
| `/regions` | GET | instant |
| `/capabilities` | GET | instant |
| `/estimate-scene` | POST | 2–5 s |
| `/scenes/{id}/perceive` | POST | 2–5 s |
| `/scenes/{id}/rooms/{id}/edit-session` | POST | 3–8 s |
| `/scenes/{id}/rooms/{id}/plan.svg` | GET | instant |
| `/edit-sessions/{id}/location` | POST | instant |
| `/edit-sessions/{id}/questionnaire` | POST | instant |
| `/edit-sessions/{id}/select` | POST | instant |
| `/edit-sessions/{id}/apply` | POST | **30–90 s** |
| `/edit-sessions/{id}/instruct` | POST | **30–90 s** |
| `/edit-sessions/{id}/undo` | POST | instant |
| `/edit-sessions/{id}/redetect` | POST | 3–8 s |
| `/edit-sessions/{id}/quote` | POST | instant |
| `/edit-sessions/{id}/quotation` | POST | **20–60 s** |
| `/catalogue` | GET | instant |
| `/catalogue/{sku}/image` | GET | instant |

Full schemas at `/docs` on your deployment.