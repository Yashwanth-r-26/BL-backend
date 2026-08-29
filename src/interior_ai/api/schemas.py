"""Wire schemas for the HTTP gateway.

Kept separate from the scene graph models on purpose. The scene graph is the
internal source of truth and is free to change shape; the API is a contract with
clients. Collapsing the two means every internal refactor is a breaking API
change.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from ..core.enums import ObjectClass, OpeningKind, Phase, SwingDirection


class Vec2In(BaseModel):
    x: int
    y: int


class OpeningIn(BaseModel):
    kind: OpeningKind
    centre: Vec2In
    width_mm: int = Field(gt=0)
    height_mm: int = Field(gt=0)
    wall_index: int = Field(ge=0)
    sill_height_mm: int = 0
    swing: SwingDirection | None = None
    swing_radius_mm: int | None = None


class ObstacleIn(BaseModel):
    label: str
    origin: Vec2In
    width_mm: int = Field(gt=0)
    depth_mm: int = Field(gt=0)


class SurfaceStateIn(BaseModel):
    walls_painted: str = "unknown"
    flooring_installed: str = "unknown"
    ceiling_finished: str = "unknown"
    electrical_terminated: str = "unknown"
    plumbing_terminated: str = "unknown"
    carpentry_installed: str = "unknown"
    furniture_present: str = "unknown"


class RoomIn(BaseModel):
    name: str
    polygon: list[Vec2In] = Field(min_length=3)
    ceiling_height_mm: int = Field(gt=0)
    openings: list[OpeningIn] = []
    obstacles: list[ObstacleIn] = []
    surfaces: SurfaceStateIn = SurfaceStateIn()


class SceneIn(BaseModel):
    project_id: str | None = None
    rooms: list[RoomIn] = Field(min_length=1)


class FootprintIn(BaseModel):
    width_mm: int = Field(gt=0)
    depth_mm: int = Field(gt=0)
    height_mm: int = Field(gt=0)


class CatalogueItemIn(BaseModel):
    sku: str
    name: str
    object_class: ObjectClass
    footprint: FootprintIn
    requires_wall: bool = False
    clearance_front_mm: int = 0
    vendor: str | None = None


class FitCheckIn(BaseModel):
    room_id: str
    item: CatalogueItemIn
    origin: Vec2In
    yaw: Literal[0, 90, 180, 270] = 0
    collect_all: bool = False


class RejectionOut(BaseModel):
    code: str
    message: str
    overage_mm: int


class FitCheckOut(BaseModel):
    ok: bool
    rejections: list[RejectionOut] = []
    placement_bounds: list[int] | None = None


class PhaseCheckIn(BaseModel):
    surfaces: SurfaceStateIn


class PhaseCheckOut(BaseModel):
    phase: str
    confidence: float
    needs_review: bool
    reasons: list[str]
    blocking_signals: list[str]
    unknown_signals: list[str]


class RestructureIn(BaseModel):
    room_id: str
    items: list[CatalogueItemIn] = Field(min_length=1)
    focal_point: Vec2In | None = None
    time_limit_s: float = 10.0


class PlacementOut(BaseModel):
    sku: str
    object_class: str
    origin: Vec2In
    yaw: int
    bounds: list[int]


class ValidationOut(BaseModel):
    ok: bool
    containment_ok: bool
    overlap_ok: bool
    door_swing_ok: bool
    obstacle_ok: bool
    violations: list[str]


class RestructureOut(BaseModel):
    ok: bool
    status: str
    placements: list[PlacementOut] = []
    validation: ValidationOut | None = None
    scene_version_id: str | None = None
    reasons: list[str] = []


class PriceObservationIn(BaseModel):
    sku: str
    vendor: str
    unit: str
    amount: Decimal
    currency: str = "INR"
    source: str | None = None


class BOQLineOut(BaseModel):
    sku: str
    description: str
    quantity: Decimal
    unit: str
    basis: str
    status: str
    vendor: str | None = None
    unit_price: Decimal | None = None
    line_total: Decimal | None = None
    observed_at: str | None = None
    age_days: int | None = None


class QuoteOut(BaseModel):
    scene_id: str
    scene_version_id: str
    currency: str
    lines: list[BOQLineOut]
    total: Decimal
    stale_total: Decimal
    is_complete: bool
    warnings: list[str]


class PipelineIn(BaseModel):
    room_id: str
    image_ref: str = "mock://room.jpg"
    items: list[CatalogueItemIn] = []
    focal_point: Vec2In | None = None
    time_limit_s: float = 10.0
    # Override the classified phase. Use when perception left a signal UNKNOWN
    # (e.g. plumbing not visible in the photo) but you know the room is ready.
    # The phase gate is a safety rail, not a lock -- this is the documented way
    # past it, and it is recorded in the committed scene version.
    force_phase: Phase | None = None


class PipelineOut(BaseModel):
    ok: bool
    scene_id: str
    scene_version_id: str
    execution_path: str
    phase: str | None = None
    phase_confidence: float | None = None
    blocked_reason: str | None = None
    stages: list[str]
    placements: list[PlacementOut] = []
    validation: ValidationOut | None = None
    quote: QuoteOut | None = None


class CapabilitiesOut(BaseModel):
    path: str
    forced: bool
    gpu_present: bool
    gpu_name: str | None = None
    vram_mb: int | None = None
    full_weights: bool
    light_weights: bool
    api_key_present: bool
    api_healthy: bool
    reasons: list[str]


class EstimateOut(BaseModel):
    """An estimated room, with its provenance made loud and unmissable."""

    scene_id: str
    scene_version_id: str
    room_id: str
    room_type: str
    size_bucket: str
    width_mm: int
    depth_mm: int
    ceiling_mm: int
    area_m2: float
    dimension_source: str
    confidence: float
    basis: str
    caveat: str
    notes: list[str] = []


class PerceptionOut(BaseModel):
    """What the vision model saw, plus how sure it was and who answered."""

    provider: str
    execution_path: str
    confidence: float
    surfaces: SurfaceStateIn
    notes: list[str] = []
    # The phase that these surfaces classify to, so a caller gets the
    # construction-state answer and its consequence in one round trip.
    phase: str
    phase_confidence: float
    phase_needs_review: bool
    blocking_signals: list[str] = []
    unknown_signals: list[str] = []
    # Present only when the perception was applied to a scene room.
    scene_version_id: str | None = None


# ---- interactive photo editing ------------------------------------------


class DetectionOut(BaseModel):
    id: str
    label: str
    object_class: str
    box: list[int]  # [x_min, y_min, x_max, y_max], normalised 0-1000
    confidence: float


class EditSessionOut(BaseModel):
    session_id: str
    scene_id: str
    room_id: str
    detections: list[DetectionOut]
    current_image_ref: str
    notes: list[str] = []


class OfferOut(BaseModel):
    sku: str
    name: str
    object_class: str
    width_mm: int
    depth_mm: int
    height_mm: int
    display_price: str
    currency: str
    # URL to fetch the product image (never the inline data URI -- offers stay
    # light; the browser loads thumbnails separately).
    image_url: str | None = None
    fits_room: bool
    fit_note: str | None = None
    suggested: bool = False
    swatch: str | None = None


class SelectIn(BaseModel):
    # Either a click (normalised 0-1000 coordinates; frontend divides pixel
    # position by rendered image size) or a direct detection_id when the user
    # picks from the object list. detection_id wins if both are sent.
    x: int | None = Field(default=None, ge=0, le=1000)
    y: int | None = Field(default=None, ge=0, le=1000)
    detection_id: str | None = None


class SelectOut(BaseModel):
    hit: bool
    detection: DetectionOut | None = None
    offers: list[OfferOut] = []
    # Objects a swap here would cover, shown before the user commits.
    affects: list[DetectionOut] = []


class ApplyIn(BaseModel):
    detection_id: str
    sku: str
    # Multiplies the editable region when the automatic allowance misjudges --
    # e.g. a replacement whose base is still clipped. 1.0 is the default; the
    # UI offers a retry at 2.0.
    expand: float = Field(default=1.0, ge=1.0, le=4.0)
    # Re-analyse the edited image so boxes and labels reflect what is actually
    # there. Costs one extra (fast, text) detection call per swap; set false to
    # fall back to estimating the new box from the product's dimensions.
    redetect: bool = True
    # Proceed even though the preflight flagged the product as too large for
    # the position. The check informs; the person decides.
    confirm_oversize: bool = False


class StepOut(BaseModel):
    step_id: int
    detection_id: str
    detection_label: str
    replacement_sku: str
    result_image_ref: str
    swapped_skus: dict[str, str]
    # The session's detections AFTER the swap. A replacement changes an
    # object's size and name, so a client holding the boxes from the original
    # detect call would keep drawing the outline of something that is no
    # longer there.
    detections: list[DetectionOut] = []


class InstructIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    # The object the user had selected, if any. Their words still win when the
    # two disagree -- a misplaced click is common.
    detection_id: str | None = None
    # Proceed even though the request describes something other than what was
    # selected.
    confirm_mismatch: bool = False


class IntentOut(BaseModel):
    target_id: str | None = None
    target_label: str | None = None
    # Every region the request resolved to. "Paint the wall" in a room with
    # several wall regions is all of them, not the most prominent one.
    target_ids: list[str] = []
    target_labels: list[str] = []
    operation: str
    instruction: str
    confidence: float
    selection_matches: bool | None = None
    note: str = ""


class InstructOut(BaseModel):
    applied: bool
    intent: IntentOut
    # Present only when the edit ran.
    step_id: int | None = None
    result_image_ref: str | None = None
    detections: list[DetectionOut] = []
    swapped_skus: dict[str, str] = {}
    # Why nothing happened, when applied is false.
    needs_confirmation: bool = False
    message: str = ""


class LocationIn(BaseModel):
    country: str = "IN"
    # Either a typed city or a device fix. Coordinates are resolved to the
    # nearest known city on the server, so no third-party geocoder ever sees
    # them.
    city: str | None = Field(default=None, max_length=80)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class LocationOut(BaseModel):
    # How the city was arrived at, and how sure we are of it.
    source: str = "manual"        # manual | device
    distance_km: float | None = None
    confident: bool = True
    country: str
    country_name: str
    city: str
    currency: str
    currency_symbol: str
    city_tier: str
    prior_region: str
    supported: bool
    note: str = ""


class QuestionnaireIn(BaseModel):
    # Which categories are in play. Anything not listed is out of scope and
    # should not appear in the quote.
    scope: list[str] = []
    quality_tier: str | None = None       # budget | mid-range | premium
    budget_band: str | None = None        # the owner's own words are fine
    timeline: str | None = None
    occupied_during_work: str | None = None
    notes: str | None = None


class QuotationOut(BaseModel):
    status: str                            # ok | mock | error
    provider: str | None = None
    # The model's structured quotation: three options plus contractors.
    data: dict | None = None
    # What the quote was built from, so a number can be traced back.
    location: dict = {}
    questionnaire: dict = {}
    known_products: list[dict] = []
    instructions: list[dict] = []
    notes: list[str] = []


class SessionQuoteOut(BaseModel):
    session_id: str
    swaps: list[dict]
    quote: "QuoteOut"


class CatalogueItemCreate(BaseModel):
    sku: str
    name: str
    object_class: str
    description: str | None = None
    width_mm: int = Field(gt=0)
    depth_mm: int = Field(gt=0)
    height_mm: int = Field(gt=0)
    display_price: Decimal
    currency: str = "INR"
    vendor: str | None = None
    image_ref: str | None = None
    # e.g. {"suggested": true, "hex": "#C7D2C0"} for paint options
    style_tags: dict = {}


class CatalogueUploadOut(BaseModel):
    sku: str
    stored: bool
    # True when the background strip ran; False means the original photo was
    # stored as-is (MOCK path or cutout failure) -- stated, never hidden.
    image_processed: bool
    image_url: str
    notes: list[str] = []

# ---- accounts ------------------------------------------------------------


class SignupIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(min_length=1, max_length=200)
    display_name: str = Field(default="", max_length=80)


class LoginIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(min_length=1, max_length=200)


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: str


class AuthOut(BaseModel):
    """A successful signup or login.

    The token is a bearer credential -- the client stores it and sends it as
    ``Authorization: Bearer <token>``. ``expires_in`` is seconds, so a client
    can decide to re-authenticate without decoding the token itself.
    """

    token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class SessionSummaryOut(BaseModel):
    """One design in the account's list.

    Deliberately without the image. A data URI is a couple of hundred
    kilobytes and a list of forty of them is a payload nobody wants on mobile
    data; the client opens a design to get its picture.
    """

    session_id: str
    scene_id: str
    room_id: str
    title: str | None = None
    city: str | None = None
    currency_symbol: str | None = None
    swap_count: int = 0
    step_count: int = 0
    created_at: str | None = None


class SessionListOut(BaseModel):
    sessions: list[SessionSummaryOut] = []


class SessionClaimIn(BaseModel):
    """Attach an existing session to the caller's account."""

    session_id: str
    title: str | None = Field(default=None, max_length=120)
