"""HTTP gateway.

Scenes live in an in-memory store by default so the service runs with no
database (useful for CI and demos). Setting ``DATABASE_URL`` swaps in the
SQLAlchemy repository -- see :mod:`interior_ai.db.repository`.

Note that a blocked pipeline returns **200, not 4xx**. "This room is not ready
for furniture" is a successful analysis with a negative answer, and clients
need the full reasoning payload to render it. Returning 422 would push callers
into treating a normal outcome as an error.
"""

from __future__ import annotations

import base64
import os
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm.attributes import flag_modified

from ..core.enums import Unit
from ..core.scene import (
    CatalogueItem,
    Footprint,
    Obstacle,
    Opening,
    Room,
    Scene,
    SurfaceState,
    Vec2,
)
from ..fit.engine import FitEngine
from ..orchestrator import Orchestrator
from ..perception.probe import get_probe
from ..phase.rules import classify
from ..pricing.engine import PricingEngine, Quote
from ..pricing.prices import PriceBook, PriceObservation
from ..providers.base import PerceptionResult, ProviderError
from ..providers.mock import MockPerceptionProvider
from ..restructure.solver import LayoutSolver, SolveRequest, validate_solution
from . import schemas as S

# --------------------------------------------------------------- state


class SceneStore:
    """In-memory scene store keyed by scene id, holding every version.

    Versions are appended, never replaced -- the same immutability the scene
    graph promises, enforced at the storage layer.
    """

    def __init__(self) -> None:
        self._latest: dict[str, Scene] = {}
        self._versions: dict[str, Scene] = {}

    def put(self, scene: Scene) -> Scene:
        self._latest[scene.id] = scene
        self._versions[scene.version_id] = scene
        return scene

    def get(self, scene_id: str) -> Scene:
        if scene_id not in self._latest:
            raise KeyError(scene_id)
        return self._latest[scene_id]

    def get_version(self, version_id: str) -> Scene:
        if version_id not in self._versions:
            raise KeyError(version_id)
        return self._versions[version_id]

    def all_versions(self, scene_id: str) -> list[Scene]:
        return sorted(
            (s for s in self._versions.values() if s.id == scene_id),
            key=lambda s: s.version,
        )


# --------------------------------------------------------- converters


def _vec(v: S.Vec2In) -> Vec2:
    return Vec2(x=v.x, y=v.y)


def _room_from_in(r: S.RoomIn) -> Room:
    openings = tuple(
        Opening(
            kind=o.kind,
            centre=_vec(o.centre),
            width_mm=o.width_mm,
            height_mm=o.height_mm,
            wall_index=o.wall_index,
            sill_height_mm=o.sill_height_mm,
            swing=o.swing,
            swing_radius_mm=o.swing_radius_mm,
        )
        for o in r.openings
    )
    obstacles = tuple(
        Obstacle(
            label=o.label,
            origin=_vec(o.origin),
            width_mm=o.width_mm,
            depth_mm=o.depth_mm,
        )
        for o in r.obstacles
    )
    return Room(
        name=r.name,
        polygon=tuple(_vec(p) for p in r.polygon),
        ceiling_height_mm=r.ceiling_height_mm,
        openings=openings,
        obstacles=obstacles,
        surfaces=SurfaceState(**r.surfaces.model_dump()),
    )


def _item_from_in(i: S.CatalogueItemIn) -> CatalogueItem:
    return CatalogueItem(
        sku=i.sku,
        name=i.name,
        object_class=i.object_class,
        footprint=Footprint(**i.footprint.model_dump()),
        requires_wall=i.requires_wall,
        clearance_front_mm=i.clearance_front_mm,
        vendor=i.vendor,
    )


def _quote_out(q: Quote) -> S.QuoteOut:
    return S.QuoteOut(
        scene_id=q.scene_id,
        scene_version_id=q.scene_version_id,
        currency=q.currency,
        lines=[
            S.BOQLineOut(
                sku=l.sku,
                description=l.description,
                quantity=l.quantity,
                unit=str(l.unit),
                basis=l.basis,
                status=str(l.status),
                vendor=l.price.vendor,
                unit_price=l.price.amount,
                line_total=l.line_total,
                observed_at=(
                    l.price.observed_at.isoformat() if l.price.observed_at else None
                ),
                age_days=l.price.age_days,
            )
            for l in q.lines
        ],
        total=q.total,
        stale_total=q.stale_total,
        is_complete=q.is_complete,
        warnings=list(q.warnings()),
    )


def _placement_out(p: Any) -> S.PlacementOut:
    return S.PlacementOut(
        sku=p.sku,
        object_class=str(p.object_class),
        origin=S.Vec2In(x=p.origin.x, y=p.origin.y),
        yaw=p.yaw,
        bounds=list(p.bounds),
    )


# Image types Gemini's vision endpoint accepts. Anything else is rejected up
# front rather than sent and bounced, so the caller gets a clear 415 instead of
# an opaque provider error.
_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
    "image/heic": "image/heic",
    "image/heif": "image/heif",
}

# 20 MB. Gemini's own inline-data ceiling is 20 MB; refusing larger here avoids
# base64-inflating a huge file only for the API to reject it.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _select_perception_provider():
    """Pick a perception backend from the current capability probe.

    CLOUD_API -> Gemini (the only path that actually reads pixels today).
    Everything else -> MOCK, which is honest about not looking at the image.
    Constructed per request so a key added at runtime is picked up without a
    restart.
    """
    caps = get_probe().detect()
    if caps.path.value == "CLOUD_API":
        try:
            from ..providers.gemini import GeminiPerceptionProvider

            return GeminiPerceptionProvider(), caps
        except Exception:
            return MockPerceptionProvider(), caps
    return MockPerceptionProvider(), caps


def _downscale(raw: bytes, *, max_side: int = 1536) -> tuple[bytes, str] | None:
    """Shrink an oversized photo for API calls. Returns (bytes, mime) or None
    to keep the original.

    Phone photos arrive at 4000px+; the vision models neither need nor return
    that resolution, and megabyte-scale base64 payloads are what turn a
    2-minute image edit into a timeout. 1536px preserves everything detection
    and editing use. If Pillow is not installed, the original passes through
    untouched -- slower, but never broken.
    """
    try:
        import io

        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        if max(img.size) <= max_side:
            return None
        img.thumbnail((max_side, max_side))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=88)
        return out.getvalue(), "image/jpeg"
    except Exception:
        # A corrupt or exotic image should fail later with a clear provider
        # error, not here inside an optimisation.
        return None


def _image_to_data_uri(raw: bytes, content_type: str | None) -> str:
    mime = _ALLOWED_IMAGE_TYPES.get((content_type or "").lower(), "image/jpeg")
    scaled = _downscale(raw)
    if scaled is not None:
        raw, mime = scaled
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _perception_out(
    result: PerceptionResult, *, scene_version_id: str | None = None
) -> S.PerceptionOut:
    verdict = classify(result.surfaces)
    return S.PerceptionOut(
        provider=result.provider,
        execution_path=str(result.path),
        confidence=result.confidence,
        surfaces=S.SurfaceStateIn(**result.surfaces.model_dump()),
        notes=list(result.notes),
        phase=str(verdict.phase),
        phase_confidence=verdict.confidence,
        phase_needs_review=verdict.needs_review,
        blocking_signals=list(verdict.blocking_signals),
        unknown_signals=list(verdict.unknown_signals),
        scene_version_id=scene_version_id,
    )


# ------------------------------------------------------------- factory


def create_app(
    *,
    store: SceneStore | None = None,
    price_book: PriceBook | None = None,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Interior Design AI",
        version="0.1.0",
        description=(
            "Scene-graph-driven interior design service: geometric fit checking, "
            "rules-based phase classification, CP-SAT layout optimisation, and "
            "reproducible quantity-based pricing."
        ),
    )

    app.state.fit = FitEngine()

    # Relational storage. With DATABASE_URL set (Neon or any Postgres),
    # scenes, prices, catalogue and edit sessions all persist across restarts.
    # Without it, an in-process SQLite keeps every feature working for local
    # experiments -- but it is wiped when the process exits, which is exactly
    # why an uploaded product seemed to vanish.
    from ..db import catalogue as _catalogue_models  # noqa: F401 -- registers tables on Base.metadata
    from ..db.repository import create_all, make_engine, make_session_factory

    import os as _os

    database_url = _os.getenv("DATABASE_URL")
    if database_url:
        _engine = make_engine()
        # Alembic owns a real database. Calling create_all here as well is
        # what produces "relation already exists" on the next `alembic
        # upgrade head`: the tables appear without a version stamp, so the
        # first migration tries to create them again. Opt in with
        # AUTO_CREATE_SCHEMA=1 for throwaway databases where running
        # migrations is more ceremony than it is worth.
        if _os.getenv("AUTO_CREATE_SCHEMA") == "1":
            create_all(_engine)
    else:
        # Shared in-memory SQLite: StaticPool pins a single connection so every
        # request sees the same database. Without it each checkout would get a
        # fresh empty :memory: db and sessions would vanish between requests.
        # Migrations cannot apply to a database that dies with the process, so
        # here create_all is the only option.
        from sqlalchemy.pool import StaticPool

        _engine = make_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        create_all(_engine)
    app.state.db_sessionmaker = make_session_factory(_engine)
    app.state.persistent = bool(database_url)

    # Explicit arguments always win -- tests pass their own stores. Otherwise
    # persistence follows the database: durable when one is configured, and
    # in-memory when it is not, rather than silently half-persisting.
    if store is not None:
        app.state.store = store
    elif database_url:
        from ..db.stores import SqlSceneStore

        app.state.store = SqlSceneStore(app.state.db_sessionmaker)
    else:
        app.state.store = SceneStore()

    if price_book is not None:
        app.state.price_book = price_book
    elif database_url:
        from ..db.stores import SqlPriceBookAdapter

        app.state.price_book = SqlPriceBookAdapter(app.state.db_sessionmaker)
    else:
        app.state.price_book = PriceBook()

    app.state.orchestrator = orchestrator or Orchestrator(
        price_book=app.state.price_book
    )

    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        """Liveness plus storage durability.

        ``persistent`` answers the question that matters operationally: will
        anything I create here still exist after a restart? A green health
        check that hides "your data is in RAM" is worse than no check.
        """
        from sqlalchemy import text as _text

        persistent = bool(getattr(app.state, "persistent", False))
        backend = "in-memory"
        if persistent:
            try:
                backend = app.state.db_sessionmaker.kw["bind"].dialect.name
            except Exception:
                backend = "database"
        info: dict = {
            "status": "ok",
            "persistent": persistent,
            "storage": backend,
        }
        try:
            with app.state.db_sessionmaker() as db:
                db.execute(_text("SELECT 1"))
            info["database"] = "reachable"
        except Exception as exc:  # pragma: no cover - depends on environment
            info["status"] = "degraded"
            info["database"] = f"unreachable: {type(exc).__name__}"
            return info

        if persistent:
            # A reachable database with no tables is the most likely
            # first-run state, and the least obvious: every later request
            # fails with a driver-level error that says nothing about
            # migrations. Name the fix here instead.
            from sqlalchemy import inspect as _inspect

            try:
                tables = set(_inspect(app.state.db_sessionmaker.kw["bind"]).get_table_names())
            except Exception:
                tables = set()
            missing = {"scene_versions", "price_history", "catalogue_items"} - tables
            if missing:
                info["status"] = "degraded"
                info["schema"] = (
                    f"missing tables: {', '.join(sorted(missing))} -- "
                    "run `alembic upgrade head`"
                )
            else:
                info["schema"] = "ready"
        return info

    @router.get("/config")
    def config_status() -> dict:
        """Which configuration the server actually loaded, and from where.

        Reports presence, never values -- knowing a key is set is useful for
        debugging, printing it into a browser tab is not. This exists because
        "why isn't it reading my .env" is otherwise guesswork.
        """
        from ..config import describe_env

        info = describe_env()
        info["persistent_storage"] = bool(getattr(app.state, "persistent", False))
        return info

    @router.get("/capabilities", response_model=S.CapabilitiesOut)
    def capabilities() -> S.CapabilitiesOut:
        caps = get_probe().detect()
        return S.CapabilitiesOut(
            path=str(caps.path),
            forced=caps.forced,
            gpu_present=caps.gpu.present,
            gpu_name=caps.gpu.name,
            vram_mb=caps.gpu.vram_mb,
            full_weights=caps.full_weights,
            light_weights=caps.light_weights,
            api_key_present=caps.api_key_present,
            api_healthy=caps.api_healthy,
            reasons=list(caps.reasons),
        )

    # ---- scenes -----------------------------------------------------

    @router.post("/scenes", status_code=201)
    def create_scene(body: S.SceneIn) -> dict:
        scene = Scene(
            project_id=body.project_id,
            rooms=tuple(_room_from_in(r) for r in body.rooms),
        )
        app.state.store.put(scene)
        return {
            "scene_id": scene.id,
            "version_id": scene.version_id,
            "version": scene.version,
            "rooms": [{"id": r.id, "name": r.name} for r in scene.rooms],
        }

    @router.get("/scenes/{scene_id}")
    def get_scene(scene_id: str) -> dict:
        try:
            scene = app.state.store.get(scene_id)
        except KeyError:
            raise HTTPException(404, f"no scene {scene_id}")
        return {
            "scene_id": scene.id,
            "version_id": scene.version_id,
            "parent_version_id": scene.parent_version_id,
            "version": scene.version,
            "rooms": [
                {
                    "id": r.id,
                    "name": r.name,
                    "ceiling_height_mm": r.ceiling_height_mm,
                    "placements": [_placement_out(p).model_dump() for p in r.placements],
                    "phase": str(r.phase) if r.phase else None,
                }
                for r in scene.rooms
            ],
        }

    @router.get("/scenes/{scene_id}/versions")
    def scene_versions(scene_id: str) -> dict:
        versions = app.state.store.all_versions(scene_id)
        if not versions:
            raise HTTPException(404, f"no scene {scene_id}")
        return {
            "scene_id": scene_id,
            "versions": [
                {
                    "version": s.version,
                    "version_id": s.version_id,
                    "parent_version_id": s.parent_version_id,
                    "notes": s.notes,
                    "created_at": s.created_at.isoformat(),
                }
                for s in versions
            ],
        }

    # ---- fit --------------------------------------------------------

    @router.post("/scenes/{scene_id}/fit", response_model=S.FitCheckOut)
    def fit_check(scene_id: str, body: S.FitCheckIn) -> S.FitCheckOut:
        try:
            scene = app.state.store.get(scene_id)
            room = scene.room(body.room_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        res = app.state.fit.check(
            _item_from_in(body.item),
            room,
            _vec(body.origin),
            body.yaw,
            collect_all=body.collect_all,
        )
        return S.FitCheckOut(
            ok=res.ok,
            rejections=[
                S.RejectionOut(code=str(r.code), message=r.message, overage_mm=r.overage_mm)
                for r in res.rejections
            ],
            placement_bounds=list(res.placement.bounds) if res.placement else None,
        )

    # ---- phase ------------------------------------------------------

    @router.post("/phase", response_model=S.PhaseCheckOut)
    def phase_check(body: S.PhaseCheckIn) -> S.PhaseCheckOut:
        verdict = classify(SurfaceState(**body.surfaces.model_dump()))
        return S.PhaseCheckOut(
            phase=str(verdict.phase),
            confidence=verdict.confidence,
            needs_review=verdict.needs_review,
            reasons=list(verdict.reasons),
            blocking_signals=list(verdict.blocking_signals),
            unknown_signals=list(verdict.unknown_signals),
        )

    # ---- restructure ------------------------------------------------

    @router.post("/scenes/{scene_id}/restructure", response_model=S.RestructureOut)
    def restructure(scene_id: str, body: S.RestructureIn) -> S.RestructureOut:
        try:
            scene = app.state.store.get(scene_id)
            room = scene.room(body.room_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        req = SolveRequest(
            room=room,
            items=tuple(_item_from_in(i) for i in body.items),
            focal_point=_vec(body.focal_point) if body.focal_point else None,
            time_limit_s=body.time_limit_s,
        )
        result = LayoutSolver().solve(req)
        if not result.ok:
            return S.RestructureOut(
                ok=False, status=result.status, reasons=list(result.reasons)
            )

        report = validate_solution(room, result.placements)
        validation = S.ValidationOut(
            ok=report.ok,
            containment_ok=report.containment_ok,
            overlap_ok=report.overlap_ok,
            door_swing_ok=report.door_swing_ok,
            obstacle_ok=report.obstacle_ok,
            violations=list(report.violations),
        )
        if not report.ok:
            return S.RestructureOut(
                ok=False,
                status="VALIDATION_FAILED",
                placements=[_placement_out(p) for p in result.placements],
                validation=validation,
                reasons=list(report.violations),
            )

        new_room = room.model_copy(update={"placements": result.placements})
        new_scene = scene.replace_room(new_room, notes="restructure via API")
        app.state.store.put(new_scene)

        return S.RestructureOut(
            ok=True,
            status=result.status,
            placements=[_placement_out(p) for p in result.placements],
            validation=validation,
            scene_version_id=new_scene.version_id,
        )

    # ---- pricing ----------------------------------------------------

    @router.post("/prices", status_code=201)
    def record_price(body: S.PriceObservationIn) -> dict:
        try:
            unit = Unit(body.unit)
        except ValueError:
            raise HTTPException(422, f"unknown unit {body.unit!r}")
        obs = PriceObservation(
            sku=body.sku,
            vendor=body.vendor,
            unit=unit,
            amount=Decimal(body.amount),
            currency=body.currency,
            source=body.source,
        )
        app.state.price_book.record(obs)
        return {
            "recorded": True,
            "sku": obs.sku,
            "observed_at": obs.observed_at.isoformat(),
            "history_depth": len(app.state.price_book.history_for(obs.sku)),
        }

    @router.get("/prices/{sku}")
    def get_price(sku: str) -> dict:
        snap = app.state.price_book.snapshot(sku)
        return {
            "sku": sku,
            "status": str(snap.status),
            "amount": str(snap.amount) if snap.amount is not None else None,
            "vendor": snap.vendor,
            "observed_at": snap.observed_at.isoformat() if snap.observed_at else None,
            "age_days": snap.age_days,
            "explanation": snap.explain(),
            "history_depth": len(app.state.price_book.history_for(sku)),
        }

    @router.post("/scenes/{scene_id}/quote", response_model=S.QuoteOut)
    def quote(scene_id: str) -> S.QuoteOut:
        try:
            scene = app.state.store.get(scene_id)
        except KeyError:
            raise HTTPException(404, f"no scene {scene_id}")
        q = PricingEngine(app.state.price_book).quote_scene(scene)
        return _quote_out(q)

    # ---- full pipeline ----------------------------------------------

    @router.post("/scenes/{scene_id}/pipeline", response_model=S.PipelineOut)
    def pipeline(scene_id: str, body: S.PipelineIn) -> S.PipelineOut:
        try:
            scene = app.state.store.get(scene_id)
        except KeyError:
            raise HTTPException(404, f"no scene {scene_id}")

        report = app.state.orchestrator.run(
            scene,
            body.room_id,
            image_ref=body.image_ref,
            catalogue=tuple(_item_from_in(i) for i in body.items),
            focal_point=_vec(body.focal_point) if body.focal_point else None,
            solve_time_limit_s=body.time_limit_s,
            force_phase=body.force_phase,
        )

        if report.new_scene is not None:
            app.state.store.put(report.new_scene)

        placements: list[S.PlacementOut] = []
        if report.new_scene is not None:
            try:
                placements = [
                    _placement_out(p)
                    for p in report.new_scene.room(body.room_id).placements
                ]
            except KeyError:
                placements = []

        validation = None
        if report.validation is not None:
            validation = S.ValidationOut(
                ok=report.validation.ok,
                containment_ok=report.validation.containment_ok,
                overlap_ok=report.validation.overlap_ok,
                door_swing_ok=report.validation.door_swing_ok,
                obstacle_ok=report.validation.obstacle_ok,
                violations=list(report.validation.violations),
            )

        # A blocked pipeline is a 200 with reasons -- see module docstring.
        return S.PipelineOut(
            ok=report.ok,
            scene_id=report.scene_id,
            scene_version_id=report.scene_version_id,
            execution_path=(
                str(report.capabilities.path) if report.capabilities else "UNKNOWN"
            ),
            phase=(
                str(report.effective_phase)
                if report.effective_phase
                else (str(report.phase.phase) if report.phase else None)
            ),
            phase_confidence=report.phase.confidence if report.phase else None,
            blocked_reason=report.blocked_reason,
            stages=list(report.stages),
            placements=placements,
            validation=validation,
            quote=_quote_out(report.quote) if report.quote else None,
        )

    # ---- perception (image upload) ----------------------------------

    @router.post("/perceive", response_model=S.PerceptionOut)
    async def perceive(image: UploadFile = File(...)) -> S.PerceptionOut:
        """Analyse an uploaded room photo.

        Reads the seven construction-state signals from the image and returns
        them alongside the phase they classify to. Routes to Gemini when the
        probe reports CLOUD_API; otherwise the MOCK provider answers (and says
        so in ``notes`` -- it does not actually read the pixels).

        This endpoint does not touch any scene. Use ``/scenes/{id}/perceive``
        to also record the result onto a room.
        """
        content_type = (image.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                415,
                f"unsupported image type {image.content_type!r}; "
                f"accepted: {', '.join(sorted(set(_ALLOWED_IMAGE_TYPES)))}",
            )

        raw = await image.read()
        if not raw:
            raise HTTPException(422, "uploaded file is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                413, f"image exceeds the {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )

        provider, _caps = _select_perception_provider()
        data_uri = _image_to_data_uri(raw, content_type)
        try:
            result = provider.analyse(data_uri)
        except ProviderError as exc:
            # A cloud failure is not a 500 -- fall back to MOCK so the caller
            # still gets a (clearly-labelled) answer, mirroring the pipeline.
            result = MockPerceptionProvider().analyse(data_uri)
            result = PerceptionResult(
                surfaces=result.surfaces,
                confidence=result.confidence,
                path=result.path,
                provider=result.provider,
                raw=result.raw,
                notes=result.notes + (f"cloud perception failed: {exc}",),
            )
        return _perception_out(result)

    @router.post("/estimate-scene", response_model=S.EstimateOut, status_code=201)
    async def estimate_scene(
        image: UploadFile = File(...),
        region: str = Form("GENERIC"),
        housing: str = Form("UNKNOWN"),
        room_name: str | None = Form(None),
    ) -> S.EstimateOut:
        """Create a scene from a photo when no real measurement exists yet.

        The model classifies room type and a coarse size class; a prior table
        keyed on (region, housing, type, size) supplies typical dimensions. The
        resulting room is tagged ``estimated_prior`` and carries a caveat --
        these numbers are indicative, not measured, and every quote built on
        them says so. When a real measurement arrives, it overwrites this and
        the source flips to ``measured``.
        """
        from ..perception.estimator import (
            RoomClassification,
            build_estimated_room,
            estimate_dimensions,
            parse_classification,
        )
        from ..perception.priors import HousingType, Region, RoomType, SizeBucket

        content_type = (image.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                415,
                f"unsupported image type {image.content_type!r}; "
                f"accepted: {', '.join(sorted(set(_ALLOWED_IMAGE_TYPES)))}",
            )
        raw = await image.read()
        if not raw:
            raise HTTPException(422, "uploaded file is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(413, f"image exceeds the {_MAX_IMAGE_BYTES // (1024*1024)} MB limit")

        try:
            region_enum = Region(region.strip().upper())
        except ValueError:
            raise HTTPException(422, f"unknown region {region!r}; one of {[r.value for r in Region]}")
        try:
            housing_enum = HousingType(housing.strip().upper())
        except ValueError:
            raise HTTPException(
                422, f"unknown housing {housing!r}; one of {[h.value for h in HousingType]}"
            )

        provider, caps = _select_perception_provider()
        data_uri = _image_to_data_uri(raw, content_type)

        notes: list[str] = []
        # Only the Gemini provider can classify a room from pixels. On any other
        # path we cannot, so the estimate rests on region/housing priors alone
        # with room type unknown -- still honest, just less specific.
        raw_classification: str | None = None
        classify = getattr(provider, "classify_room", None)
        if callable(classify):
            try:
                raw_classification = classify(data_uri)
            except ProviderError as exc:
                notes.append(f"room classification failed ({exc}); using region priors only")
        else:
            notes.append(
                f"provider {provider.name} cannot classify room type from an image; "
                "estimate uses region/housing priors with unknown room type"
            )

        if raw_classification is not None:
            classification = parse_classification(raw_classification)
            notes.extend(classification.notes)
        else:
            classification = RoomClassification(
                room_type=RoomType.UNKNOWN,
                size_bucket=SizeBucket.AVERAGE,
                confidence=0.2,
            )

        estimate = estimate_dimensions(
            classification, region=region_enum, housing=housing_enum
        )
        room = build_estimated_room(estimate, name=room_name)
        scene = Scene(rooms=(room,))
        app.state.store.put(scene)

        return S.EstimateOut(
            scene_id=scene.id,
            scene_version_id=scene.version_id,
            room_id=room.id,
            room_type=estimate.room_type.value,
            size_bucket=estimate.size_bucket.value,
            width_mm=estimate.dimensions.width_mm,
            depth_mm=estimate.dimensions.depth_mm,
            ceiling_mm=estimate.dimensions.ceiling_mm,
            area_m2=round(estimate.dimensions.area_m2, 2),
            dimension_source=estimate.source,
            confidence=estimate.confidence,
            basis=estimate.basis,
            caveat=estimate.caveat,
            notes=notes,
        )

    @router.post("/scenes/{scene_id}/perceive", response_model=S.PerceptionOut)
    async def perceive_and_apply(
        scene_id: str,
        room_id: str = Form(...),
        image: UploadFile = File(...),
    ) -> S.PerceptionOut:
        """Analyse a photo and record the result onto a room.

        Commits a new immutable scene version whose room carries the perceived
        surface state and its classified phase. The prior version is untouched,
        so the assessment is auditable.
        """
        try:
            scene = app.state.store.get(scene_id)
            room = scene.room(room_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        content_type = (image.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                415,
                f"unsupported image type {image.content_type!r}; "
                f"accepted: {', '.join(sorted(set(_ALLOWED_IMAGE_TYPES)))}",
            )

        raw = await image.read()
        if not raw:
            raise HTTPException(422, "uploaded file is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(
                413, f"image exceeds the {_MAX_IMAGE_BYTES // (1024 * 1024)} MB limit"
            )

        provider, _caps = _select_perception_provider()
        data_uri = _image_to_data_uri(raw, content_type)
        try:
            result = provider.analyse(data_uri, room_id=room_id)
        except ProviderError as exc:
            result = MockPerceptionProvider().analyse(data_uri, room_id=room_id)
            result = PerceptionResult(
                surfaces=result.surfaces,
                confidence=result.confidence,
                path=result.path,
                provider=result.provider,
                raw=result.raw,
                notes=result.notes + (f"cloud perception failed: {exc}",),
            )

        verdict = classify(result.surfaces)
        new_room = room.model_copy(
            update={"surfaces": result.surfaces, "phase": verdict.phase}
        )
        new_scene = scene.replace_room(
            new_room, notes=f"perception via {result.provider} -> {verdict.phase.value}"
        )
        app.state.store.put(new_scene)

        return _perception_out(result, scene_version_id=new_scene.version_id)

    # ---- floor-plan rendering ---------------------------------------

    @router.get("/scenes/{scene_id}/rooms/{room_id}/plan.svg")
    def floor_plan(
        scene_id: str,
        room_id: str,
        swings: bool = True,
        labels: bool = True,
        dimensions: bool = True,
        legend: bool = True,
    ) -> Response:
        """Render a room's current layout as a designer-grade top-down SVG.

        A view of the scene, drawn from its own geometry -- architectural walls,
        openings, door swings, dimension lines, a legend, and every placed piece
        drawn as a recognisable furniture icon at its true footprint and
        rotation. Returns image/svg+xml so a browser renders it directly.
        """
        from .render import render_floor_plan

        try:
            scene = app.state.store.get(scene_id)
            room = scene.room(room_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        svg = render_floor_plan(
            room,
            show_swings=swings,
            show_labels=labels,
            show_dimensions=dimensions,
            show_legend=legend,
            title=room.name,
        )
        return Response(content=svg, media_type="image/svg+xml")

    # ---- catalogue --------------------------------------------------

    def _catalogue_snapshot(db, sku: str):
        """Price a catalogue item from its own row.

        The catalogue is the source of truth for what a product costs: it is
        the number the operator typed, the number the picker shows, and
        therefore the number the quote must commit to. Requiring a second,
        separate price recording only created ways for the two to disagree --
        a product could be in the catalogue at 34,000 and quote as unpriced
        because nothing had written a price_history row for it.

        price_history is still consulted first, because an explicit vendor
        price recorded through /prices is a deliberate override, and it is
        still where materials (tiles, paint) get their rates -- those have no
        catalogue row at all.
        """
        from datetime import datetime, timezone

        from ..db.catalogue import CatalogueItemRow
        from ..pricing.prices import PriceSnapshot, PriceStatus

        row = db.get(CatalogueItemRow, sku)
        if row is None or row.display_price is None:
            return None
        return PriceSnapshot(
            sku=sku,
            vendor=row.vendor or "Catalogue",
            unit=Unit.PIECE,
            amount=row.display_price,
            currency=row.currency or "INR",
            observed_at=datetime.now(timezone.utc),
            status=PriceStatus.FRESH,
            age_days=0,
        )

    def _sync_opening_price(sku: str, amount, vendor: str | None, currency: str) -> bool:
        """Record a price observation when a catalogue price changes.

        The catalogue's ``display_price`` is what the picker shows; the quote
        commits to whatever ``price_history`` last recorded. Change one without
        the other and the two silently disagree -- the customer is shown 52,000
        and quoted 40,000. Recording the change keeps the shown price and the
        committed price the same number.

        Returns True when a new observation was written.
        """
        from ..pricing.prices import PriceObservation

        try:
            current = app.state.price_book.current(sku)
        except Exception:
            current = None
        if current is not None and current.amount == amount:
            return False
        app.state.price_book.record(
            PriceObservation(
                sku=sku,
                vendor=vendor or "Catalogue",
                unit=Unit.PIECE,
                amount=amount,
                currency=currency,
                source="catalogue-update",
            )
        )
        return True

    @router.post("/catalogue", status_code=201)
    def add_catalogue_item(body: S.CatalogueItemCreate) -> dict:
        """Add a product to the store catalogue (idempotent upsert by sku)."""
        from ..db.catalogue import CatalogueItemRow

        with app.state.db_sessionmaker() as db:
            row = db.get(CatalogueItemRow, body.sku)
            if row is None:
                row = CatalogueItemRow(sku=body.sku)
                db.add(row)
            row.name = body.name
            row.object_class = body.object_class
            row.description = body.description
            row.width_mm = body.width_mm
            row.depth_mm = body.depth_mm
            row.height_mm = body.height_mm
            row.display_price = body.display_price
            row.currency = body.currency
            row.vendor = body.vendor
            row.image_ref = body.image_ref
            row.style_tags = body.style_tags or {}
            row.active = 1
            db.commit()
        priced = _sync_opening_price(
            body.sku, body.display_price, body.vendor, body.currency
        )
        return {"sku": body.sku, "stored": True, "price_recorded": priced}

    @router.get("/catalogue")
    def list_catalogue(object_class: str | None = None) -> dict:
        from sqlalchemy import select as _select

        from ..db.catalogue import CatalogueItemRow

        with app.state.db_sessionmaker() as db:
            stmt = _select(CatalogueItemRow).where(CatalogueItemRow.active == 1)
            if object_class:
                stmt = stmt.where(CatalogueItemRow.object_class == object_class)
            rows = list(db.execute(stmt).scalars())
            return {
                "items": [
                    {
                        "sku": r.sku, "name": r.name, "object_class": r.object_class,
                        "width_mm": r.width_mm, "depth_mm": r.depth_mm,
                        "height_mm": r.height_mm,
                        "display_price": str(r.display_price), "currency": r.currency,
                        "vendor": r.vendor,
                        "has_image": bool((r.image_ref or "").startswith("data:")),
                        "style_tags": r.style_tags if isinstance(r.style_tags, dict) else {},
                    }
                    for r in rows
                ]
            }

    # ---- interactive photo editing ----------------------------------

    def _make_editor():
        """Gemini editor on CLOUD_API, mock otherwise -- same routing rule as
        perception, so a keyless dev box still exercises the whole loop."""
        from ..perception.editing import GeminiPhotoEditor, MockPhotoEditor

        caps = get_probe().detect()
        if caps.path.value == "CLOUD_API":
            try:
                return GeminiPhotoEditor()
            except Exception:
                return MockPhotoEditor()
        return MockPhotoEditor()

    @router.post(
        "/catalogue/upload", response_model=S.CatalogueUploadOut, status_code=201
    )
    async def upload_product(
        image: UploadFile = File(...),
        sku: str = Form(...),
        name: str = Form(...),
        object_class: str = Form(...),
        width_mm: int = Form(...),
        depth_mm: int = Form(...),
        height_mm: int = Form(...),
        display_price: str = Form(...),
        description: str = Form(""),
        vendor: str = Form(""),
        currency: str = Form("INR"),
        suggested: bool = Form(False),
        hex: str = Form(""),
        record_price: bool = Form(True),
        strip_background: bool = Form(True),
    ) -> S.CatalogueUploadOut:
        """Add a product with a photo; the photo's background is stripped to
        product-on-white before storing.

        The strip runs at UPLOAD time on purpose: the expensive edit happens
        once per product instead of on every swap, the operator sees the
        cutout immediately, and replacements become a clean two-image call.
        On the MOCK path (or a failed cutout) the original photo is stored and
        the response says so -- an unprocessed image is stated, never hidden.

        Set ``strip_background=false`` when the image is already isolated on
        white -- a generated catalogue image, or a vendor's cut-out product
        shot. Stripping an already-stripped image is a second image-generation
        call that buys nothing.
        """
        from decimal import Decimal as _D

        from ..db.catalogue import CatalogueItemRow

        content_type = (image.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(415, f"unsupported image type {image.content_type!r}")
        raw = await image.read()
        if not raw:
            raise HTTPException(422, "uploaded file is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(413, "image too large")
        if width_mm <= 0 or depth_mm <= 0 or height_mm <= 0:
            raise HTTPException(422, "dimensions must be positive millimetres")
        try:
            price = _D(display_price)
        except Exception:
            raise HTTPException(422, f"display_price {display_price!r} is not a number")

        data_uri = _image_to_data_uri(raw, content_type)
        editor = _make_editor()
        notes: list[str] = []
        processed = False
        stored_image = data_uri
        if not strip_background:
            notes.append("background strip skipped (image supplied already isolated)")
        else:
            try:
                cut = editor.cutout(data_uri)
                if cut.startswith("data:"):
                    stored_image = cut
                    processed = True
                else:
                    notes.append(
                        "background not stripped (no image model on this path); "
                        "original photo stored"
                    )
            except ProviderError as exc:
                notes.append(f"background strip failed ({exc}); original photo stored")

        tags: dict = {}
        if suggested:
            tags["suggested"] = True
        if hex:
            tags["hex"] = hex

        with app.state.db_sessionmaker() as db:
            row = db.get(CatalogueItemRow, sku)
            if row is None:
                row = CatalogueItemRow(sku=sku)
                db.add(row)
            row.name = name
            row.object_class = object_class
            row.description = description or None
            row.width_mm, row.depth_mm, row.height_mm = width_mm, depth_mm, height_mm
            row.display_price = price
            row.currency = currency
            row.vendor = vendor or None
            row.image_ref = stored_image
            row.style_tags = tags
            row.active = 1
            db.commit()

        if record_price:
            # Opening price so day-one quotes are complete, and re-recorded
            # whenever the catalogue price changes so the two never drift.
            _sync_opening_price(sku, price, vendor or "Console", currency)

        return S.CatalogueUploadOut(
            sku=sku,
            stored=True,
            image_processed=processed,
            image_url=f"/catalogue/{sku}/image",
            notes=notes,
        )

    @router.get("/catalogue/{sku}/image", include_in_schema=True)
    def catalogue_image(sku: str) -> Response:
        """Serve a product's stored image (the cutout, or the original when
        the strip could not run)."""
        import base64 as _b64

        from ..db.catalogue import CatalogueItemRow

        with app.state.db_sessionmaker() as db:
            row = db.get(CatalogueItemRow, sku)
        if row is None or not row.image_ref or not row.image_ref.startswith("data:"):
            raise HTTPException(404, f"no image stored for {sku!r}")
        header, _, b64 = row.image_ref.partition(",")
        mime = header.split(";")[0].removeprefix("data:") or "image/jpeg"
        try:
            body = _b64.b64decode(b64)
        except Exception:
            raise HTTPException(500, f"stored image for {sku!r} is corrupt")
        return Response(content=body, media_type=mime)

    @router.post("/catalogue/{sku}/deactivate")
    def deactivate_product(sku: str) -> dict:
        from ..db.catalogue import CatalogueItemRow

        with app.state.db_sessionmaker() as db:
            row = db.get(CatalogueItemRow, sku)
            if row is None:
                raise HTTPException(404, f"no product {sku!r}")
            row.active = 0
            db.commit()
        return {"sku": sku, "active": False}

    @router.post(
        "/scenes/{scene_id}/rooms/{room_id}/edit-session",
        response_model=S.EditSessionOut,
        status_code=201,
    )
    async def start_edit_session(
        scene_id: str, room_id: str, image: UploadFile = File(...)
    ) -> S.EditSessionOut:
        """Start an interactive editing session on a room photo.

        Detects every furnishable object once, up front. The response's
        detections (normalised 0-1000 boxes) are what the frontend overlays
        so clicks can be resolved."""
        from ..perception.edit_session import EditSessionService

        try:
            scene = app.state.store.get(scene_id)
            scene.room(room_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

        content_type = (image.content_type or "").lower()
        if content_type not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(415, f"unsupported image type {image.content_type!r}")
        raw = await image.read()
        if not raw:
            raise HTTPException(422, "uploaded file is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise HTTPException(413, "image too large")

        data_uri = _image_to_data_uri(raw, content_type)
        editor = _make_editor()

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=editor)
            try:
                session = svc.start(
                    scene_id=scene_id, room_id=room_id, image_ref=data_uri
                )
            except ProviderError as exc:
                raise HTTPException(502, f"object detection failed: {exc}")
            db.commit()
            return S.EditSessionOut(
                session_id=session.id,
                scene_id=scene_id,
                room_id=room_id,
                detections=[S.DetectionOut(**d) for d in session.detections],
                current_image_ref=session.original_image_ref,
            )

    @router.post("/edit-sessions/{session_id}/select", response_model=S.SelectOut)
    def select_object(session_id: str, body: S.SelectIn) -> S.SelectOut:
        """Resolve a click to an object and list catalogue offers for it.

        Offers are fit-gated against the room: items that physically cannot
        fit are still listed, last, with the measured reason -- hidden stock
        reads as a bug, an explained no is information."""
        from ..perception.edit_session import EditSessionService

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")

            room = None
            try:
                room = app.state.store.get(session.scene_id).room(session.room_id)
            except KeyError:
                pass

            sel = svc.select(
                session, body.x, body.y,
                detection_id=body.detection_id, room=room,
            )
            if sel is None:
                return S.SelectOut(hit=False)
            return S.SelectOut(
                hit=True,
                detection=S.DetectionOut(**sel.detection.to_dict()),
                offers=[
                    S.OfferOut(
                        sku=o.sku, name=o.name, object_class=o.object_class,
                        width_mm=o.width_mm, depth_mm=o.depth_mm,
                        height_mm=o.height_mm, display_price=o.display_price,
                        currency=o.currency,
                        image_url=(
                            f"/catalogue/{o.sku}/image"
                            if (o.image_ref or "").startswith("data:") else None
                        ),
                        fits_room=o.fits_room, fit_note=o.fit_note,
                        suggested=o.suggested, swatch=o.swatch,
                    )
                    for o in sel.offers
                ],
                affects=[S.DetectionOut(**d.to_dict()) for d in sel.affects],
            )

    @router.post("/edit-sessions/{session_id}/apply", response_model=S.StepOut)
    def apply_replacement(session_id: str, body: S.ApplyIn) -> S.StepOut:
        """Swap the selected object for a catalogue item. Appends a step; the
        user can call this any number of times, on any object, until happy."""
        from ..perception.edit_session import (
            EditSessionService,
            OversizeReplacement,
        )

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            room = None
            try:
                room = app.state.store.get(session.scene_id).room(session.room_id)
            except KeyError:
                pass
            try:
                step = svc.apply(
                    session, body.detection_id, body.sku,
                    expand=body.expand, redetect=body.redetect,
                    confirm_oversize=body.confirm_oversize, room=room,
                )
            except OversizeReplacement as exc:
                # 409, not 422: the request is well formed and may well be
                # what the person wants -- it just needs a decision first.
                # Returned before the image edit runs, so nothing is spent.
                raise HTTPException(409, {
                    "code": "oversize_replacement",
                    "reasons": exc.reasons,
                    "message": "This product is larger than the space it would "
                               "occupy. Continue anyway?",
                })
            except KeyError as exc:
                raise HTTPException(404, str(exc))
            except ProviderError as exc:
                raise HTTPException(502, f"image edit failed: {exc}")
            db.commit()
            return S.StepOut(
                step_id=step.id,
                detection_id=step.detection_id,
                detection_label=step.detection_label,
                replacement_sku=step.replacement_sku,
                result_image_ref=step.result_image_ref,
                swapped_skus=svc.swapped_skus(session),
                detections=[S.DetectionOut(**d) for d in session.detections],
            )

    # ---- location, questionnaire, quotation --------------------------

    @router.get("/regions")
    def list_regions() -> dict:
        """Countries the estimator can price, and those it cannot.

        Unsupported markets are listed with a reason rather than hidden: the
        rate data behind this system is Indian, and a confident quote for a
        market we cannot check would be worse than saying so."""
        from ..db.regions import COUNTRIES

        return {"countries": [
            {"code": c.code, "name": c.name, "currency": c.currency,
             "symbol": c.symbol, "supported": c.supported, "note": c.note}
            for c in COUNTRIES
        ]}

    @router.post("/edit-sessions/{session_id}/location",
                 response_model=S.LocationOut)
    def set_location(session_id: str, body: S.LocationIn) -> S.LocationOut:
        """Record where the room is.

        The city drives the currency, the market tier a quote is priced
        against, and which room-size prior applies -- so it replaces a
        dropdown the owner had to interpret with something they know."""
        from ..db.regions import country as _country
        from ..db.regions import describe, nearest_city
        from ..perception.edit_session import EditSessionService

        resolved = _country(body.country)
        if resolved is None:
            raise HTTPException(422, f"unknown country {body.country!r}")
        if not resolved.supported:
            raise HTTPException(422, resolved.note or "country not supported")

        source, distance, confident = "manual", None, True
        city = (body.city or "").strip()
        if not city:
            if body.latitude is None or body.longitude is None:
                raise HTTPException(
                    422, "provide a city, or latitude and longitude")
            fix = nearest_city(body.latitude, body.longitude)
            city, source = fix["city"], "device"
            distance, confident = fix["distance_km"], fix["confident"]

        described = describe(body.country, city)
        described.update({"source": source, "distance_km": distance,
                          "confident": confident})
        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            session.location = described
            flag_modified(session, "location")
            db.commit()
        return S.LocationOut(**described)

    @router.post("/edit-sessions/{session_id}/questionnaire")
    def set_questionnaire(session_id: str, body: S.QuestionnaireIn) -> dict:
        """Record the scope and preference answers a quote is shaped by."""
        from ..perception.edit_session import EditSessionService

        answers = {k: v for k, v in body.model_dump().items()
                   if v not in (None, "", [], {})}
        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            session.questionnaire = answers
            flag_modified(session, "questionnaire")
            db.commit()
        return {"session_id": session_id, "questionnaire": answers}

    @router.post("/edit-sessions/{session_id}/quotation",
                 response_model=S.QuotationOut)
    def session_quotation(session_id: str) -> S.QuotationOut:
        """Full quotation: contractor, DIY and hybrid options.

        Sends the before and after photographs together with everything
        already known -- the city, the questionnaire, and the real catalogue
        prices of every product swapped in. Those known prices are anchors the
        model must reproduce, not figures for it to re-estimate; only labour,
        materials and regional rates are estimated. A quote that contradicts
        the price the customer just saw in the picker is not worth having.
        """
        from datetime import datetime

        from ..perception.edit_session import EditSessionService
        from ..perception.quotation import GeminiQuoter, MockQuoter

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")

            location = dict(session.location or {})
            if not location.get("city"):
                raise HTTPException(
                    409, {"code": "location_required",
                          "message": "Set the location before quoting -- a "
                                     "price without a city is a guess."},
                )

            manifest = svc.change_manifest(session)
            if not manifest["known_products"] and not manifest["instructions"]:
                raise HTTPException(
                    409, {"code": "nothing_to_quote",
                          "message": "Nothing has been changed in this photo "
                                     "yet."},
                )

            before = session.original_image_ref
            after = svc.current_image(session)
            caps = get_probe().detect()
            quoter = GeminiQuoter() if caps.path.value == "CLOUD_API" else MockQuoter()
            result = quoter.quote(
                before, after,
                location=location,
                questionnaire=dict(session.questionnaire or {}),
                manifest=manifest,
                date_str=datetime.now().strftime("%B %d, %Y"),
            )

        return S.QuotationOut(
            status=result["status"],
            provider=result.get("provider"),
            data=result.get("data"),
            location=location,
            questionnaire=dict(session.questionnaire or {}),
            known_products=manifest["known_products"],
            instructions=manifest["instructions"],
            notes=result.get("notes", []),
        )

    @router.post("/edit-sessions/{session_id}/instruct",
                 response_model=S.InstructOut)
    def instruct_session(session_id: str, body: S.InstructIn) -> S.InstructOut:
        """Edit the photo from a typed request.

        The text is interpreted first -- a fast text call -- to work out which
        object or surface it means and what should happen to it. That decides
        whether the edit can be region-locked (most requests) or has to touch
        the whole image (a genuine scene change), and it catches the common
        case where the words describe something other than what was clicked.
        """
        from ..perception.edit_session import EditSessionService

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            try:
                step, intent = svc.instruct(
                    session, body.text,
                    detection_id=body.detection_id,
                    confirm_mismatch=body.confirm_mismatch,
                )
            except ProviderError as exc:
                raise HTTPException(502, f"edit failed: {exc}")

            labels_by_id = {
                d.get("id"): d.get("label") for d in session.detections
            }
            labels = [
                labels_by_id[i] for i in intent.target_ids if i in labels_by_id
            ]
            label = labels[0] if labels else None
            intent_out = S.IntentOut(
                target_id=intent.target_id,
                target_label=label,
                target_ids=list(intent.target_ids),
                target_labels=labels,
                operation=intent.operation,
                instruction=intent.instruction,
                confidence=intent.confidence,
                selection_matches=intent.selection_matches,
                note=intent.note,
            )

            if step is None:
                needs_confirmation = (
                    intent.is_actionable and intent.selection_matches is False
                )
                message = (
                    f"That sounds like the {label or 'something else'}, not what "
                    "you selected. Apply it there instead?"
                    if needs_confirmation
                    else (intent.note or
                          "I could not tell what to change from that. Try naming "
                          "the object and what should happen to it.")
                )
                return S.InstructOut(
                    applied=False, intent=intent_out,
                    needs_confirmation=needs_confirmation, message=message,
                )

            db.commit()
            return S.InstructOut(
                applied=True,
                intent=intent_out,
                step_id=step.id,
                result_image_ref=step.result_image_ref,
                detections=[S.DetectionOut(**d) for d in session.detections],
                swapped_skus=svc.swapped_skus(session),
            )

    @router.post("/edit-sessions/{session_id}/redetect")
    def redetect_objects(session_id: str) -> dict:
        """Re-run object detection on the current, edited image.

        Boxes updated after a swap are estimates derived from the product's
        dimensions. This measures the image instead. Previous swaps survive --
        the quote is built from the step chain, not from these boxes."""
        from ..perception.edit_session import EditSessionService

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            try:
                count, notes = svc.redetect(session)
            except ProviderError as exc:
                raise HTTPException(502, f"detection failed: {exc}")
            db.commit()
            return {
                "detections": session.detections,
                "count": count,
                "current_image_ref": svc.current_image(session),
                "swapped_skus": svc.swapped_skus(session),
                "notes": notes,
            }

    @router.post("/edit-sessions/{session_id}/undo")
    def undo_step(session_id: str) -> dict:
        from ..perception.edit_session import EditSessionService

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            image = svc.undo(session)
            db.commit()
            return {
                "current_image_ref": image,
                "swapped_skus": svc.swapped_skus(session),
                "detections": session.detections,
            }

    @router.get("/edit-sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        from ..perception.edit_session import EditSessionService

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")
            return {
                "session_id": session.id,
                "detections": session.detections,
                "current_image_ref": svc.current_image(session),
                "swapped_skus": svc.swapped_skus(session),
                "steps": [
                    {
                        "step_id": s.id,
                        "detection_label": s.detection_label,
                        "sku": s.replacement_sku,
                    }
                    for s in session.steps
                ],
            }

    @router.post("/edit-sessions/{session_id}/quote", response_model=S.QuoteOut)
    def quote_session(session_id: str) -> S.QuoteOut:
        """Price the CURRENT image: one line per swapped-in catalogue item,
        priced through the same frozen-snapshot machinery as everything else.

        Prices come from price_history keyed by sku. A swapped item with no
        recorded price surfaces as UNPRICED rather than using its display
        price -- the picker's display price is marketing, the quote's price is
        a commitment, and conflating them is how the two drift apart."""
        from decimal import Decimal as _D

        from ..db.catalogue import CatalogueItemRow
        from ..perception.edit_session import EditSessionService
        from ..pricing.engine import BOQLine, Quote
        from ..pricing.takeoff import TakeoffLine

        with app.state.db_sessionmaker() as db:
            svc = EditSessionService(db, editor=_make_editor())
            session = svc.get(session_id)
            if session is None:
                raise HTTPException(404, f"no session {session_id}")

            swaps = svc.swapped_skus(session)
            lines: list[BOQLine] = []
            for det_id, sku in swaps.items():
                row = db.get(CatalogueItemRow, sku)
                name = row.name if row else sku
                # An explicitly recorded vendor price wins; otherwise the
                # product's own catalogue price is used, so anything in the
                # catalogue is always quotable.
                snap = app.state.price_book.snapshot(sku)
                if snap.amount is None:
                    snap = _catalogue_snapshot(db, sku) or snap
                lines.append(
                    BOQLine(
                        sku=sku,
                        description=name,
                        quantity=_D(1),
                        unit=Unit.PIECE,
                        price=snap,
                        basis=f"swapped into photo (replaced detection {det_id})",
                        room_id=session.room_id,
                    )
                )
            quote = Quote(
                scene_id=session.scene_id,
                scene_version_id=session.scene_id,  # session-scoped quote
                lines=tuple(lines),
            )
            return _quote_out(quote)

    # ---- operator console -------------------------------------------

    @router.get("/ui", include_in_schema=False)
    def console() -> Response:
        """The pipeline console: a single-file UI that walks the whole flow --
        photo upload, estimate, perceive, layout+quote, floor plan, and the
        click-to-swap editing loop -- against this same server, so there is no
        CORS and no build step. Open http://localhost:8000/ui in a browser."""
        from pathlib import Path

        page = Path(__file__).parent / "static" / "ui.html"
        if not page.exists():
            raise HTTPException(404, "console not bundled in this install")
        return Response(content=page.read_text(encoding="utf-8"), media_type="text/html")

    @router.get("/admin", include_in_schema=False)
    def product_console() -> Response:
        """The product console: upload products with photos (backgrounds are
        stripped to product-on-white on ingest), browse inventory, deactivate.
        Open http://localhost:8000/admin in a browser."""
        from pathlib import Path

        page = Path(__file__).parent / "static" / "admin.html"
        if not page.exists():
            raise HTTPException(404, "product console not bundled in this install")
        return Response(content=page.read_text(encoding="utf-8"), media_type="text/html")

    app.include_router(router)
    return app


app = create_app()