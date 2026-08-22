"""Shared fixtures."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from interior_ai.core.enums import ObjectClass, OpeningKind, SwingDirection, Unit
from interior_ai.core.scene import (
    CatalogueItem,
    Footprint,
    Obstacle,
    Opening,
    Room,
    Scene,
    SurfaceState,
    Vec2,
)
from interior_ai.perception.probe import CapabilityProbe, GpuInfo
from interior_ai.pricing.prices import PriceBook, PriceObservation

# --------------------------------------------------------------------------
# Isolation from local configuration.
#
# The application loads .env on import so the server needs no shell exports.
# That convenience is a hazard here: a developer whose .env points
# DATABASE_URL at a real Neon database would have this suite write to live
# data, and a real GEMINI_API_KEY would turn unit tests into billable network
# calls. So every test starts with none of it, which is also the state a fresh
# contributor's machine is in. Tests that need a database set DATABASE_URL
# themselves (see test_persistence.py).
# --------------------------------------------------------------------------

_ISOLATED_ENV = (
    "DATABASE_URL",
    "AUTO_CREATE_SCHEMA",
    "GEMINI_API_KEY",
    "CLOUD_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_IMAGE_MODEL",
    "GEMINI_DETECT_MODEL",
    "GEMINI_ENDPOINT",
    "GEMINI_TIMEOUT_S",
    "GEMINI_EDIT_TIMEOUT_S",
    "EDIT_OUTPUT_FORMAT",
    "FORCE_EXECUTION_PATH",
    "PROBE_SKIP_HEALTHCHECK",
    "BASIS_ASCII",
)


def pytest_configure(config) -> None:
    """Clear host configuration before collection.

    This must happen here, not only in a fixture: some test modules construct
    an application (and therefore a database engine) at import time, which is
    *before* any fixture runs. With a developer's .env present, collection
    itself would try to reach their real database and error out.
    """
    os.environ["INTERIOR_AI_SKIP_DOTENV"] = "1"
    for name in _ISOLATED_ENV:
        os.environ.pop(name, None)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Keep each test clear of host configuration.

    ``pytest_configure`` handles the session; this guards against a test that
    sets one of these and would otherwise leak into the next.
    """
    for name in _ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INTERIOR_AI_SKIP_DOTENV", "1")
    yield


def pytest_report_header(config) -> str:
    """Make the isolation visible -- silent isolation confuses anyone
    debugging why a test cannot see their database."""
    leaked = [name for name in _ISOLATED_ENV if os.getenv(name)]
    detail = f" (host had {', '.join(leaked)})" if leaked else ""
    return f"interior_ai: environment isolated from .env{detail}"


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


def rect(w: int, d: int) -> tuple[Vec2, ...]:
    return (Vec2(x=0, y=0), Vec2(x=w, y=0), Vec2(x=w, y=d), Vec2(x=0, y=d))


@pytest.fixture
def bare_room() -> Room:
    """5m x 4m, no openings or obstacles."""
    return Room(name="Bare", polygon=rect(5000, 4000), ceiling_height_mm=2700)


@pytest.fixture
def door_room() -> Room:
    """5m x 4m with an inward-swinging door on the bottom wall."""
    return Room(
        name="WithDoor",
        polygon=rect(5000, 4000),
        ceiling_height_mm=2700,
        openings=(
            Opening(
                kind=OpeningKind.DOOR,
                centre=Vec2(x=1000, y=0),
                width_mm=900,
                height_mm=2100,
                wall_index=0,
                swing=SwingDirection.INWARD,
            ),
        ),
    )


@pytest.fixture
def windowed_room() -> Room:
    """4m x 3m with one door and one window -- for wall-area arithmetic."""
    return Room(
        name="Windowed",
        polygon=rect(4000, 3000),
        ceiling_height_mm=2700,
        openings=(
            Opening(
                kind=OpeningKind.DOOR,
                centre=Vec2(x=1000, y=0),
                width_mm=900,
                height_mm=2100,
                wall_index=0,
                swing=SwingDirection.INWARD,
            ),
            Opening(
                kind=OpeningKind.WINDOW,
                centre=Vec2(x=3000, y=0),
                width_mm=1500,
                height_mm=1200,
                wall_index=0,
                sill_height_mm=900,
            ),
        ),
    )


@pytest.fixture
def obstacle_room() -> Room:
    return Room(
        name="Obstructed",
        polygon=rect(4000, 3000),
        ceiling_height_mm=2700,
        obstacles=(
            Obstacle(label="column", origin=Vec2(x=1800, y=1300), width_mm=400, depth_mm=400),
        ),
    )


@pytest.fixture
def sofa() -> CatalogueItem:
    return CatalogueItem(
        sku="SOFA-3S",
        name="Three-seat sofa",
        object_class=ObjectClass.SOFA,
        footprint=Footprint(width_mm=2200, depth_mm=900, height_mm=800),
        clearance_front_mm=450,
    )


@pytest.fixture
def coffee_table() -> CatalogueItem:
    return CatalogueItem(
        sku="CT-01",
        name="Coffee table",
        object_class=ObjectClass.COFFEE_TABLE,
        footprint=Footprint(width_mm=1100, depth_mm=600, height_mm=400),
    )


@pytest.fixture
def tv_unit() -> CatalogueItem:
    return CatalogueItem(
        sku="TV-01",
        name="TV unit",
        object_class=ObjectClass.TV_UNIT,
        footprint=Footprint(width_mm=1800, depth_mm=450, height_mm=500),
        requires_wall=True,
    )


@pytest.fixture
def wardrobe() -> CatalogueItem:
    return CatalogueItem(
        sku="WR-01",
        name="Wardrobe",
        object_class=ObjectClass.WARDROBE,
        footprint=Footprint(width_mm=1200, depth_mm=600, height_mm=2200),
        requires_wall=True,
        clearance_front_mm=700,
    )


@pytest.fixture
def finished_surfaces() -> SurfaceState:
    return SurfaceState(
        walls_painted="yes",
        flooring_installed="yes",
        ceiling_finished="yes",
        electrical_terminated="yes",
        plumbing_terminated="yes",
        carpentry_installed="yes",
        furniture_present="no",
    )


@pytest.fixture
def scene(bare_room: Room) -> Scene:
    return Scene(rooms=(bare_room,))


@pytest.fixture
def price_book(now: datetime) -> PriceBook:
    book = PriceBook()
    book.record_many(
        [
            PriceObservation(
                sku="TILE-STD", vendor="Kajaria", unit=Unit.SQM,
                amount=Decimal("850"), observed_at=now - timedelta(days=1),
            ),
            PriceObservation(
                sku="ADHESIVE-STD", vendor="Roff", unit=Unit.KG,
                amount=Decimal("28"), observed_at=now - timedelta(days=2),
            ),
            PriceObservation(
                sku="PAINT-STD", vendor="Asian Paints", unit=Unit.LITRE,
                amount=Decimal("420"), observed_at=now - timedelta(days=30),
            ),
        ]
    )
    return book


@pytest.fixture
def mock_probe():
    """Probe pinned to MOCK: no GPU, empty weights dir, unreachable cloud."""
    return CapabilityProbe(
        model_dir=tempfile.mkdtemp(),
        gpu_detector=lambda: GpuInfo(present=False),
        health_check=lambda key: False,
    )


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "FORCE_EXECUTION_PATH",
        "GEMINI_API_KEY",
        "CLOUD_API_KEY",
        "PROBE_SKIP_HEALTHCHECK",
    ):
        monkeypatch.delenv(var, raising=False)