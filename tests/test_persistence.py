"""Durable storage.

The bug these guard against: a product uploaded through the console vanished
on restart, because the catalogue lived in an in-process SQLite database and
prices and scenes lived in plain dictionaries. With DATABASE_URL configured,
everything a user creates must outlive the process.

Each test builds an app, writes through the API, throws the app away, and
builds a *fresh* app against the same database -- the closest thing to a real
restart that a test can do.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from interior_ai.api.app import create_app


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = f"sqlite+pysqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    # Throwaway database per test: creating the schema directly is honest
    # here, where running Alembic would add nothing but time. Real databases
    # are owned by migrations, which is why this must be opted into.
    monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
    return url


def _photo(size=(900, 700)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "tan").save(buf, "JPEG")
    return buf.getvalue()


class TestDurableStorage:
    def test_health_reports_durability(self, db_url):
        body = TestClient(create_app()).get("/health").json()
        assert body["persistent"] is True
        assert body["database"] == "reachable"

    def test_health_admits_when_storage_is_ephemeral(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        body = TestClient(create_app()).get("/health").json()
        assert body["persistent"] is False
        assert body["storage"] == "in-memory"

    def test_catalogue_survives_restart(self, db_url):
        first = TestClient(create_app())
        assert first.post("/catalogue/upload", data={
            "sku": "SOFA-KEEP", "name": "Keeper", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "44000",
        }, files={"image": ("s.jpg", _photo(), "image/jpeg")}).status_code == 201

        restarted = TestClient(create_app())
        skus = [i["sku"] for i in restarted.get("/catalogue").json()["items"]]
        assert "SOFA-KEEP" in skus

    def test_product_image_survives_restart(self, db_url):
        TestClient(create_app()).post("/catalogue/upload", data={
            "sku": "SOFA-IMG", "name": "Imaged", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "1000",
        }, files={"image": ("s.jpg", _photo(), "image/jpeg")})

        got = TestClient(create_app()).get("/catalogue/SOFA-IMG/image")
        assert got.status_code == 200
        assert got.headers["content-type"].startswith("image/")

    def test_prices_survive_restart(self, db_url):
        TestClient(create_app()).post("/prices", json={
            "sku": "TILE-STD", "vendor": "Local", "unit": "sqm", "amount": "900",
        })
        got = TestClient(create_app()).get("/prices/TILE-STD")
        assert got.status_code == 200
        assert got.json()["amount"] == "900.00"

    def test_scenes_survive_restart(self, db_url):
        created = TestClient(create_app()).post("/estimate-scene", data={
            "region": "IN_METRO", "housing": "FLAT_2BHK",
        }, files={"image": ("r.jpg", _photo(), "image/jpeg")}).json()

        restarted = TestClient(create_app())
        got = restarted.get(f"/scenes/{created['scene_id']}")
        assert got.status_code == 200
        assert got.json()["rooms"][0]["id"] == created["room_id"]

    def test_scene_versions_survive_restart(self, db_url):
        """Version history is the audit trail behind every quote; losing it
        would make old quotes unreproducible."""
        client = TestClient(create_app())
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 4000, "y": 0},
                        {"x": 4000, "y": 3000}, {"x": 0, "y": 3000}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        sid, rid = scene["scene_id"], scene["rooms"][0]["id"]
        client.post("/prices", json={"sku": "SOFA-V", "vendor": "V", "unit": "piece", "amount": "40000"})
        client.post(f"/scenes/{sid}/pipeline", json={
            "room_id": rid, "force_phase": "STYLING_RESTRUCTURE", "time_limit_s": 15,
            "items": [{"sku": "SOFA-V", "name": "Sofa", "object_class": "sofa",
                       "footprint": {"width_mm": 2000, "depth_mm": 850, "height_mm": 800}}],
        })

        versions = TestClient(create_app()).get(f"/scenes/{sid}/versions").json()
        assert len(versions["versions"]) >= 2

    def test_edit_session_survives_restart(self, db_url):
        client = TestClient(create_app())
        client.post("/catalogue/upload", data={
            "sku": "SOFA-ES", "name": "ES", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "40000",
        }, files={"image": ("s.jpg", _photo(), "image/jpeg")})
        est = client.post("/estimate-scene", data={
            "region": "IN_METRO", "housing": "FLAT_2BHK",
        }, files={"image": ("r.jpg", _photo(), "image/jpeg")}).json()
        session = client.post(
            f"/scenes/{est['scene_id']}/rooms/{est['room_id']}/edit-session",
            files={"image": ("r.jpg", _photo(), "image/jpeg")},
        ).json()

        got = TestClient(create_app()).get(f"/edit-sessions/{session['session_id']}")
        assert got.status_code == 200
        assert len(got.json()["detections"]) == len(session["detections"])

    def test_quote_after_restart_uses_persisted_prices(self, db_url):
        """The end that matters: a quote produced after a restart is complete,
        because the prices behind it were durable."""
        client = TestClient(create_app())
        for sku, unit, amount in [
            ("TILE-STD", "sqm", "900"), ("ADHESIVE-STD", "kg", "30"),
            ("GROUT-STD", "kg", "45"), ("PAINT-STD", "litre", "420"),
            ("PRIMER-STD", "litre", "280"), ("PUTTY-STD", "kg", "35"),
        ]:
            client.post("/prices", json={"sku": sku, "vendor": "L", "unit": unit, "amount": amount})
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 4000, "y": 0},
                        {"x": 4000, "y": 3000}, {"x": 0, "y": 3000}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()

        restarted = TestClient(create_app())
        quote = restarted.post(f"/scenes/{scene['scene_id']}/quote").json()
        assert quote["is_complete"] is True
        assert float(quote["total"]) > 0


class TestNeonUrlHandling:
    """Neon hands out plain postgresql:// URLs and suspends idle computes."""

    def test_postgres_urls_get_a_driver(self):
        from interior_ai.db.repository import make_engine

        for raw in ["postgresql://u:p@host/db", "postgres://u:p@host/db"]:
            engine = make_engine(raw)
            assert engine.url.drivername == "postgresql+psycopg"

    def test_ssl_is_required_for_postgres(self):
        from interior_ai.db.repository import make_engine

        engine = make_engine("postgresql://u:p@host/db")
        assert engine.url.query.get("sslmode") == "require"

    def test_existing_sslmode_is_respected(self):
        from interior_ai.db.repository import make_engine

        engine = make_engine("postgresql://u:p@host/db?sslmode=verify-full")
        assert engine.url.query.get("sslmode") == "verify-full"

    def test_pre_ping_enabled_for_serverless(self):
        """Neon drops idle connections; without pre-ping the first request
        after a pause fails with a confusing socket error."""
        from interior_ai.db.repository import make_engine

        engine = make_engine("postgresql://u:p@host/db")
        assert engine.pool._pre_ping is True

    def test_sqlite_is_left_alone(self):
        from interior_ai.db.repository import make_engine

        engine = make_engine("sqlite+pysqlite:///:memory:")
        assert engine.url.drivername == "sqlite+pysqlite"
        assert "sslmode" not in engine.url.query