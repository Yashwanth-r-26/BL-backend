"""HTTP gateway and persistence."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from interior_ai.api.app import SceneStore, create_app
from interior_ai.core.enums import Unit
from interior_ai.core.scene import Room, Scene, Vec2
from interior_ai.db.repository import (
    PriceRepository,
    QuoteRepository,
    SceneRepository,
    SqlPriceBook,
    create_all,
    make_engine,
    make_session_factory,
)
from interior_ai.pricing.engine import PricingEngine
from interior_ai.pricing.prices import PriceObservation

ROOM_BODY = {
    "name": "Living",
    "polygon": [
        {"x": 0, "y": 0}, {"x": 5000, "y": 0},
        {"x": 5000, "y": 4000}, {"x": 0, "y": 4000},
    ],
    "ceiling_height_mm": 2700,
    "openings": [
        {
            "kind": "door", "centre": {"x": 700, "y": 0},
            "width_mm": 900, "height_mm": 2100, "wall_index": 0, "swing": "inward",
        }
    ],
    "surfaces": {
        "walls_painted": "yes", "flooring_installed": "yes", "ceiling_finished": "yes",
        "electrical_terminated": "yes", "plumbing_terminated": "yes",
        "carpentry_installed": "yes", "furniture_present": "no",
    },
}

SOFA_BODY = {
    "sku": "SOFA-3S", "name": "Sofa", "object_class": "sofa",
    "footprint": {"width_mm": 2200, "depth_mm": 900, "height_mm": 800},
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(store=SceneStore()))


@pytest.fixture
def scene_ids(client: TestClient) -> tuple[str, str]:
    resp = client.post("/scenes", json={"rooms": [ROOM_BODY]})
    body = resp.json()
    return body["scene_id"], body["rooms"][0]["id"]


class TestBasicEndpoints:
    def test_health(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_capabilities_reports_a_path(self, client):
        body = client.get("/capabilities").json()
        assert body["path"] in ("LOCAL_FULL", "LOCAL_LIGHT", "CLOUD_API", "MOCK")
        assert body["reasons"]

    def test_create_scene(self, client):
        resp = client.post("/scenes", json={"rooms": [ROOM_BODY]})
        assert resp.status_code == 201
        assert resp.json()["version"] == 1

    def test_get_scene(self, client, scene_ids):
        scene_id, _ = scene_ids
        body = client.get(f"/scenes/{scene_id}").json()
        assert body["scene_id"] == scene_id
        assert len(body["rooms"]) == 1

    def test_missing_scene_404s(self, client):
        assert client.get("/scenes/nope").status_code == 404

    def test_invalid_polygon_rejected(self, client):
        bad = {**ROOM_BODY, "polygon": [{"x": 0, "y": 0}, {"x": 100, "y": 0}]}
        assert client.post("/scenes", json={"rooms": [bad]}).status_code == 422


class TestFitEndpoint:
    def test_rejection_carries_a_measurement(self, client, scene_ids):
        scene_id, room_id = scene_ids
        resp = client.post(
            f"/scenes/{scene_id}/fit",
            json={
                "room_id": room_id,
                "item": {**SOFA_BODY, "footprint": {"width_mm": 9000, "depth_mm": 900, "height_mm": 800}},
                "origin": {"x": 0, "y": 0},
            },
        )
        body = resp.json()
        assert not body["ok"]
        assert body["rejections"][0]["overage_mm"] == 4000

    def test_valid_placement_returns_bounds(self, client, scene_ids):
        scene_id, room_id = scene_ids
        body = client.post(
            f"/scenes/{scene_id}/fit",
            json={"room_id": room_id, "item": SOFA_BODY, "origin": {"x": 2000, "y": 2000}},
        ).json()
        assert body["ok"]
        assert body["placement_bounds"] == [2000, 2000, 4200, 2900]

    def test_unknown_room_404s(self, client, scene_ids):
        scene_id, _ = scene_ids
        resp = client.post(
            f"/scenes/{scene_id}/fit",
            json={"room_id": "nope", "item": SOFA_BODY, "origin": {"x": 0, "y": 0}},
        )
        assert resp.status_code == 404


class TestPhaseEndpoint:
    def test_partial_blocks_and_names_the_signal(self, client):
        body = client.post(
            "/phase",
            json={
                "surfaces": {
                    "walls_painted": "partial", "flooring_installed": "yes",
                    "ceiling_finished": "yes", "electrical_terminated": "yes",
                    "plumbing_terminated": "yes", "carpentry_installed": "yes",
                    "furniture_present": "no",
                }
            },
        ).json()
        assert body["phase"] == "SURFACE_FINISHING"
        assert "walls_painted" in body["blocking_signals"]

    def test_unknown_flags_for_review(self, client):
        body = client.post("/phase", json={"surfaces": {}}).json()
        assert body["needs_review"]


class TestPricingEndpoints:
    def test_record_and_read_price(self, client):
        client.post("/prices", json={"sku": "TILE-STD", "vendor": "K", "unit": "sqm", "amount": "900"})
        body = client.get("/prices/TILE-STD").json()
        assert body["status"] == "fresh"
        assert body["history_depth"] == 1

    def test_history_accumulates(self, client):
        for amount in ("800", "900", "950"):
            client.post("/prices", json={"sku": "X", "vendor": "V", "unit": "sqm", "amount": amount})
        assert client.get("/prices/X").json()["history_depth"] == 3

    def test_unknown_sku_is_unpriced_not_404(self, client):
        body = client.get("/prices/NOPE").json()
        assert body["status"] == "unpriced"

    def test_bad_unit_rejected(self, client):
        resp = client.post("/prices", json={"sku": "X", "vendor": "V", "unit": "furlongs", "amount": "1"})
        assert resp.status_code == 422

    def test_quote_reports_incompleteness(self, client, scene_ids):
        scene_id, _ = scene_ids
        client.post("/prices", json={"sku": "TILE-STD", "vendor": "K", "unit": "sqm", "amount": "900"})
        body = client.post(f"/scenes/{scene_id}/quote").json()
        assert not body["is_complete"]
        assert body["warnings"]
        assert Decimal(body["total"]) > 0


@pytest.mark.slow
class TestRestructureEndpoint:
    def test_solves_and_validates(self, client, scene_ids):
        scene_id, room_id = scene_ids
        body = client.post(
            f"/scenes/{scene_id}/restructure",
            json={"room_id": room_id, "items": [SOFA_BODY], "time_limit_s": 20},
        ).json()
        assert body["ok"]
        assert body["validation"]["ok"]
        assert body["scene_version_id"]

    def test_creates_a_new_version(self, client, scene_ids):
        scene_id, room_id = scene_ids
        client.post(
            f"/scenes/{scene_id}/restructure",
            json={"room_id": room_id, "items": [SOFA_BODY], "time_limit_s": 20},
        )
        versions = client.get(f"/scenes/{scene_id}/versions").json()["versions"]
        assert [v["version"] for v in versions] == [1, 2]
        assert versions[1]["parent_version_id"] == versions[0]["version_id"]


@pytest.mark.slow
class TestPipelineEndpoint:
    def test_full_run(self, client, scene_ids):
        scene_id, room_id = scene_ids
        client.post("/prices", json={"sku": "SOFA-3S", "vendor": "V", "unit": "piece", "amount": "45000"})
        body = client.post(
            f"/scenes/{scene_id}/pipeline",
            json={"room_id": room_id, "items": [SOFA_BODY], "time_limit_s": 20},
        ).json()
        assert body["ok"]
        assert body["phase"] == "STYLING_RESTRUCTURE"
        assert body["placements"]
        assert body["validation"]["ok"]

    def test_blocked_pipeline_returns_200_with_reasons(self, client):
        """'Not ready for furniture' is a successful analysis with a negative
        answer, not an HTTP error."""
        unfinished = {
            **ROOM_BODY,
            "surfaces": {
                "walls_painted": "no", "flooring_installed": "no", "ceiling_finished": "no",
                "electrical_terminated": "no", "plumbing_terminated": "no",
                "carpentry_installed": "no", "furniture_present": "no",
            },
        }
        created = client.post("/scenes", json={"rooms": [unfinished]}).json()
        resp = client.post(
            f"/scenes/{created['scene_id']}/pipeline",
            json={"room_id": created["rooms"][0]["id"], "items": [SOFA_BODY]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["ok"]
        assert body["blocked_reason"]
        assert body["quote"] is not None


import struct
import zlib


def _tiny_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


class TestPerceptionUpload:
    def test_mock_path_reads_a_real_upload(self, client):
        """On MOCK the pixels are not read, but the upload plumbing still works
        end to end and the response is honest about being a mock."""
        resp = client.post("/perceive", files={"image": ("room.png", _tiny_png(), "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["execution_path"] == "MOCK"
        assert body["phase"] in ("SURFACE_FINISHING", "FIXTURES_CARPENTRY", "STYLING_RESTRUCTURE")
        assert any("MOCK" in n for n in body["notes"])

    def test_non_image_rejected(self, client):
        resp = client.post("/perceive", files={"image": ("x.txt", b"hello", "text/plain")})
        assert resp.status_code == 415

    def test_empty_file_rejected(self, client):
        resp = client.post("/perceive", files={"image": ("e.png", b"", "image/png")})
        assert resp.status_code == 422

    def test_gemini_path_sends_image_and_prompt(self, monkeypatch):
        """The image must reach the provider as inline_data with the
        seven-question prompt alongside it."""
        import interior_ai.api.app as appmod
        from interior_ai.perception.probe import CapabilityProbe, GpuInfo
        from interior_ai.providers.gemini import GeminiPerceptionProvider

        captured = {}

        def transport(payload):
            parts = payload["contents"][0]["parts"]
            captured["has_image"] = any("inline_data" in p for p in parts)
            captured["has_prompt"] = any("walls_painted" in p.get("text", "") for p in parts)
            return {
                "candidates": [
                    {"content": {"parts": [{"text": (
                        '{"walls_painted":"yes","flooring_installed":"yes",'
                        '"ceiling_finished":"yes","electrical_terminated":"yes",'
                        '"plumbing_terminated":"yes","carpentry_installed":"yes",'
                        '"furniture_present":"no"}'
                    )}]}}
                ]
            }

        def fake_factory():
            probe = CapabilityProbe(
                model_dir="/tmp",
                gpu_detector=lambda: GpuInfo(present=False),
                health_check=lambda k: True,
            )
            monkeypatch.setenv("GEMINI_API_KEY", "fake")
            caps = probe.detect()
            return GeminiPerceptionProvider(api_key="fake", transport=transport), caps

        monkeypatch.setattr(appmod, "_select_perception_provider", fake_factory)
        app = appmod.create_app(store=appmod.SceneStore())
        client = TestClient(app)

        resp = client.post("/perceive", files={"image": ("room.jpg", _tiny_png(), "image/jpeg")})
        assert resp.status_code == 200
        assert captured["has_image"] is True
        assert captured["has_prompt"] is True
        body = resp.json()
        assert body["provider"] == "gemini-perception"
        assert body["phase"] == "STYLING_RESTRUCTURE"

    def test_apply_to_scene_commits_a_new_version(self, client, scene_ids):
        scene_id, room_id = scene_ids
        resp = client.post(
            f"/scenes/{scene_id}/perceive",
            data={"room_id": room_id},
            files={"image": ("room.png", _tiny_png(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["scene_version_id"]
        versions = client.get(f"/scenes/{scene_id}/versions").json()["versions"]
        assert [v["version"] for v in versions] == [1, 2]

    def test_apply_to_unknown_scene_404s(self, client):
        resp = client.post(
            "/scenes/nope/perceive",
            data={"room_id": "nope"},
            files={"image": ("room.png", _tiny_png(), "image/png")},
        )
        assert resp.status_code == 404


class TestEstimateScene:
    def test_estimate_creates_a_scene_with_flagged_dimensions(self, client):
        resp = client.post(
            "/estimate-scene",
            data={"region": "IN_METRO", "housing": "FLAT_2BHK"},
            files={"image": ("living.jpg", _tiny_png(), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["dimension_source"] == "estimated_prior"
        assert body["width_mm"] > 0 and body["depth_mm"] > 0
        assert "ESTIMATED" in body["caveat"]

    def test_estimated_scene_runs_the_pipeline(self, client):
        est = client.post(
            "/estimate-scene",
            data={"region": "IN_METRO", "housing": "FLAT_2BHK"},
            files={"image": ("living.jpg", _tiny_png(), "image/jpeg")},
        ).json()
        client.post("/prices", json={"sku": "SOFA-3S", "vendor": "V", "unit": "piece", "amount": "45000"})
        resp = client.post(
            f"/scenes/{est['scene_id']}/pipeline",
            json={
                "room_id": est["room_id"],
                "items": [{"sku": "SOFA-3S", "name": "Sofa", "object_class": "sofa",
                           "footprint": {"width_mm": 2000, "depth_mm": 850, "height_mm": 800}}],
                "force_phase": "STYLING_RESTRUCTURE",
                "time_limit_s": 15,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["quote"] is not None

    def test_bad_region_rejected(self, client):
        resp = client.post(
            "/estimate-scene",
            data={"region": "MARS", "housing": "FLAT_2BHK"},
            files={"image": ("x.jpg", _tiny_png(), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_mock_path_notes_it_cannot_classify(self, client):
        """On MOCK there is no room classifier, so the estimate leans on region
        priors with room type unknown -- and says so."""
        body = client.post(
            "/estimate-scene",
            data={"region": "IN_METRO", "housing": "FLAT_2BHK"},
            files={"image": ("x.jpg", _tiny_png(), "image/jpeg")},
        ).json()
        assert any("cannot classify" in n for n in body["notes"])

    def test_estimated_room_has_openings(self, client):
        """Estimated rooms carry a typical door + window so paint math deducts
        them instead of quoting solid walls."""
        est = client.post(
            "/estimate-scene",
            data={"region": "IN_METRO", "housing": "FLAT_2BHK"},
            files={"image": ("x.jpg", _tiny_png(), "image/jpeg")},
        ).json()
        client.post("/prices", json={"sku": "PAINT-STD", "vendor": "V", "unit": "litre", "amount": "420"})
        quote = client.post(f"/scenes/{est['scene_id']}/quote").json()
        paint = next(l for l in quote["lines"] if l["sku"] == "PAINT-STD")
        assert "of openings" in paint["basis"]
        assert "less 0.00" not in paint["basis"]  # openings actually deducted


class TestFloorPlanEndpoint:
    def test_returns_svg(self, client, scene_ids):
        scene_id, room_id = scene_ids
        resp = client.get(f"/scenes/{scene_id}/rooms/{room_id}/plan.svg")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert resp.text.startswith("<svg")

    def test_unknown_room_404s(self, client, scene_ids):
        scene_id, _ = scene_ids
        resp = client.get(f"/scenes/{scene_id}/rooms/nope/plan.svg")
        assert resp.status_code == 404

    def test_renders_placements_after_pipeline(self, client, scene_ids):
        scene_id, room_id = scene_ids
        client.post("/prices", json={"sku": "SOFA-3S", "vendor": "V", "unit": "piece", "amount": "45000"})
        client.post(
            f"/scenes/{scene_id}/pipeline",
            json={
                "room_id": room_id,
                "items": [{"sku": "SOFA-3S", "name": "Sofa", "object_class": "sofa",
                           "footprint": {"width_mm": 2000, "depth_mm": 850, "height_mm": 800}}],
                "force_phase": "STYLING_RESTRUCTURE",
                "time_limit_s": 15,
            },
        )
        # The pipeline committed a new version; the latest scene has the sofa.
        resp = client.get(f"/scenes/{scene_id}/rooms/{room_id}/plan.svg")
        assert resp.status_code == 200


@pytest.fixture
def session():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    create_all(engine)
    return make_session_factory(engine)()


class TestSceneRepository:
    def test_round_trip_is_exact(self, session, scene):
        repo = SceneRepository(session)
        repo.save(scene)
        session.commit()
        assert repo.get_version(scene.version_id) == scene

    def test_versions_accumulate(self, session, scene):
        repo = SceneRepository(session)
        v1 = scene
        v2 = v1.next_version(notes="second")
        v3 = v2.next_version(notes="third")
        for s in (v1, v2, v3):
            repo.save(s)
        session.commit()
        assert repo.get_latest(scene.id).version == 3
        assert len(repo.list_versions(scene.id)) == 3

    def test_lineage_walks_to_root(self, session, scene):
        repo = SceneRepository(session)
        v2 = scene.next_version()
        v3 = v2.next_version()
        for s in (scene, v2, v3):
            repo.save(s)
        session.commit()
        assert [r.version for r in repo.lineage(v3.version_id)] == [3, 2, 1]

    def test_resave_is_idempotent(self, session, scene):
        repo = SceneRepository(session)
        repo.save(scene)
        repo.save(scene)
        session.commit()
        assert len(repo.list_versions(scene.id)) == 1

    def test_unknown_version_returns_none(self, session):
        assert SceneRepository(session).get_version("nope") is None


class TestSqlPriceBook:
    def test_projection_tracks_latest(self, session, now):
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("100"), observed_at=now - timedelta(days=5)))
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("120"), observed_at=now))
        session.commit()
        assert book.current("X").amount == Decimal("120.00")

    def test_backfill_does_not_clobber(self, session, now):
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("120"), observed_at=now))
        book.record(PriceObservation(sku="X", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("70"), observed_at=now - timedelta(days=90)))
        session.commit()
        assert book.current("X").amount == Decimal("120.00")
        assert len(book.history_for("X")) == 2

    def test_staleness_survives_the_database(self, session, now):
        """SQLite strips tzinfo; naive-vs-aware arithmetic would raise inside
        the code that decides whether money is trustworthy."""
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="OLD", vendor="V", unit=Unit.LITRE,
                                     amount=Decimal("420"), observed_at=now - timedelta(days=30)))
        session.commit()
        snap = book.snapshot("OLD")
        assert snap.status.value == "stale"
        assert snap.age_days == 30

    def test_rebuild_projection_from_history(self, session, now):
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="A", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("10"), observed_at=now - timedelta(days=2)))
        book.record(PriceObservation(sku="A", vendor="V", unit=Unit.SQM,
                                     amount=Decimal("20"), observed_at=now))
        book.record(PriceObservation(sku="B", vendor="V", unit=Unit.KG,
                                     amount=Decimal("5"), observed_at=now))
        session.commit()
        assert PriceRepository(session).rebuild_projection() == 2
        session.commit()
        assert book.current("A").amount == Decimal("20.00")

    def test_unpriced_sku(self, session):
        assert SqlPriceBook(session).snapshot("NOPE").status.value == "unpriced"


class TestQuotePersistence:
    def test_prices_are_frozen_against_later_changes(self, session, windowed_room, now):
        """The reproducibility guarantee: a vendor raising rates after the
        quote must not change what the quote says."""
        scene = Scene(rooms=(windowed_room,))
        SceneRepository(session).save(scene)
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="TILE-STD", vendor="K", unit=Unit.SQM,
                                     amount=Decimal("900"), observed_at=now))
        session.commit()

        quote = PricingEngine(book).quote_scene(scene, include_furniture=False)
        record = QuoteRepository(session).save(quote)
        session.commit()

        book.record(PriceObservation(sku="TILE-STD", vendor="K", unit=Unit.SQM,
                                     amount=Decimal("1500"), observed_at=now + timedelta(days=1)))
        session.commit()

        stored = QuoteRepository(session).get(record.id)
        tile_line = next(l for l in stored.lines if l.sku == "TILE-STD")
        assert tile_line.unit_price == Decimal("900.00")
        assert book.current("TILE-STD").amount == Decimal("1500.00")

    def test_quote_pins_the_scene_version(self, session, windowed_room, now):
        scene = Scene(rooms=(windowed_room,))
        SceneRepository(session).save(scene)
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="TILE-STD", vendor="K", unit=Unit.SQM,
                                     amount=Decimal("900"), observed_at=now))
        session.commit()
        quote = PricingEngine(book).quote_scene(scene, include_furniture=False)
        record = QuoteRepository(session).save(quote)
        session.commit()
        assert record.scene_version_id == scene.version_id

    def test_unpriced_lines_persist_as_unpriced(self, session, windowed_room):
        scene = Scene(rooms=(windowed_room,))
        SceneRepository(session).save(scene)
        session.commit()
        quote = PricingEngine(SqlPriceBook(session)).quote_scene(scene, include_furniture=False)
        record = QuoteRepository(session).save(quote)
        session.commit()
        stored = QuoteRepository(session).get(record.id)
        assert all(l.price_status == "unpriced" for l in stored.lines)
        assert all(l.line_total is None for l in stored.lines)

    def test_lines_keep_their_basis(self, session, windowed_room, now):
        scene = Scene(rooms=(windowed_room,))
        SceneRepository(session).save(scene)
        book = SqlPriceBook(session)
        book.record(PriceObservation(sku="PAINT-STD", vendor="A", unit=Unit.LITRE,
                                     amount=Decimal("420"), observed_at=now))
        session.commit()
        quote = PricingEngine(book).quote_scene(scene, include_furniture=False)
        record = QuoteRepository(session).save(quote)
        session.commit()
        paint = next(l for l in QuoteRepository(session).get(record.id).lines if l.sku == "PAINT-STD")
        assert "openings" in paint.basis