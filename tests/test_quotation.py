"""Location, brief, and the full quotation.

The invariant these guard: a price we already know must never be re-estimated.
A swapped-in product has a SKU, a catalogue price and a vendor -- those are
facts. Labour, materials and regional rates are not, and nothing in this system
knows what a painter charges in Mysuru. Letting a model re-price the sofa whose
price the customer was shown a minute ago produces a quote that contradicts the
picker, and there is no good way to explain that to them.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import zlib
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from interior_ai.api.app import create_app
from interior_ai.db.regions import city_tier, describe, prior_region


def _png() -> bytes:
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
            + chunk(b"IEND", b""))


class TestRegions:
    def test_only_india_is_supported(self):
        """Extending the picker to markets we have no rate data for would mean
        asking a model to price somewhere we cannot check its answer."""
        assert describe("IN", "Bengaluru")["supported"]
        assert not describe("US", "Austin")["supported"]

    def test_unsupported_markets_explain_themselves(self):
        assert describe("AE", "Dubai")["note"]

    def test_metro_cities_are_recognised(self):
        for city in ("Bengaluru", "Mumbai", "Gurugram", "  CHENNAI "):
            assert city_tier(city) == "metro", city

    def test_tier_two_cities_are_recognised(self):
        for city in ("Jaipur", "Mysuru", "Kochi"):
            assert city_tier(city) == "tier2", city

    def test_unknown_cities_default_to_the_cheaper_tier(self):
        """Overstating a small-town budget makes the whole quote read as wrong
        to the person who lives there."""
        assert city_tier("Some Small Town") == "tier3"

    def test_city_chooses_the_dimension_prior(self):
        assert prior_region("Bengaluru") == "IN_METRO"
        assert prior_region("Mysuru") == "IN_NONMETRO"


class TestJsonRepair:
    def test_clean_json(self):
        from interior_ai.perception.quotation import repair_json

        assert repair_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        from interior_ai.perception.quotation import repair_json

        assert repair_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_truncated_response_is_salvaged(self):
        """A long quotation can hit the output limit mid-string. Losing an
        otherwise complete estimate to one missing brace is a poor trade."""
        from interior_ai.perception.quotation import repair_json

        parsed = repair_json(
            '{"currency":"INR","contractor":{"total":120000,'
            '"line_items":[{"name":"Paint'
        )
        assert parsed is not None
        assert parsed["currency"] == "INR"

    def test_unusable_text_returns_none(self):
        """A half-invented quote would be worse than none."""
        from interior_ai.perception.quotation import repair_json

        assert repair_json("sorry, I cannot help with that") is None


class TestQuotationPrompt:
    def _quoter(self):
        from interior_ai.perception.quotation import GeminiQuoter

        seen: dict = {}

        def transport(model, payload):
            parts = payload["contents"][0]["parts"]
            seen["model"] = model
            seen["text"] = "\n".join(p["text"] for p in parts if "text" in p)
            seen["images"] = sum(1 for p in parts if "inline_data" in p)
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(
                {"currency": "INR", "contractor": {"total": 1}})}]}}]}

        return GeminiQuoter(api_key="x", transport=transport), seen

    def _image(self) -> str:
        buf = io.BytesIO()
        Image.new("RGB", (1800, 1200), "tan").save(buf, "JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    def _manifest(self):
        return {
            "known_products": [{
                "sku": "SOFA-B", "name": "Milano 3-Seater", "object_class": "sofa",
                "description": "charcoal fabric", "replaced": "three-seat sofa",
                "width_mm": 2100, "depth_mm": 880, "height_mm": 820,
                "price": "52000", "currency": "INR", "vendor": "NovaHome",
            }],
            "instructions": [{"instruction": "paint the walls sage green",
                              "applied_to": "whole scene"}],
        }

    def _quote(self, quoter, **over):
        kwargs = {
            "location": {"city": "Bengaluru", "country_name": "India",
                         "city_tier": "metro", "currency": "INR"},
            "questionnaire": {"quality_tier": "mid-range"},
            "manifest": self._manifest(),
            "date_str": "August 21, 2026",
        }
        kwargs.update(over)
        return quoter.quote(self._image(), self._image(), **kwargs)

    def test_both_images_are_sent(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert seen["images"] == 2

    def test_known_prices_are_marked_as_not_negotiable(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert "KNOWN ITEMS ARE NOT YOURS TO ESTIMATE" in seen["text"]
        assert "Use that exact figure" in seen["text"]

    def test_known_items_carry_price_and_vendor(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert "INR 52000 from NovaHome" in seen["text"]

    def test_instructions_are_flagged_for_estimation(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert "paint the walls sage green" in seen["text"]
        assert "estimate materials and labour" in seen["text"]

    def test_city_and_market_tier_reach_the_prompt(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert "Bengaluru" in seen["text"]
        assert "a major metro" in seen["text"]

    def test_tier_changes_with_the_city(self):
        quoter, seen = self._quoter()
        self._quote(quoter, location={"city": "Hosur", "country_name": "India",
                                      "city_tier": "tier3", "currency": "INR"})
        assert "smaller city or town" in seen["text"]

    def test_each_line_must_declare_its_pricing_kind(self):
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert '"known|estimated"' in seen["text"]

    def test_diy_guidance_is_safety_aware(self):
        """Encouraging someone to do their own electrical work cheaply is the
        one way this feature could actually hurt somebody."""
        quoter, seen = self._quoter()
        self._quote(quoter)
        assert "unsafe work cheaply" in seen["text"]
        assert "needs a professional" in seen["text"].lower()

    def test_model_failure_is_reported_not_raised(self):
        from interior_ai.perception.quotation import GeminiQuoter
        from interior_ai.providers.base import ProviderError

        def dead(model, payload):
            raise ProviderError("busy", status_code=503)

        result = self._quote(GeminiQuoter(api_key="x", transport=dead))
        assert result["status"] == "error"
        assert result["data"] is None
        assert result["notes"]


class TestQuotationEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'q.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        client = TestClient(create_app())
        client.post("/catalogue", json={
            "sku": "SOFA-B", "name": "Milano 3-Seater", "object_class": "sofa",
            "width_mm": 2100, "depth_mm": 880, "height_mm": 820,
            "display_price": "52000", "vendor": "NovaHome",
        })
        return client

    def _session(self, client):
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        session = client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", _png(), "image/png")},
        ).json()
        target = next(d for d in session["detections"] if d["object_class"] == "sofa")
        return session["session_id"], target["id"]

    def test_regions_endpoint_lists_support(self, client):
        countries = client.get("/regions").json()["countries"]
        assert any(c["code"] == "IN" and c["supported"] for c in countries)
        assert any(not c["supported"] for c in countries)

    def test_location_is_stored_and_described(self, client):
        session, _ = self._session(client)
        out = client.post(f"/edit-sessions/{session}/location",
                          json={"country": "IN", "city": "Bengaluru"}).json()
        assert out["city_tier"] == "metro"
        assert out["currency"] == "INR"
        assert out["prior_region"] == "IN_METRO"

    def test_unsupported_country_is_refused(self, client):
        session, _ = self._session(client)
        resp = client.post(f"/edit-sessions/{session}/location",
                           json={"country": "US", "city": "Austin"})
        assert resp.status_code == 422

    def test_quotation_requires_a_location(self, client):
        """A price without a city is a guess dressed as a number."""
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": target, "sku": "SOFA-B"})
        resp = client.post(f"/edit-sessions/{session}/quotation")
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "location_required"

    def test_quotation_requires_a_change(self, client):
        session, _ = self._session(client)
        client.post(f"/edit-sessions/{session}/location",
                    json={"country": "IN", "city": "Bengaluru"})
        resp = client.post(f"/edit-sessions/{session}/quotation")
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "nothing_to_quote"

    def test_questionnaire_is_stored(self, client):
        session, _ = self._session(client)
        out = client.post(f"/edit-sessions/{session}/questionnaire", json={
            "scope": ["walls", "furniture"], "quality_tier": "mid-range",
        }).json()
        assert out["questionnaire"]["scope"] == ["walls", "furniture"]

    def test_quotation_returns_its_inputs(self, client):
        """Every number should be traceable to what produced it."""
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/location",
                    json={"country": "IN", "city": "Bengaluru"})
        client.post(f"/edit-sessions/{session}/questionnaire",
                    json={"quality_tier": "premium"})
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": target, "sku": "SOFA-B"})
        out = client.post(f"/edit-sessions/{session}/quotation").json()
        assert out["location"]["city"] == "Bengaluru"
        assert out["questionnaire"]["quality_tier"] == "premium"
        assert out["known_products"][0]["sku"] == "SOFA-B"

    def test_offline_quotation_says_it_estimated_nothing(self, client):
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/location",
                    json={"country": "IN", "city": "Bengaluru"})
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": target, "sku": "SOFA-B"})
        out = client.post(f"/edit-sessions/{session}/quotation").json()
        assert out["status"] == "mock"
        assert any("nothing has been estimated" in n.lower() for n in out["notes"])


class TestChangeManifest:
    """What the quote is built from: real prices kept apart from guesses."""

    @pytest.fixture
    def service(self, tmp_path):
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import create_all, make_engine, make_session_factory
        from interior_ai.perception.edit_session import EditSessionService
        from interior_ai.perception.editing import MockPhotoEditor

        engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'm.db'}")
        create_all(engine)
        db = make_session_factory(engine)()
        for sku, name, cls, w, d, h, price in [
            ("SOFA-A", "Oslo 2-Seater", "sofa", 1650, 850, 800, "38000"),
            ("SOFA-B", "Milano 3-Seater", "sofa", 2100, 880, 820, "52000"),
        ]:
            db.add(CatalogueItemRow(sku=sku, name=name, object_class=cls,
                                    width_mm=w, depth_mm=d, height_mm=h,
                                    display_price=Decimal(price),
                                    vendor="NovaHome"))
        db.commit()
        svc = EditSessionService(db, editor=MockPhotoEditor())
        session = svc.start(scene_id="s", room_id="r", image_ref="mock://o")
        return svc, session, db

    def test_known_products_carry_real_prices(self, service):
        svc, session, db = service
        svc.apply(session, "det-sofa", "SOFA-B")
        manifest = svc.change_manifest(session)
        assert manifest["known_products"][0]["price"] == "52000.00"
        assert manifest["known_products"][0]["vendor"] == "NovaHome"
        db.close()

    def test_superseded_products_are_dropped(self, service):
        """Only what is in the final image should be priced."""
        svc, session, db = service
        svc.apply(session, "det-sofa", "SOFA-A")
        svc.apply(session, "det-sofa", "SOFA-B")
        manifest = svc.change_manifest(session)
        assert [k["sku"] for k in manifest["known_products"]] == ["SOFA-B"]
        db.close()

    def test_instructions_are_listed_separately(self, service):
        svc, session, db = service
        svc.instruct(session, "paint the walls sage green")
        manifest = svc.change_manifest(session)
        assert manifest["known_products"] == []
        assert manifest["instructions"][0]["instruction"] == "paint the walls sage green"
        db.close()


class TestDeviceLocation:
    """A phone already knows where it is; asking someone to type it is asking
    them to repeat what their device holds.

    Coordinates are resolved to a city on our own server against a local
    table. A reverse-geocoding API would be a key to manage, a rate limit to
    hit, a privacy question to answer, and a dependency that fails offline --
    all to choose a pricing tier that does not change between one suburb and
    the next.
    """

    def test_a_fix_resolves_to_its_city(self):
        from interior_ai.db.regions import nearest_city

        assert nearest_city(12.9716, 77.5946)["city"] == "Bengaluru"
        assert nearest_city(19.076, 72.877)["city"] == "Mumbai"

    def test_a_suburb_resolves_to_its_city(self):
        """Whitefield is Bengaluru for pricing purposes."""
        from interior_ai.db.regions import nearest_city

        fix = nearest_city(12.9698, 77.7500)
        assert fix["city"] == "Bengaluru"
        assert fix["confident"]

    def test_a_remote_fix_is_flagged(self):
        """Pricing a village at a metro's rates would be wrong in a way nobody
        could see, so distance is reported rather than hidden."""
        from interior_ai.db.regions import NEAREST_CITY_WARN_KM, nearest_city

        fix = nearest_city(21.0, 92.5)
        assert fix["distance_km"] > NEAREST_CITY_WARN_KM
        assert not fix["confident"]

    def test_every_named_city_has_coordinates(self):
        """A city we price but cannot locate is unreachable from a device."""
        from interior_ai.db.regions import (
            CITY_COORDS,
            METRO_CITIES,
            TIER_TWO_CITIES,
        )

        missing = (METRO_CITIES | TIER_TWO_CITIES) - set(CITY_COORDS)
        # Aliases (bangalore/bengaluru) need not be duplicated in the table.
        aliases = {"bangalore", "secunderabad", "new delhi", "gurgaon",
                   "mysore", "cochin", "ernakulam", "trivandrum", "calicut",
                   "mangalore", "goa", "mohali", "panchkula", "chandigarh"}
        assert not (missing - aliases), missing - aliases


class TestLocationEndpointFromDevice:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'loc.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        return TestClient(create_app())

    def _session(self, client):
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        return client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", _png(), "image/png")},
        ).json()["session_id"]

    def test_coordinates_are_accepted(self, client):
        session = self._session(client)
        out = client.post(f"/edit-sessions/{session}/location", json={
            "country": "IN", "latitude": 12.9716, "longitude": 77.5946,
        }).json()
        assert out["city"] == "Bengaluru"
        assert out["source"] == "device"
        assert out["city_tier"] == "metro"

    def test_a_typed_city_still_works(self, client):
        session = self._session(client)
        out = client.post(f"/edit-sessions/{session}/location",
                          json={"country": "IN", "city": "Mysuru"}).json()
        assert out["source"] == "manual"
        assert out["city_tier"] == "tier2"

    def test_neither_is_refused(self, client):
        session = self._session(client)
        assert client.post(f"/edit-sessions/{session}/location",
                           json={"country": "IN"}).status_code == 422

    def test_a_far_fix_reports_its_distance(self, client):
        session = self._session(client)
        out = client.post(f"/edit-sessions/{session}/location", json={
            "country": "IN", "latitude": 21.0, "longitude": 92.5,
        }).json()
        assert out["confident"] is False
        assert out["distance_km"] > 100

    def test_impossible_coordinates_are_rejected(self, client):
        session = self._session(client)
        assert client.post(f"/edit-sessions/{session}/location", json={
            "country": "IN", "latitude": 991.0, "longitude": 77.0,
        }).status_code == 422


class TestFlowOrder:
    """The brief is asked while detection runs, not after it.

    Detection needs no input from the person, so making them watch it finish
    before answering questions wastes the one stretch of time they were going
    to spend typing anyway.
    """

    @pytest.fixture
    def html(self):
        from interior_ai.api.app import SceneStore

        return TestClient(create_app(store=SceneStore())).get("/ui").text

    def test_detection_starts_after_perception(self, html):
        assert "startDetectionInBackground" in html

    def test_the_brief_comes_before_the_swap_stage(self, html):
        assert html.index('id="s-questions"') < html.index('id="s-edit"')

    def test_device_location_is_offered(self, html):
        assert "navigator.geolocation" in html
        assert "btn-geo" in html

    def test_denied_permission_is_handled(self, html):
        """A refusal must leave a usable path, not a dead end."""
        assert "permission denied" in html

    def test_a_second_detection_is_not_started(self, html):
        """The background run and the button must not race each other."""
        assert "S.detecting" in html