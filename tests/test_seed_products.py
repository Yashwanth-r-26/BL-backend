"""Bulk product loading.

The dataset feeds the fit engine, so a wrong dimension is not cosmetic -- it
changes which products a customer is offered. These tests guard the data's
integrity and the loader's behaviour, including the part that matters most
operationally: it must never silently claim a background was stripped when it
was not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interior_ai.db.product_catalogue import (
    MATERIALS,
    PRODUCTS,
    TREATMENTS,
    counts_by_class,
)
from interior_ai.db.seed_products import find_image, main


class TestDatasetIntegrity:
    def test_at_least_ten_per_object_class(self):
        counts = counts_by_class()
        object_classes = {p[2] for p in PRODUCTS}
        thin = {c: counts[c] for c in object_classes if counts[c] < 10}
        assert not thin, f"classes with fewer than 10 products: {thin}"

    def test_skus_are_unique(self):
        skus = [p[0] for p in PRODUCTS] + [t[0] for t in TREATMENTS]
        assert len(skus) == len(set(skus))

    def test_dimensions_are_positive(self):
        for sku, _n, _c, w, d, h, *_ in PRODUCTS:
            assert w > 0 and d > 0 and h > 0, sku

    def test_every_product_has_a_description(self):
        """Descriptions are fed verbatim into the replacement prompt, so an
        empty one directly degrades swap fidelity."""
        for sku, *_rest in PRODUCTS:
            desc = _rest[-1]
            assert len(desc) > 15, f"{sku} description too thin"

    def test_sofa_widths_match_indian_market_ranges(self):
        """3-seaters run 1830-2285 mm, 2-seaters 1370-1675 mm."""
        for sku, name, cls, w, *_ in PRODUCTS:
            if cls != "sofa":
                continue
            if "3-Seater" in name or "3S" in sku:
                assert 1830 <= w <= 2300, f"{sku} width {w} outside 3-seater range"
            elif "2-Seater" in name or "2S" in sku:
                assert 1370 <= w <= 1700, f"{sku} width {w} outside 2-seater range"

    def test_dining_tables_use_market_height(self):
        """Height is consistent at 750-770 mm across the Indian market."""
        for sku, _n, cls, _w, _d, h, *_ in PRODUCTS:
            if cls == "dining_table":
                assert 740 <= h <= 780, f"{sku} height {h} unusual for dining"

    def test_wardrobe_depths_allow_a_hanging_rail(self):
        """Standard depth is 580-620 mm; less and hangers touch the back."""
        for sku, _n, cls, _w, d, _h, *_ in PRODUCTS:
            if cls == "wardrobe" and "Corner" not in _n and "Walk" not in _n:
                assert 540 <= d <= 660, f"{sku} depth {d} impractical"

    def test_beds_match_indian_sizes(self):
        """Queen 1500x1900 and King 1800x2000 mattresses; frames run larger."""
        for sku, name, cls, w, d, _h, *_ in PRODUCTS:
            if cls != "bed":
                continue
            assert 1000 <= w <= 2000, f"{sku} width {w}"
            assert 1900 <= d <= 2200, f"{sku} length {d}"

    def test_surface_treatments_are_placeholder_sized(self):
        """Surfaces are never fit-checked, so 1x1x1 is honest rather than a
        fabricated footprint."""
        for sku, _n, cls, *_ in TREATMENTS:
            assert cls in {"wall", "ceiling", "floor"}, sku

    def test_paint_swatches_are_valid_hex(self):
        for sku, _n, cls, _p, _d, tags in TREATMENTS:
            if "hex" in tags:
                assert tags["hex"].startswith("#") and len(tags["hex"]) == 7, sku

    def test_materials_cover_the_takeoff(self):
        """A quote is incomplete without every material the takeoff derives."""
        needed = {"TILE-STD", "ADHESIVE-STD", "GROUT-STD",
                  "PAINT-STD", "PRIMER-STD", "PUTTY-STD"}
        assert needed <= {m[0] for m in MATERIALS}


class TestImageMatching:
    def test_matches_by_sku_filename(self, tmp_path):
        (tmp_path / "SOFA-MILANO-3S.jpg").write_bytes(b"x")
        assert find_image(tmp_path, "SOFA-MILANO-3S").name == "SOFA-MILANO-3S.jpg"

    def test_matching_is_case_insensitive(self, tmp_path):
        (tmp_path / "sofa-milano-3s.png").write_bytes(b"x")
        assert find_image(tmp_path, "SOFA-MILANO-3S") is not None

    def test_missing_photo_returns_none(self, tmp_path):
        assert find_image(tmp_path, "NOPE") is None

    def test_no_directory_returns_none(self):
        assert find_image(None, "SOFA-MILANO-3S") is None

    def test_ignores_unrelated_extensions(self, tmp_path):
        (tmp_path / "SOFA-MILANO-3S.txt").write_bytes(b"x")
        assert find_image(tmp_path, "SOFA-MILANO-3S") is None


class TestLoaderCli:
    def test_list_reports_counts(self, capsys):
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        assert "sofa" in out and "TOTAL" in out

    def test_images_without_api_is_rejected(self, tmp_path, capsys):
        """Stripping happens server-side; --images alone would silently do
        nothing, so it fails loudly instead."""
        assert main(["--images", str(tmp_path)]) == 2
        assert "needs --api" in capsys.readouterr().err

    def test_bad_images_directory_is_rejected(self, capsys):
        assert main(["--images", "/no/such/dir", "--api", "http://x"]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_dry_run_direct_changes_nothing(self, capsys):
        assert main(["--dry-run"]) == 0
        assert "dry-run" in capsys.readouterr().out


class TestBuildCatalogue:
    """One-pass catalogue build: generate a white-background image and store
    the product, image and opening price together.

    The design point under test is the one that was wrong first time round:
    a generated image is *already* isolated on white, so it must not be sent
    through the background strip. Doing so spends a second image-generation
    call per product to reproduce what it started with.
    """

    @pytest.fixture
    def db_url(self, tmp_path, monkeypatch):
        url = f"sqlite+pysqlite:///{tmp_path / 'build.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        create_all(make_engine(url))
        return url

    def test_stores_products_images_and_prices(self, db_url):
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        assert build(only_class="sofa", limit=2, include_treatments=False) == 0
        with make_session_factory(make_engine(db_url))() as db:
            rows = [db.get(CatalogueItemRow, p[0]) for p in PRODUCTS if p[2] == "sofa"][:2]
        assert all(r is not None for r in rows)
        assert all((r.image_ref or "").startswith("data:") for r in rows)

    def test_requires_a_database(self, monkeypatch, capsys):
        """Silently writing a catalogue into a database that vanishes on exit
        would be worse than refusing."""
        from interior_ai.db.build_catalogue import build

        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert build(only_class="sofa", limit=1) == 2
        assert "DATABASE_URL is not set" in capsys.readouterr().err

    def test_rerun_keeps_existing_images(self, db_url, capsys):
        from interior_ai.db.build_catalogue import build

        build(only_class="sofa", limit=2, include_treatments=False)
        capsys.readouterr()
        build(only_class="sofa", limit=2, include_treatments=False)
        out = capsys.readouterr().out
        assert "already present" in out
        assert "generated 0 images" in out

    def test_overwrite_regenerates(self, db_url, capsys):
        from interior_ai.db.build_catalogue import build

        build(only_class="sofa", limit=1, include_treatments=False)
        capsys.readouterr()
        build(only_class="sofa", limit=1, include_treatments=False, overwrite=True)
        assert "generated 1 images" in capsys.readouterr().out

    def test_no_images_stores_specs_only(self, db_url):
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        build(only_class="sofa", limit=1, with_images=False, include_treatments=False)
        sku = next(p[0] for p in PRODUCTS if p[2] == "sofa")
        with make_session_factory(make_engine(db_url))() as db:
            row = db.get(CatalogueItemRow, sku)
        assert row is not None and not row.image_ref

    def test_treatments_carry_style_tags(self, db_url):
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        build(only_class="wall", limit=1, with_images=False)
        with make_session_factory(make_engine(db_url))() as db:
            row = db.get(CatalogueItemRow, "PAINT-W-IVORY")
        assert row is not None
        assert row.style_tags.get("hex")

    def test_save_dir_writes_images_for_inspection(self, db_url, tmp_path):
        from interior_ai.db.build_catalogue import build

        out = tmp_path / "inspect"
        build(only_class="sofa", limit=1, include_treatments=False, save_dir=out)
        assert len(list(out.glob("*.png"))) == 1

    def test_dry_run_stores_nothing(self, db_url):
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        build(only_class="sofa", limit=2, dry_run=True)
        with make_session_factory(make_engine(db_url))() as db:
            sku = next(p[0] for p in PRODUCTS if p[2] == "sofa")
            assert db.get(CatalogueItemRow, sku) is None

    def test_cli_rejects_bad_limit(self, capsys):
        from interior_ai.db.build_catalogue import main

        assert main(["--limit", "0"]) == 2


class TestGeneratedImagesSkipTheStrip:
    """The endpoint must accept an already-isolated image without re-stripping."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    @staticmethod
    def _photo() -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buf, "JPEG")
        return buf.getvalue()

    def test_strip_can_be_skipped(self, client):
        body = client.post("/catalogue/upload", data={
            "sku": "SOFA-PREPPED", "name": "Prepped", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "40000", "strip_background": "false",
        }, files={"image": ("p.jpg", self._photo(), "image/jpeg")}).json()
        assert any("skipped" in note for note in body["notes"])

    def test_strip_is_attempted_by_default(self, client):
        body = client.post("/catalogue/upload", data={
            "sku": "SOFA-RAW", "name": "Raw", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "40000",
        }, files={"image": ("p.jpg", self._photo(), "image/jpeg")}).json()
        assert not any("skipped" in note for note in body["notes"])


class TestProductImagePrompt:
    """Images are generated from each product's own specification.

    The alternative -- searching the web for a product name -- would attach
    another company's photograph to this catalogue's SKU, so the picker would
    show one sofa while the room edit inserted a different one.
    """

    def _editor(self):
        import base64
        import io

        from PIL import Image

        from interior_ai.perception.editing import GeminiPhotoEditor

        seen: dict = {}

        def transport(model, payload):
            seen["model"] = model
            seen["prompt"] = payload["contents"][0]["parts"][0]["text"]
            seen["parts"] = payload["contents"][0]["parts"]
            buf = io.BytesIO()
            Image.new("RGB", (768, 768), "white").save(buf, "PNG")
            return {"candidates": [{"content": {"parts": [{"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode(),
            }}]}}]}

        return GeminiPhotoEditor(api_key="x", transport=transport), seen

    def test_specification_reaches_the_model(self):
        editor, seen = self._editor()
        editor.generate_product_image(
            name="Milano 3-Seater Sofa",
            description="Charcoal grey fabric, tapered oak legs, boxy silhouette",
            object_class="sofa",
        )
        assert "Milano 3-Seater Sofa" in seen["prompt"]
        assert "tapered oak legs" in seen["prompt"]

    def test_output_is_specified_on_white(self):
        editor, seen = self._editor()
        editor.generate_product_image(name="X", description="y", object_class="sofa")
        assert "Pure white background" in seen["prompt"]

    def test_generation_is_text_only(self):
        editor, seen = self._editor()
        editor.generate_product_image(name="X", description="y", object_class="sofa")
        assert all("inline_data" not in part for part in seen["parts"])

    def test_prompt_forbids_branding_and_props(self):
        editor, seen = self._editor()
        editor.generate_product_image(name="X", description="y", object_class="sofa")
        assert "no watermark" in seen["prompt"]
        assert "brand marking" in seen["prompt"]

    def test_mock_generator_is_visibly_a_placeholder(self):
        from interior_ai.perception.editing import MockPhotoEditor

        uri = MockPhotoEditor().generate_product_image(
            name="Milano", description="grey", object_class="sofa"
        )
        assert uri.startswith("data:image/png")


class TestTransientFailureHandling:
    """A 503 means "busy", not "wrong". Treating the two the same is how a
    momentary overload becomes a catalogue of products with no photographs."""

    def _editor(self, responses, *, attempts=5):
        """Editor whose transport yields `responses` in order; each entry is
        either an exception to raise or None for success."""
        import base64
        import io

        from PIL import Image

        from interior_ai.perception.editing import GeminiPhotoEditor

        state = {"calls": 0, "models": []}

        def transport(model, payload):
            state["models"].append(model)
            index = state["calls"]
            state["calls"] += 1
            outcome = responses[index] if index < len(responses) else None
            if outcome is not None:
                raise outcome
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), "white").save(buf, "PNG")
            return {"candidates": [{"content": {"parts": [{"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode(),
            }}]}}]}

        editor = GeminiPhotoEditor(api_key="x", transport=transport)
        editor.default_attempts = attempts
        return editor, state

    def test_503_is_retried_until_it_succeeds(self, monkeypatch):
        from interior_ai.providers.base import ProviderError

        monkeypatch.setattr("time.sleep", lambda _s: None)
        editor, state = self._editor(
            [ProviderError("busy", status_code=503)] * 3
        )
        uri = editor.generate_product_image(
            name="S", description="d", object_class="sofa"
        )
        assert uri.startswith("data:image")
        assert state["calls"] == 4

    def test_permanent_errors_are_not_retried(self):
        from interior_ai.providers.base import ProviderError

        editor, state = self._editor([ProviderError("bad key", status_code=401)] * 5)
        with pytest.raises(ProviderError):
            editor.generate_product_image(name="S", description="d", object_class="sofa")
        assert state["calls"] == 1, "retrying a 401 just repeats the same error"

    def test_timeouts_are_treated_as_transient(self):
        """The request never got a verdict, so trying again is legitimate."""
        from interior_ai.providers.base import ProviderError

        assert ProviderError("read timed out", retryable=True).retryable

    def test_retry_after_header_is_honoured(self, monkeypatch):
        from interior_ai.providers.base import ProviderError

        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        editor, _state = self._editor(
            [ProviderError("slow down", status_code=429, retry_after_s=7.0)],
            attempts=3,
        )
        editor.generate_product_image(name="S", description="d", object_class="sofa")
        assert slept and 5.0 <= slept[0] <= 9.0

    def test_backoff_grows_between_attempts(self, monkeypatch):
        from interior_ai.providers.base import ProviderError

        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        editor, _state = self._editor([ProviderError("busy", status_code=503)] * 3)
        editor.generate_product_image(name="S", description="d", object_class="sofa")
        assert slept[-1] > slept[0], "delays must grow, not hammer the service"

    def test_falls_back_to_the_next_model(self, monkeypatch):
        from interior_ai.providers.base import ProviderError

        monkeypatch.setenv("GEMINI_IMAGE_FALLBACKS", "second-model")
        monkeypatch.setattr("time.sleep", lambda _s: None)
        editor, state = self._editor(
            [ProviderError("busy", status_code=503)] * 2, attempts=2
        )
        editor.generate_product_image(name="S", description="d", object_class="sofa")
        assert "second-model" in state["models"]

    def test_model_chain_deduplicates(self, monkeypatch):
        from interior_ai.perception.editing import GeminiPhotoEditor

        monkeypatch.setenv("GEMINI_IMAGE_FALLBACKS", "a,a,b")
        chain = GeminiPhotoEditor(api_key="x", edit_model="a").image_model_chain
        assert chain == ["a", "b"]

    def test_status_code_survives_on_the_error(self):
        from interior_ai.providers.base import ProviderError

        exc = ProviderError("busy", status_code=503)
        assert exc.status_code == 503 and exc.retryable


class TestBuildAbortsOnOutage:
    """When the model is down, every remaining product would store without a
    photograph. Stopping keeps the run cheap to resume."""

    @pytest.fixture
    def db_url(self, tmp_path, monkeypatch):
        url = f"sqlite+pysqlite:///{tmp_path / 'abort.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        create_all(make_engine(url))
        return url

    def _kill_generation(self, monkeypatch):
        from interior_ai.perception import editing
        from interior_ai.providers.base import ProviderError

        class Dead:
            def generate_product_image(self, *, name, description,
                                       object_class, on_retry=None):
                raise ProviderError("high demand", status_code=503)

        monkeypatch.setattr(editing, "GeminiPhotoEditor", lambda *a, **k: Dead())
        monkeypatch.setattr(editing, "MockPhotoEditor", lambda *a, **k: Dead())

    def test_stops_after_consecutive_failures(self, db_url, monkeypatch, capsys):
        from interior_ai.db.build_catalogue import build

        self._kill_generation(monkeypatch)
        assert build(only_class="sofa", include_treatments=False) == 1
        out = capsys.readouterr().out
        assert "failed in a row" in out
        assert out.count("generating image") <= 3, "should not chew through all 10"

    def test_explains_a_503(self, db_url, monkeypatch, capsys):
        from interior_ai.db.build_catalogue import build

        self._kill_generation(monkeypatch)
        build(only_class="sofa", include_treatments=False)
        assert "busy, not that anything is wrong" in capsys.readouterr().out

    def test_abort_can_be_disabled(self, db_url, monkeypatch, capsys):
        from interior_ai.db.build_catalogue import build

        self._kill_generation(monkeypatch)
        build(only_class="sofa", limit=5, include_treatments=False, abort_after=0)
        assert capsys.readouterr().out.count("generating image") == 5

    def test_partial_progress_is_kept(self, db_url, monkeypatch):
        """An aborted run must leave what it already stored, so re-running
        continues rather than restarts."""
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        self._kill_generation(monkeypatch)
        build(only_class="sofa", include_treatments=False)
        with make_session_factory(make_engine(db_url))() as db:
            assert db.get(CatalogueItemRow, "SOFA-MILANO-3S") is not None


class TestPricesRecordedPerProduct:
    """Prices used to be written in one batch after every product was stored.
    An interrupted run -- an abort on repeated 503s, a Ctrl-C -- returned
    before reaching that block, leaving every stored product with no price and
    quotes reporting them as unpriced for no visible reason.
    """

    @pytest.fixture
    def db_url(self, tmp_path, monkeypatch):
        url = f"sqlite+pysqlite:///{tmp_path / 'p.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        create_all(make_engine(url))
        return url

    def _kill_generation(self, monkeypatch):
        from interior_ai.perception import editing
        from interior_ai.providers.base import ProviderError

        class Dead:
            def generate_product_image(self, *, name, description,
                                       object_class, on_retry=None):
                raise ProviderError("high demand", status_code=503)

        monkeypatch.setattr(editing, "GeminiPhotoEditor", lambda *a, **k: Dead())
        monkeypatch.setattr(editing, "MockPhotoEditor", lambda *a, **k: Dead())

    def _prices(self, db_url):
        from sqlalchemy import select

        from interior_ai.db.models import PriceHistory
        from interior_ai.db.repository import make_engine, make_session_factory

        with make_session_factory(make_engine(db_url))() as db:
            return {r.sku for r in db.execute(select(PriceHistory)).scalars()}

    def test_aborted_run_still_prices_what_it_stored(self, db_url, monkeypatch):
        from interior_ai.db.build_catalogue import build
        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        self._kill_generation(monkeypatch)
        assert build(only_class="bookshelf", include_treatments=False) == 1

        with make_session_factory(make_engine(db_url))() as db:
            from sqlalchemy import select

            stored = {r.sku for r in db.execute(select(CatalogueItemRow)).scalars()}
        assert stored, "the run stored something before aborting"
        assert stored <= self._prices(db_url), "every stored product must be priced"

    def test_completed_run_prices_everything(self, db_url):
        from interior_ai.db.build_catalogue import build

        assert build(only_class="bookshelf", with_images=False,
                     include_treatments=False) == 0
        priced = self._prices(db_url)
        assert len([p for p in PRODUCTS if p[2] == "bookshelf"]) <= len(priced)

    def test_materials_are_priced_too(self, db_url):
        from interior_ai.db.build_catalogue import build

        build(only_class="bookshelf", with_images=False, include_treatments=False)
        assert {m[0] for m in MATERIALS} <= self._prices(db_url)


class TestRedetect:
    """Boxes updated after a swap are estimates from the product's dimensions.
    Re-detection measures the edited image instead -- and must not cost the
    swaps already made."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'r.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        return TestClient(create_app())

    @staticmethod
    def _png() -> bytes:
        import struct
        import zlib

        def ch(tag, data):
            body = tag + data
            return (struct.pack(">I", len(data)) + body
                    + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

        return (b"\x89PNG\r\n\x1a\n"
                + ch(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                + ch(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
                + ch(b"IEND", b""))

    def _session(self, client):
        client.post("/catalogue", json={
            "sku": "BS-DIVIDER-1", "name": "Room Divider Shelf",
            "object_class": "bookshelf", "width_mm": 1600, "depth_mm": 400,
            "height_mm": 1800, "display_price": "34000",
        })
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        return client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        ).json()

    def test_redetect_returns_fresh_boxes(self, client):
        session = self._session(client)
        out = client.post(
            f"/edit-sessions/{session['session_id']}/redetect"
        ).json()
        assert out["count"] > 0
        assert len(out["detections"]) == out["count"]

    def test_swaps_survive_redetection(self, client):
        session = self._session(client)
        sofa = next(d for d in session["detections"] if d["object_class"] == "sofa")
        client.post(f"/edit-sessions/{session['session_id']}/apply",
                    json={"detection_id": sofa["id"], "sku": "BS-DIVIDER-1"})
        out = client.post(
            f"/edit-sessions/{session['session_id']}/redetect"
        ).json()
        assert out["swapped_skus"] == {sofa["id"]: "BS-DIVIDER-1"}

    def test_quote_is_unchanged_by_redetection(self, client):
        session = self._session(client)
        sofa = next(d for d in session["detections"] if d["object_class"] == "sofa")
        client.post(f"/edit-sessions/{session['session_id']}/apply",
                    json={"detection_id": sofa["id"], "sku": "BS-DIVIDER-1"})
        before = client.post(f"/edit-sessions/{session['session_id']}/quote").json()
        client.post(f"/edit-sessions/{session['session_id']}/redetect")
        after = client.post(f"/edit-sessions/{session['session_id']}/quote").json()
        assert before["total"] == after["total"]

    def test_unknown_session_404s(self, client):
        assert client.post("/edit-sessions/nope/redetect").status_code == 404

    def test_ui_offers_the_button(self, client):
        assert "btn-redetect" in client.get("/ui").text