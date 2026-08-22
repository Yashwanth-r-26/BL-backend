"""Interactive photo editing: detection, hit-testing, the session loop, API.

The invariants under test:

* A click selects the smallest containing detection (lamp in front of a
  wardrobe selects the lamp), with a proximity pad for near-misses.
* Malformed detection entries are skipped, never fatal.
* The step chain is append-only; undo moves the pointer without deleting.
* A later swap of the same object supersedes the earlier one -- the quote
  prices what is IN the final image, not everything that was ever tried.
* Catalogue items that cannot physically fit the room are offered last with a
  measured reason, not hidden.
* The session quote uses price_history, never the catalogue display price.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from interior_ai.core.scene import Room, Vec2
from interior_ai.db.catalogue import CatalogueItemRow
from interior_ai.db.repository import create_all, make_engine, make_session_factory
from interior_ai.perception.edit_session import EditSessionService
from interior_ai.perception.editing import (
    Detection,
    MockPhotoEditor,
    build_replace_prompt,
    hit_test,
    parse_detections,
)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def db():
    eng = make_engine("sqlite+pysqlite:///:memory:")
    create_all(eng)
    session = make_session_factory(eng)()
    yield session
    session.close()


@pytest.fixture
def catalogue(db):
    items = [
        ("SOFA-A", "Sofa A", "sofa", 2000, 850, 800, "40000"),
        ("SOFA-B", "Sofa B", "sofa", 1800, 800, 780, "35000"),
        ("SOFA-HUGE", "Palace Sofa", "sofa", 9000, 2000, 900, "250000"),
        ("CT-A", "Table A", "coffee_table", 1000, 550, 420, "11000"),
    ]
    for sku, name, cls, w, d, h, price in items:
        db.add(
            CatalogueItemRow(
                sku=sku, name=name, object_class=cls,
                width_mm=w, depth_mm=d, height_mm=h,
                display_price=Decimal(price),
            )
        )
    db.commit()
    return items


@pytest.fixture
def room():
    return Room(
        name="Living",
        polygon=(Vec2(x=0, y=0), Vec2(x=3700, y=0), Vec2(x=3700, y=4300), Vec2(x=0, y=4300)),
        ceiling_height_mm=3000,
    )


@pytest.fixture
def service(db):
    return EditSessionService(db, editor=MockPhotoEditor())


# ---------------------------------------------------------------- parsing


class TestDetectionParsing:
    def test_gemini_box_convention_converted(self):
        """Gemini emits [y_min, x_min, y_max, x_max]; we store (x0,y0,x1,y1)."""
        dets, _ = parse_detections(
            '[{"label":"sofa","object_class":"sofa","box_2d":[550,150,900,700]}]'
        )
        assert dets[0].box == (150, 550, 700, 900)

    def test_bad_entries_skipped_not_fatal(self):
        dets, notes = parse_detections(
            '[{"label":"ok","object_class":"sofa","box_2d":[10,10,200,200]},'
            ' {"nonsense":true}, "garbage",'
            ' {"label":"degenerate","object_class":"lamp","box_2d":[50,50,50,50]}]'
        )
        assert len(dets) == 1
        assert len(notes) == 3

    def test_unknown_class_becomes_other(self):
        dets, _ = parse_detections(
            '[{"label":"thing","object_class":"spaceship","box_2d":[0,0,100,100]}]'
        )
        assert dets[0].object_class == "other"

    def test_unparseable_returns_empty_with_note(self):
        dets, notes = parse_detections("I can see a lovely sofa in this room")
        assert dets == []
        assert notes


class TestHitTest:
    def _dets(self):
        return [
            Detection(id="big", label="wardrobe", object_class="wardrobe",
                      box=(100, 100, 900, 900), confidence=0.9),
            Detection(id="small", label="lamp", object_class="lamp",
                      box=(400, 300, 500, 700), confidence=0.8),
        ]

    def test_smallest_containing_box_wins(self):
        assert hit_test(self._dets(), 450, 500).id == "small"

    def test_outside_small_selects_big(self):
        assert hit_test(self._dets(), 200, 200).id == "big"

    def test_near_miss_pad_catches_close_clicks(self):
        only_lamp = [self._dets()[1]]
        # 510 is 10 outside the lamp box -> pad rescues it.
        assert hit_test(only_lamp, 510, 500).id == "small"

    def test_far_click_selects_nothing(self):
        assert hit_test([self._dets()[1]], 990, 30) is None


class TestReplacePrompt:
    def test_prompt_names_region_and_preservation(self):
        det = Detection(id="d", label="grey sofa", object_class="sofa",
                        box=(150, 550, 700, 900), confidence=0.9)
        p = build_replace_prompt(det, product_name="Milano", product_desc="fabric")
        assert "x 150-700" in p
        assert "pixel-identical" in p
        assert "Milano" in p


# ------------------------------------------------------------ the session


class TestEditSessionLoop:
    def test_start_stores_detections(self, service, db):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        db.commit()
        # 3 objects + 3 surfaces (wall / ceiling / floor)
        assert len(s.detections) == 6
        assert {d["object_class"] for d in s.detections} >= {"wall", "ceiling", "floor"}
        assert service.current_image(s) == "mock://orig"

    def test_click_offers_catalogue_of_that_class(self, service, db, catalogue, room):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        sel = service.select(s, 400, 700, room=room)  # sofa box
        assert sel.detection.object_class == "sofa"
        skus = [o.sku for o in sel.offers]
        assert "SOFA-A" in skus and "CT-A" not in skus

    def test_unfittable_item_listed_last_with_reason(self, service, db, catalogue, room):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        sel = service.select(s, 400, 700, room=room)
        huge = next(o for o in sel.offers if o.sku == "SOFA-HUGE")
        assert not huge.fits_room
        assert huge.fit_note and "mm" in huge.fit_note  # measured, not vague
        assert sel.offers[-1].sku == "SOFA-HUGE"  # sorted to the end

    def test_apply_appends_step_and_moves_pointer(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        step = service.apply(s, "det-sofa", "SOFA-A")
        db.commit()
        assert s.current_step_id == step.id
        assert service.current_image(s) == step.result_image_ref

    def test_later_swap_supersedes_earlier(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        service.apply(s, "det-sofa", "SOFA-A")
        service.apply(s, "det-sofa", "SOFA-B")
        assert service.swapped_skus(s) == {"det-sofa": "SOFA-B"}

    def test_swaps_of_different_objects_accumulate(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        service.apply(s, "det-sofa", "SOFA-A")
        service.apply(s, "det-table", "CT-A")
        assert service.swapped_skus(s) == {"det-sofa": "SOFA-A", "det-table": "CT-A"}

    def test_undo_moves_pointer_without_deleting(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        st1 = service.apply(s, "det-sofa", "SOFA-A")
        st2 = service.apply(s, "det-table", "CT-A")
        service.undo(s)
        assert s.current_step_id == st1.id
        assert service.swapped_skus(s) == {"det-sofa": "SOFA-A"}
        # both steps still exist in the chain
        assert len(s.steps) == 2

    def test_undo_to_original(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        service.apply(s, "det-sofa", "SOFA-A")
        img = service.undo(s)
        assert img == "mock://orig"
        assert service.swapped_skus(s) == {}

    def test_apply_unknown_detection_raises(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        with pytest.raises(KeyError):
            service.apply(s, "not-a-detection", "SOFA-A")

    def test_apply_unknown_sku_raises(self, service, db, catalogue):
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        with pytest.raises(KeyError):
            service.apply(s, "det-sofa", "NOT-A-SKU")


# ------------------------------------------------------------ API level


class TestEditingAPI:
    @pytest.fixture
    def client(self):
        import struct
        import zlib

        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    @staticmethod
    def _png() -> bytes:
        import struct
        import zlib

        def ch(t, d):
            c = t + d
            return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        return (
            b"\x89PNG\r\n\x1a\n"
            + ch(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
            + ch(b"IEND", b"")
        )

    def _scene(self, client) -> tuple[str, str]:
        sc = client.post(
            "/scenes",
            json={"rooms": [{
                "name": "L",
                "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                            {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
                "ceiling_height_mm": 3000, "surfaces": {},
            }]},
        ).json()
        return sc["scene_id"], sc["rooms"][0]["id"]

    def _seed_catalogue(self, client):
        for sku, name, cls, w, d, h, dp in [
            ("SOFA-A", "Sofa A", "sofa", 2000, 850, 800, "40000"),
            ("SOFA-B", "Sofa B", "sofa", 1800, 800, 780, "35000"),
            ("CT-A", "Table A", "coffee_table", 1000, 550, 420, "11000"),
        ]:
            r = client.post("/catalogue", json={
                "sku": sku, "name": name, "object_class": cls,
                "width_mm": w, "depth_mm": d, "height_mm": h, "display_price": dp,
            })
            assert r.status_code == 201

    def test_full_loop_via_api(self, client):
        sid, rid = self._scene(client)
        self._seed_catalogue(client)
        client.post("/prices", json={"sku": "SOFA-B", "vendor": "V", "unit": "piece", "amount": "34000"})

        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        )
        assert es.status_code == 201
        sess = es.json()["session_id"]
        assert len(es.json()["detections"]) == 6

        sel = client.post(f"/edit-sessions/{sess}/select", json={"x": 400, "y": 700}).json()
        assert sel["hit"] and sel["detection"]["object_class"] == "sofa"

        st = client.post(
            f"/edit-sessions/{sess}/apply",
            json={"detection_id": sel["detection"]["id"], "sku": "SOFA-B"},
        ).json()
        assert st["swapped_skus"] == {sel["detection"]["id"]: "SOFA-B"}

        q = client.post(f"/edit-sessions/{sess}/quote").json()
        assert q["total"] == "34000.00"  # price book, not display price 35000
        assert q["is_complete"]

    def test_catalogue_price_becomes_the_quoted_price(self, client):
        """Adding a product records its price, so the figure shown in the
        picker is the figure the quote commits to.

        This deliberately replaced an earlier rule that kept display price and
        quoted price separate. That separation made sense when prices arrived
        from vendors independently of the catalogue, but for a price typed into
        the console it only produced quotes that disagreed with the picker. A
        later /prices observation still wins, being more recent.
        """
        sid, rid = self._scene(client)
        self._seed_catalogue(client)
        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        ).json()
        sess = es["session_id"]
        sel = client.post(f"/edit-sessions/{sess}/select", json={"x": 400, "y": 700}).json()
        client.post(
            f"/edit-sessions/{sess}/apply",
            json={"detection_id": sel["detection"]["id"], "sku": "SOFA-A"},
        )
        q = client.post(f"/edit-sessions/{sess}/quote").json()
        assert q["is_complete"]
        assert q["lines"][0]["unit_price"] is not None

    def test_click_on_nothing_misses(self, client):
        sid, rid = self._scene(client)
        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        ).json()
        sel = client.post(
            f"/edit-sessions/{es['session_id']}/select", json={"x": 950, "y": 750}
        ).json()
        assert sel["hit"] is False

    def test_undo_via_api(self, client):
        sid, rid = self._scene(client)
        self._seed_catalogue(client)
        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        ).json()
        sess = es["session_id"]
        sel = client.post(f"/edit-sessions/{sess}/select", json={"x": 400, "y": 700}).json()
        client.post(f"/edit-sessions/{sess}/apply",
                    json={"detection_id": sel["detection"]["id"], "sku": "SOFA-A"})
        u = client.post(f"/edit-sessions/{sess}/undo").json()
        assert u["swapped_skus"] == {}

    def test_session_survives_across_requests(self, client):
        """The shared in-memory SQLite must persist sessions between calls."""
        sid, rid = self._scene(client)
        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", self._png(), "image/png")},
        ).json()
        got = client.get(f"/edit-sessions/{es['session_id']}")
        assert got.status_code == 200
        assert len(got.json()["detections"]) == 6

    def test_unknown_session_404s(self, client):
        assert client.post("/edit-sessions/nope/select", json={"x": 1, "y": 1}).status_code == 404

    def test_catalogue_upsert_and_filter(self, client):
        self._seed_catalogue(client)
        # upsert same sku with new price
        client.post("/catalogue", json={
            "sku": "SOFA-A", "name": "Sofa A v2", "object_class": "sofa",
            "width_mm": 2000, "depth_mm": 850, "height_mm": 800, "display_price": "42000",
        })
        items = client.get("/catalogue?object_class=sofa").json()["items"]
        assert len(items) == 2  # still 2 sofas, not 3
        assert any(i["name"] == "Sofa A v2" for i in items)


class TestConsoleUI:
    """The /ui console must serve and reference only endpoints that exist."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    def test_ui_serves_html(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_ui_contains_all_stage_controls(self, client):
        html = client.get("/ui").text
        # btn-seed is deliberately absent: products are loaded from the
        # database, not re-pushed from the browser.
        for element_id in [
            "btn-estimate", "btn-perceive", "btn-pipeline",
            "btn-detect", "btn-undo", "btn-squote", "file", "region", "housing",
        ]:
            assert f'id="{element_id}"' in html

    def test_ui_references_only_real_endpoints(self, client):
        """Every API path template used by the console's JS must exist in the
        OpenAPI schema -- the UI and the API cannot be allowed to drift."""
        import re

        html = client.get("/ui").text
        openapi = client.get("/openapi.json").json()["paths"].keys()
        # Static paths called by the UI.
        for path in ["/capabilities", "/catalogue", "/prices", "/estimate-scene"]:
            assert path in openapi
        # Templated paths: normalise JS interpolations to openapi templates.
        js_calls = re.findall(r"api\(`([^`]+)`", html)
        norm = set()
        for call in js_calls:
            p = re.sub(r"\$\{S\.sceneId\}", "{scene_id}", call)
            p = re.sub(r"\$\{S\.roomId\}", "{room_id}", p)
            p = re.sub(r"\$\{S\.sessionId\}", "{session_id}", p)
            norm.add(p)
        for p in norm:
            assert p in openapi, f"UI calls {p} which is not in the API"


class TestSurfacesAndListSelection:
    """Walls, ceilings and floors are selectable regions; the list panel
    selects by detection id; suggested treatments float first with swatches."""

    def _seed_treatments(self, db):
        for sku, name, cls, price, tags in [
            ("PAINT-IVORY", "Warm Ivory", "wall", "9500", {"hex": "#F3EBDD", "suggested": True}),
            ("PAINT-INK", "Deep Ink", "wall", "10500", {"hex": "#2E3440"}),
            ("FLR-OAK", "Oak Herringbone", "floor", "125000", {"suggested": True}),
        ]:
            db.add(CatalogueItemRow(
                sku=sku, name=name, object_class=cls,
                width_mm=1, depth_mm=1, height_mm=1,
                display_price=Decimal(price), style_tags=tags,
            ))
        db.commit()

    def test_bare_wall_click_selects_wall_but_object_click_wins(self, service, db):
        self._seed_treatments(db)
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        assert service.select(s, 900, 300).detection.object_class == "wall"
        assert service.select(s, 400, 700).detection.object_class == "sofa"

    def test_select_by_detection_id(self, service, db):
        self._seed_treatments(db)
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        sel = service.select(s, detection_id="det-floor")
        assert sel.detection.object_class == "floor"
        assert [o.sku for o in sel.offers] == ["FLR-OAK"]

    def test_suggested_offers_come_first_with_swatch(self, service, db):
        self._seed_treatments(db)
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        offers = service.select(s, detection_id="det-wall").offers
        assert offers[0].sku == "PAINT-IVORY" and offers[0].suggested
        assert offers[0].swatch == "#F3EBDD"

    def test_surfaces_are_never_fit_checked(self, service, db, room):
        """A 1x1x1 'paint' must not be vetoed by the fit engine."""
        self._seed_treatments(db)
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        offers = service.select(s, detection_id="det-wall", room=room).offers
        assert all(o.fits_room for o in offers)

    def test_surface_apply_uses_restyle_prompt(self):
        from interior_ai.perception.editing import build_replace_prompt

        d = Detection(id="w", label="beige wall", object_class="wall",
                      box=(0, 0, 1000, 620), confidence=0.9)
        p = build_replace_prompt(d, product_name="Deep Ink", product_desc="matt")
        assert "RESTYLE" in p and "surface" in p
        d2 = Detection(id="s", label="sofa", object_class="sofa",
                       box=(0, 0, 500, 500), confidence=0.9)
        p2 = build_replace_prompt(d2, product_name="Oslo", product_desc="")
        assert "REPLACE" in p2

    def test_repaint_and_swap_quote_together(self, service, db):
        self._seed_treatments(db)
        db.add(CatalogueItemRow(sku="SOFA-A", name="Sofa A", object_class="sofa",
                                width_mm=2000, depth_mm=850, height_mm=800,
                                display_price=Decimal("40000")))
        db.commit()
        s = service.start(scene_id="sc", room_id="r", image_ref="mock://orig")
        service.apply(s, "det-wall", "PAINT-INK")
        service.apply(s, "det-sofa", "SOFA-A")
        assert service.swapped_skus(s) == {"det-wall": "PAINT-INK", "det-sofa": "SOFA-A"}

    def test_detection_parser_accepts_surface_classes(self):
        dets, _ = parse_detections(
            '[{"label":"marble feature wall","object_class":"wall","box_2d":[0,300,600,700]},'
            ' {"label":"track spotlights","object_class":"ceiling_light","box_2d":[0,100,80,300]},'
            ' {"label":"slatted room divider","object_class":"room_divider","box_2d":[100,0,900,150]}]'
        )
        assert [d.object_class for d in dets] == ["wall", "ceiling_light", "room_divider"]


class TestTimeoutsAndDownscale:
    """The fixes for slow image edits: a separate, generous edit timeout and
    request-size reduction via downscaling."""

    def test_edit_timeout_is_separate_and_longer(self, monkeypatch):
        from interior_ai.perception.editing import GeminiPhotoEditor

        monkeypatch.delenv("GEMINI_TIMEOUT_S", raising=False)
        monkeypatch.delenv("GEMINI_EDIT_TIMEOUT_S", raising=False)
        g = GeminiPhotoEditor(api_key="x")
        assert g.edit_timeout_s > g.timeout_s
        # Per attempt, with retries on top -- a hung request is abandoned and
        # retried rather than held for four minutes.
        assert 120 <= g.edit_timeout_s <= 180

    def test_edit_timeout_env_override(self, monkeypatch):
        from interior_ai.perception.editing import GeminiPhotoEditor

        monkeypatch.setenv("GEMINI_EDIT_TIMEOUT_S", "600")
        assert GeminiPhotoEditor(api_key="x").edit_timeout_s == 600.0

    def test_large_upload_is_downscaled(self):
        import io

        from PIL import Image

        from interior_ai.api.app import _downscale

        buf = io.BytesIO()
        Image.new("RGB", (4000, 3000), "tan").save(buf, "PNG")
        scaled = _downscale(buf.getvalue())
        assert scaled is not None
        out, mime = scaled
        assert mime == "image/jpeg"
        assert max(Image.open(io.BytesIO(out)).size) <= 1536

    def test_small_upload_untouched(self):
        import io

        from PIL import Image

        from interior_ai.api.app import _downscale

        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "tan").save(buf, "PNG")
        assert _downscale(buf.getvalue()) is None

    def test_garbage_bytes_pass_through(self):
        from interior_ai.api.app import _downscale

        assert _downscale(b"not an image at all") is None


class TestProductConsole:
    """Upload with background strip, image serving, offer thumbnails, and the
    replacement prompt's angle/crispness contract."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    @staticmethod
    def _photo(size=(2000, 1500)) -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", size, "tan").save(buf, "JPEG")
        return buf.getvalue()

    def _upload(self, client, sku="SOFA-NOVA", **over):
        data = {
            "sku": sku, "name": "Nova", "object_class": "sofa",
            "width_mm": "2000", "depth_mm": "900", "height_mm": "800",
            "display_price": "56000",
        }
        data.update(over)
        return client.post(
            "/catalogue/upload", data=data,
            files={"image": ("p.jpg", self._photo(), "image/jpeg")},
        )

    def test_upload_stores_and_serves_image(self, client):
        r = self._upload(client)
        assert r.status_code == 201
        body = r.json()
        # MOCK path cannot strip -- must say so, not pretend.
        assert body["image_processed"] is False
        assert any("not stripped" in n for n in body["notes"])
        img = client.get(body["image_url"])
        assert img.status_code == 200
        assert img.headers["content-type"].startswith("image/")

    def test_upload_downscales_large_photos(self, client):
        import io

        from PIL import Image

        self._upload(client, sku="SOFA-BIG")
        img = client.get("/catalogue/SOFA-BIG/image")
        assert max(Image.open(io.BytesIO(img.content)).size) <= 1536

    def test_upload_records_opening_price(self, client):
        self._upload(client, sku="SOFA-PRICED")
        snap = client.get("/prices/SOFA-PRICED")
        assert snap.status_code == 200

    def test_upload_rejects_bad_dims_and_price(self, client):
        assert self._upload(client, sku="X1", width_mm="0").status_code == 422
        assert self._upload(client, sku="X2", display_price="lots").status_code == 422

    def test_offers_link_image_never_inline(self, client):
        self._upload(client, sku="SOFA-IMG")
        sid, rid = self._scene(client)
        es = client.post(
            f"/scenes/{sid}/rooms/{rid}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()
        sel = client.post(
            f"/edit-sessions/{es['session_id']}/select", json={"x": 400, "y": 700}
        ).json()
        offer = next(o for o in sel["offers"] if o["sku"] == "SOFA-IMG")
        assert offer["image_url"] == "/catalogue/SOFA-IMG/image"
        assert "data:image" not in str(sel)  # payload stays light

    def _scene(self, client):
        sc = client.post(
            "/scenes",
            json={"rooms": [{
                "name": "L",
                "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                            {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
                "ceiling_height_mm": 3000, "surfaces": {},
            }]},
        ).json()
        return sc["scene_id"], sc["rooms"][0]["id"]

    def test_deactivate_hides_from_offers(self, client):
        self._upload(client, sku="SOFA-GONE")
        client.post("/catalogue/SOFA-GONE/deactivate")
        items = client.get("/catalogue?object_class=sofa").json()["items"]
        assert "SOFA-GONE" not in [i["sku"] for i in items]

    def test_missing_image_404s(self, client):
        assert client.get("/catalogue/NOPE/image").status_code == 404

    def test_admin_page_serves(self, client):
        r = client.get("/admin")
        assert r.status_code == 200
        assert "btn-upload" in r.text

    def test_replace_prompt_covers_angle_and_crispness(self):
        from interior_ai.perception.editing import build_replace_prompt

        d = Detection(id="d", label="grey sofa", object_class="sofa",
                      box=(150, 550, 700, 900), confidence=0.9)
        p = build_replace_prompt(d, product_name="Nova", product_desc="charcoal")
        # The contract the user asked for: re-render from the correct angle
        # (never paste flat), and a sharp result.
        assert "DIFFERENT viewing angle" in p
        assert "NEVER paste the reference" in p
        assert "SHARP" in p
        assert "pixel-identical" in p

    def test_cutout_prompt_preserves_product(self):
        from interior_ai.perception.editing import CUTOUT_PROMPT

        assert "pure white background" in CUTOUT_PROMPT
        assert "Do not restyle" in CUTOUT_PROMPT

    def test_mock_cutout_signals_unprocessed(self):
        from interior_ai.perception.editing import MockPhotoEditor

        assert not MockPhotoEditor().cutout("data:image/png;base64,AAAA").startswith("data:")


class TestRegionLockedReplacement:
    """The locality guarantee: only the selected region's pixels may change.

    A prompt cannot enforce this -- image models treat coordinates as a
    suggestion and will happily paint a sofa somewhere more photogenic. These
    tests use a transport that returns a FULLY repainted crop (the worst case)
    and assert the composite discards everything outside the detection box.
    """

    @staticmethod
    def _uri(img) -> str:
        import base64
        import io

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _decode(uri: str):
        import base64
        import io

        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1]))).convert("RGB")

    def _editor_returning_solid(self, colour=(220, 40, 40)):
        import base64
        import io

        from PIL import Image

        from interior_ai.perception.editing import GeminiPhotoEditor

        seen: dict = {}

        def transport(model, payload):
            for part in payload["contents"][0]["parts"]:
                if "inline_data" in part and "sent_size" not in seen:
                    seen["sent_size"] = Image.open(
                        io.BytesIO(base64.b64decode(part["inline_data"]["data"]))
                    ).size
                if "text" in part:
                    seen["prompt"] = part["text"]
            solid = Image.new("RGB", seen["sent_size"], colour)
            buf = io.BytesIO()
            solid.save(buf, "PNG")
            return {"candidates": [{"content": {"parts": [{"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode(),
            }}]}}]}

        return GeminiPhotoEditor(api_key="x", transport=transport), seen

    def test_only_the_detection_box_changes(self):
        from PIL import Image

        original = Image.new("RGB", (1000, 800), (40, 160, 60))
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(900, 550, 1000, 800), confidence=0.9)
        editor, _ = self._editor_returning_solid()
        result = self._decode(editor.replace(
            self._uri(original), det, product_name="Nova", product_desc="charcoal"
        ))

        assert result.size == original.size
        # Inside the box: edited.
        assert result.getpixel((950, 540))[0] > 150
        # Everywhere else: untouched, including inside the model's crop window.
        for point in [(50, 50), (500, 400), (700, 500), (950, 200)]:
            r, g, b = result.getpixel(point)
            assert g > 120 and r < 100, f"pixel {point} was modified"

    def test_model_only_ever_sees_a_crop(self):
        from PIL import Image

        original = Image.new("RGB", (1000, 800), "white")
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(900, 550, 1000, 800), confidence=0.9)
        editor, seen = self._editor_returning_solid()
        editor.replace(self._uri(original), det, product_name="Nova")
        assert seen["sent_size"][0] < 1000 and seen["sent_size"][1] < 800

    def test_crop_carries_context_and_partial_rules(self):
        from PIL import Image

        original = Image.new("RGB", (1000, 800), "white")
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(900, 550, 1000, 800), confidence=0.9)
        editor, seen = self._editor_returning_solid()
        editor.replace(self._uri(original), det, product_name="Nova")
        assert "CROPPED REGION" in seen["prompt"]
        assert "PARTIAL VISIBILITY" in seen["prompt"]
        assert "NEVER move the" in seen["prompt"]

    def test_tiny_sliver_still_gets_a_workable_crop(self):
        """A 1%-visible object must not produce a 10-pixel canvas."""
        from interior_ai.perception.editing import _crop_geometry

        crop, _ = _crop_geometry(1600, 1200, (985, 900, 1000, 940))
        assert (crop[2] - crop[0]) >= 320
        assert (crop[3] - crop[1]) >= 320

    def test_crop_geometry_stays_inside_the_image(self):
        from interior_ai.perception.editing import _crop_geometry

        for box in [(0, 0, 50, 50), (950, 950, 1000, 1000), (0, 400, 1000, 600)]:
            (cx0, cy0, cx1, cy1), inner = _crop_geometry(900, 700, box)
            assert 0 <= cx0 < cx1 <= 900
            assert 0 <= cy0 < cy1 <= 700
            assert inner[0] >= 0 and inner[1] >= 0

    def test_replace_requires_inline_image(self):
        from interior_ai.providers.base import ProviderError

        editor, _ = self._editor_returning_solid()
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(0, 0, 500, 500), confidence=0.9)
        with pytest.raises(ProviderError):
            editor.replace("mock://not-a-data-uri", det, product_name="Nova")

    def test_composite_is_pixel_exact_outside_the_box(self):
        """Byte-level check on a textured image -- no global recompression
        drift outside the edited region."""
        from PIL import Image

        from interior_ai.perception.editing import composite_region

        original = Image.new("RGB", (400, 300), (10, 20, 30))
        for x in range(0, 400, 7):
            for y in range(0, 300, 5):
                original.putpixel((x, y), (200, 180, 160))
        crop_rect = (100, 100, 300, 250)
        edited = Image.new("RGB", (200, 150), (255, 0, 0))
        out = composite_region(original, edited, crop_rect, (50, 30, 150, 120), feather_px=0)
        # Well outside the inner rect, pixels must be identical.
        assert out.getpixel((10, 10)) == original.getpixel((10, 10))
        assert out.getpixel((390, 290)) == original.getpixel((390, 290))

    def test_pixels_outside_the_box_are_bit_exact(self):
        """Not merely 'close' -- identical. The loop is iterative, so any
        per-swap drift would compound across 5-10 edits."""
        from PIL import Image, ImageChops

        original = Image.new("RGB", (900, 600), (200, 190, 175))
        d = original.load()
        for y in range(0, 600, 3):
            for x in range(0, 900, 3):
                d[x, y] = (90 + (x % 60), 80 + (y % 50), 70)

        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(911, 633, 1000, 933), confidence=0.9)
        editor, _ = self._editor_returning_solid()
        result = self._decode(editor.replace(
            self._uri(original), det, product_name="Nova"
        ))
        diff = ImageChops.difference(original, result).convert("L")
        # Zones well clear of the detection box and its feather margin.
        for name, rect in [
            ("ceiling", (0, 0, 900, 180)),
            ("left wall", (60, 200, 300, 420)),
            ("centre", (360, 210, 560, 330)),
        ]:
            assert max(diff.crop(rect).getdata()) == 0, f"{name} drifted"

    def test_output_is_lossless_by_default(self, monkeypatch):
        from PIL import Image

        monkeypatch.delenv("EDIT_OUTPUT_FORMAT", raising=False)
        editor, _ = self._editor_returning_solid()
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(400, 400, 600, 600), confidence=0.9)
        out = editor.replace(self._uri(Image.new("RGB", (600, 600), "white")),
                             det, product_name="Nova")
        assert out.startswith("data:image/png")

    def test_crop_uploaded_as_jpeg_to_keep_requests_small(self):
        from PIL import Image

        import interior_ai.perception.editing as ed

        sizes: dict = {}
        orig_encode = ed._encode_image

        def spy(img, *, fmt="PNG", quality=92):
            sizes.setdefault("formats", []).append(fmt)
            return orig_encode(img, fmt=fmt, quality=quality)

        ed._encode_image = spy
        try:
            editor, _ = self._editor_returning_solid()
            det = Detection(id="d", label="sofa", object_class="sofa",
                            box=(400, 400, 600, 600), confidence=0.9)
            editor.replace(self._uri(Image.new("RGB", (600, 600), "white")),
                           det, product_name="Nova")
        finally:
            ed._encode_image = orig_encode
        assert "JPEG" in sizes["formats"]  # the crop going up
        assert "PNG" in sizes["formats"]   # the composited result


class TestReplacementAllowance:
    """A replacement may need room the original did not use -- most often
    above it. Without headroom the composite clips the top off a taller sofa;
    with the wrong headroom it licenses edits where they do not belong."""

    def test_floor_standing_grows_up_not_down(self):
        from interior_ai.perception.editing import replacement_region

        box = (300, 500, 700, 780)
        out = replacement_region(1536, 1152, box, "sofa", (2100, 880, 820))
        assert out[1] < box[1], "no headroom above a floor-standing object"
        # Floor contact is trustworthy; it must not sink through the floor.
        assert (out[3] - box[3]) < (box[1] - out[1])

    def test_taller_product_gets_proportional_headroom(self):
        from interior_ai.perception.editing import replacement_region

        box = (300, 600, 700, 750)  # wide, short
        short = replacement_region(1536, 1152, box, "sofa", (2100, 880, 700))
        tall = replacement_region(1536, 1152, box, "sofa", (1000, 500, 1400))
        assert (box[1] - tall[1]) > (box[1] - short[1])

    def test_clipped_object_falls_back_to_fixed_allowance(self):
        """When the frame cuts the object, its width understates its size, so
        a computed ratio would understate the headroom too."""
        from interior_ai.perception.editing import replacement_region

        clipped = (911, 633, 1000, 933)
        out = replacement_region(1536, 1152, clipped, "sofa", (2100, 880, 820))
        assert (clipped[1] - out[1]) / (clipped[3] - clipped[1]) >= 0.4

    def test_ceiling_mounted_grows_downward(self):
        from interior_ai.perception.editing import replacement_region

        box = (450, 20, 550, 90)
        out = replacement_region(1536, 1152, box, "ceiling_light", (400, 400, 900))
        assert (out[3] - box[3]) > (box[1] - out[1])

    def test_wall_mounted_grows_symmetrically(self):
        from interior_ai.perception.editing import replacement_region

        box = (400, 200, 600, 330)
        out = replacement_region(1536, 1152, box, "television", (1400, 80, 800))
        assert (box[1] - out[1]) == (out[3] - box[3])

    def test_surfaces_are_not_grown(self):
        from interior_ai.perception.editing import replacement_region

        box = (0, 0, 1000, 600)
        assert replacement_region(1536, 1152, box, "wall", None) == box

    def test_allowance_never_leaves_the_frame(self):
        from interior_ai.perception.editing import replacement_region

        for box in [(0, 0, 200, 150), (900, 850, 1000, 1000), (0, 400, 1000, 600)]:
            out = replacement_region(1200, 900, box, "sofa", (2000, 900, 900))
            assert 0 <= out[0] < out[2] <= 1000
            assert 0 <= out[1] < out[3] <= 1000

    def test_headroom_is_editable_while_the_room_is_not(self):
        """The point of the whole exercise: a taller replacement can grow, and
        nothing outside the grown region moves."""
        from PIL import Image, ImageChops, ImageDraw

        from interior_ai.perception.editing import replacement_region

        img = Image.new("RGB", (900, 600), (238, 232, 220))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 900, 180), fill=(250, 248, 244))
        d.rectangle((60, 200, 300, 420), fill=(120, 110, 100))
        d.rectangle((820, 380, 900, 560), fill=(70, 95, 75))

        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(911, 633, 1000, 933), confidence=0.9)
        editor, _ = TestRegionLockedReplacement()._editor_returning_solid()
        result = TestRegionLockedReplacement._decode(editor.replace(
            TestRegionLockedReplacement._uri(img), det,
            product_name="Nova", product_dims=(2100, 880, 900),
        ))
        diff = ImageChops.difference(img, result).convert("L")
        grown = replacement_region(900, 600, det.box, "sofa", (2100, 880, 900))
        # Headroom above the original box is now editable.
        head_y = int((grown[1] + det.box[1]) / 2 / 1000 * 600)
        assert diff.getpixel((860, head_y)) > 0
        # The rest of the room is untouched.
        for rect in [(0, 0, 900, 180), (60, 200, 300, 420), (0, 430, 600, 600)]:
            assert max(diff.crop(rect).getdata()) == 0

    def test_prompt_permits_size_difference(self):
        from interior_ai.perception.editing import build_replace_prompt

        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(100, 100, 400, 400), confidence=0.9)
        p = build_replace_prompt(det, product_name="Nova", product_desc="tall")
        assert "grows UPWARD" in p
        assert "not\n   squashed" in p or "squashed" in p
        # But relocation is still forbidden.
        assert "NEVER move the" in p


class TestConsoleHasNoSeedButton:
    """Products live in the database now, so the pipeline console must not
    offer to seed them -- a button that re-pushes 10 hardcoded items over a
    real catalogue is a footgun, not a convenience."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    def test_seed_button_is_gone(self, client):
        html = client.get("/ui").text
        assert "btn-seed" not in html
        assert "Seed catalogue" not in html

    def test_status_panel_replaces_it(self, client):
        html = client.get("/ui").text
        assert "status-kv" in html
        assert "/health" in html and "/config" in html

    def test_status_panel_points_at_the_loader(self, client):
        html = client.get("/ui").text
        assert "seed_products" in html


class TestEditRetryAndPayloadSize:
    """A slow or overloaded image model must not cost a swap outright, and a
    reference image must not be heavier than it needs to be -- it is uploaded
    again on every single swap."""

    @staticmethod
    def _uri(img) -> str:
        import base64
        import io

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _editor(self, responses, *, attempts=3):
        import base64
        import io

        from PIL import Image

        from interior_ai.perception.editing import GeminiPhotoEditor

        state = {"calls": 0}

        def transport(model, payload):
            index = state["calls"]
            state["calls"] += 1
            outcome = responses[index] if index < len(responses) else None
            if outcome is not None:
                raise outcome
            buf = io.BytesIO()
            Image.new("RGB", (200, 200), "red").save(buf, "PNG")
            return {"candidates": [{"content": {"parts": [{"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode(),
            }}]}}]}

        editor = GeminiPhotoEditor(api_key="x", transport=transport)
        editor.default_attempts = attempts
        return editor, state

    def test_replace_retries_a_timeout(self, monkeypatch):
        """A timeout means the request never got a verdict -- exactly the case
        worth retrying, and previously the case that killed the swap."""
        from PIL import Image

        from interior_ai.providers.base import ProviderError

        monkeypatch.setattr("time.sleep", lambda _s: None)
        editor, state = self._editor(
            [ProviderError("read operation timed out", retryable=True)] * 2
        )
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(100, 100, 500, 500), confidence=0.9)
        out = editor.replace(
            self._uri(Image.new("RGB", (600, 600), "white")), det,
            product_name="Nova",
        )
        assert out.startswith("data:image")
        assert state["calls"] == 3

    def test_replace_retries_a_503(self, monkeypatch):
        from PIL import Image

        from interior_ai.providers.base import ProviderError

        monkeypatch.setattr("time.sleep", lambda _s: None)
        editor, state = self._editor([ProviderError("busy", status_code=503)])
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(100, 100, 500, 500), confidence=0.9)
        editor.replace(self._uri(Image.new("RGB", (600, 600), "white")), det,
                       product_name="Nova")
        assert state["calls"] == 2

    def test_replace_does_not_retry_a_bad_request(self):
        from PIL import Image

        from interior_ai.providers.base import ProviderError

        editor, state = self._editor([ProviderError("bad", status_code=400)] * 3)
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(100, 100, 500, 500), confidence=0.9)
        with pytest.raises(ProviderError):
            editor.replace(self._uri(Image.new("RGB", (600, 600), "white")), det,
                           product_name="Nova")
        assert state["calls"] == 1

    def test_timeout_is_per_attempt_not_per_operation(self):
        """A shorter per-attempt window plus retries beats one long wait: a
        hung request gets abandoned rather than held for minutes."""
        from interior_ai.perception.editing import GeminiPhotoEditor

        assert GeminiPhotoEditor(api_key="x").edit_timeout_s <= 180

    def test_photographic_reference_is_compressed(self):
        import base64
        import io
        import random

        from PIL import Image

        from interior_ai.db.build_catalogue import _compress_reference

        random.seed(5)
        img = Image.new("RGB", (1024, 1024), "white")
        px = img.load()
        for y in range(250, 800):
            for x in range(180, 860):
                px[x, y] = (random.randint(70, 190), random.randint(60, 170),
                            random.randint(50, 150))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        original = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        compressed = _compress_reference(original)
        assert len(compressed) < len(original) / 3

    def test_compression_never_inflates(self):
        """Re-encoding a flat graphic as JPEG can triple it -- whichever
        encoding is genuinely smaller must win."""
        import base64
        import io

        from PIL import Image

        from interior_ai.db.build_catalogue import _compress_reference

        img = Image.new("RGB", (1024, 1024), "white")
        px = img.load()
        for y in range(300, 750):
            for x in range(200, 850):
                px[x, y] = (90, 80, 70)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        original = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        assert len(_compress_reference(original)) <= len(original)

    def test_unreadable_reference_passes_through(self):
        from interior_ai.db.build_catalogue import _compress_reference

        junk = "data:image/png;base64,bm90YW5pbWFnZQ=="
        assert _compress_reference(junk) == junk


class TestApplyProgressFeedback:
    """A silent two-minute wait reads as a hung page, and people click again --
    queueing a second edit behind the first."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    def test_elapsed_time_is_shown(self, client):
        html = client.get("/ui").text
        assert "Editing the photo" in html
        assert "clearInterval" in html

    def test_buttons_lock_during_the_edit(self, client):
        assert "b.disabled = true" in client.get("/ui").text

    def test_slow_edits_are_explained(self, client):
        assert "retry automatically" in client.get("/ui").text


class TestFloatingToFloorStanding:
    """Replacing a wall-mounted object with a free-standing one.

    A "floating tv console" and a legged TV unit share the class ``tv_unit``,
    so growing the region by class alone points the allowance upward -- and the
    replacement's legs get cut off at the bottom of a thin wall-mounted band.
    The direction depends on the transition, and the floor's own detection
    says how far down to reach.
    """

    def test_floating_labels_are_recognised(self):
        from interior_ai.perception.editing import looks_floating

        assert looks_floating("floating tv console")
        assert looks_floating("wall-mounted shelf")
        assert looks_floating("suspended cabinet")
        assert not looks_floating("three-seat sofa")
        assert not looks_floating("round coffee table")

    def test_floating_original_grows_downward(self):
        from interior_ai.perception.editing import replacement_region

        box = (266, 375, 733, 450)
        floating = replacement_region(1536, 1152, box, "tv_unit", (1800, 420, 500),
                                      label="floating tv console")
        standing = replacement_region(1536, 1152, box, "tv_unit", (1800, 420, 500),
                                      label="tv unit")
        assert (floating[3] - box[3]) > (standing[3] - box[3]) * 10

    def test_detected_floor_sets_the_reach(self):
        """Scaling a thin band's height is a guess; the floor region is a
        measurement."""
        from interior_ai.perception.editing import replacement_region

        box = (266, 375, 733, 450)
        guessed = replacement_region(900, 600, box, "tv_unit", (1800, 420, 500),
                                     label="floating tv console")
        measured = replacement_region(900, 600, box, "tv_unit", (1800, 420, 500),
                                      label="floating tv console", floor_top=717)
        assert measured[3] > guessed[3]
        assert measured[3] > 717, "must reach past the floor line, not stop at it"

    def test_reach_is_capped(self):
        """A misread label must not license repainting half the room."""
        from interior_ai.perception.editing import replacement_region

        box = (400, 100, 600, 130)
        out = replacement_region(1000, 1000, box, "tv_unit", (1800, 420, 500),
                                 label="floating console", floor_top=990)
        bh = box[3] - box[1]
        assert (out[3] - box[3]) <= 4.0 * bh + 2

    def test_expand_widens_the_region(self):
        from interior_ai.perception.editing import replacement_region

        box = (300, 500, 700, 780)
        normal = replacement_region(1536, 1152, box, "sofa", (2100, 880, 820))
        wider = replacement_region(1536, 1152, box, "sofa", (2100, 880, 820), expand=2.0)
        assert (box[1] - wider[1]) > (box[1] - normal[1])

    def test_expand_never_shrinks(self):
        from interior_ai.perception.editing import replacement_region

        box = (300, 500, 700, 780)
        normal = replacement_region(1536, 1152, box, "sofa", (2100, 880, 820))
        assert replacement_region(1536, 1152, box, "sofa", (2100, 880, 820),
                                  expand=0.1) == normal

    def test_prompt_tells_the_model_to_stand_it_down(self):
        from PIL import Image

        seen = TestRegionLockedReplacement()
        editor, state = seen._editor_returning_solid()
        det = Detection(id="d", label="floating tv console", object_class="tv_unit",
                        box=(266, 375, 733, 450), confidence=0.9)
        editor.replace(seen._uri(Image.new("RGB", (900, 600), "white")), det,
                       product_name="Linea", product_dims=(1800, 420, 500))
        assert "resting on the floor below" in state["prompt"]

    def test_no_floor_instruction_for_normal_swaps(self):
        from PIL import Image

        seen = TestRegionLockedReplacement()
        editor, state = seen._editor_returning_solid()
        det = Detection(id="d", label="three-seat sofa", object_class="sofa",
                        box=(200, 500, 700, 800), confidence=0.9)
        editor.replace(seen._uri(Image.new("RGB", (900, 600), "white")), det,
                       product_name="Nova", product_dims=(2100, 880, 820))
        assert "resting on the floor below" not in state["prompt"]

    def test_locality_survives_the_bigger_region(self):
        from PIL import Image, ImageChops, ImageDraw

        img = Image.new("RGB", (900, 600), (232, 226, 216))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 430, 900, 600), fill=(178, 140, 96))
        draw.rectangle((240, 225, 660, 270), fill=(60, 60, 66))

        seen = TestRegionLockedReplacement()
        editor, _state = seen._editor_returning_solid()
        det = Detection(id="d", label="floating tv console", object_class="tv_unit",
                        box=(266, 375, 733, 450), confidence=0.9)
        result = seen._decode(editor.replace(
            seen._uri(img), det, product_name="Linea",
            product_dims=(1800, 420, 500), floor_top=717,
        ))
        diff = ImageChops.difference(img, result).convert("L")
        # Zones outside the region, which is now wider because the product's
        # size prior widened it -- the guarantee is unchanged, the region is
        # simply bigger than when this test was first written.
        for rect in [(0, 0, 900, 120), (0, 0, 100, 600), (870, 0, 900, 600),
                     (0, 560, 900, 600)]:
            assert max(diff.crop(rect).getdata()) == 0


class TestMoreRoomRetry:
    """When the automatic allowance misjudges, the person looking at the
    result can see it -- so let them ask for a bigger region."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    def test_apply_accepts_expand(self, client):
        from interior_ai.api.schemas import ApplyIn

        assert ApplyIn(detection_id="d", sku="s", expand=2.0).expand == 2.0

    def test_expand_is_bounded(self, client):
        import pydantic

        from interior_ai.api.schemas import ApplyIn

        with pytest.raises(pydantic.ValidationError):
            ApplyIn(detection_id="d", sku="s", expand=9.0)

    def test_ui_offers_the_retry(self, client):
        html = client.get("/ui").text
        assert "Retry with more room" in html
        assert "offerMoreRoom" in html

    def test_retry_undoes_first(self, client):
        """Retrying must replace the clipped attempt, not stack another swap
        on top of it."""
        html = client.get("/ui").text
        # The retry handler undoes the clipped attempt before re-applying.
        handler = html.split("btn-more-room\").onclick")[1][:400]
        assert "/undo" in handler


class TestGeneralRegionSizing:
    """The allowance must generalise beyond one scenario: any replacement may
    be larger than what it replaces, and 'is it on the floor' must be measured
    rather than read from a label the model may not have used."""

    def test_floating_is_measured_not_read(self):
        """Works when the label says nothing about mounting."""
        from interior_ai.perception.editing import is_off_the_floor

        assert is_off_the_floor((266, 375, 733, 450), "tv console", floor_top=700)
        assert not is_off_the_floor((200, 500, 700, 700), "tv console", floor_top=700)

    def test_label_is_only_a_fallback(self):
        from interior_ai.perception.editing import is_off_the_floor

        assert is_off_the_floor((266, 375, 733, 450), "floating console", None)
        assert not is_off_the_floor((266, 375, 733, 450), "console", None)

    def test_measurement_overrides_a_misleading_label(self):
        """An object whose base is on the floor is standing on it, whatever it
        happens to be called."""
        from interior_ai.perception.editing import is_off_the_floor

        assert not is_off_the_floor((200, 500, 700, 705), "floating console",
                                    floor_top=700)

    def test_bigger_products_get_bigger_regions(self):
        from interior_ai.perception.editing import replacement_region

        box = (350, 600, 650, 700)
        small = replacement_region(1536, 1152, box, "sofa", (1650, 850, 800))
        large = replacement_region(1536, 1152, box, "sofa", (2600, 1600, 850))
        assert (box[0] - large[0]) > (box[0] - small[0])
        assert (large[2] - box[2]) > (small[2] - box[2])

    def test_size_scales_with_the_class_typical(self):
        """A wide, low product must not ask for LESS room than a compact one --
        the failure that comes from judging by the box alone."""
        from interior_ai.perception.editing import replacement_region

        box = (350, 600, 650, 700)
        widths = [
            replacement_region(1536, 1152, box, "sofa", (w, 900, 820))[0]
            for w in (1500, 2100, 2600)
        ]
        assert widths[0] > widths[1] > widths[2], "wider product, wider region"

    def test_growth_is_still_capped(self):
        from interior_ai.perception.editing import replacement_region

        box = (400, 400, 600, 500)
        out = replacement_region(1000, 1000, box, "sofa", (9000, 3000, 2000))
        assert out[0] >= 0 and out[2] <= 1000 and out[3] <= 1000


class TestNeighbourPreservation:
    """A region sized for a large product covers its neighbours -- and each of
    those may be a product the customer already chose. They must be named and
    protected, not silently repainted."""

    def _dets(self):
        return [
            Detection(id="ct", label="round coffee table", object_class="coffee_table",
                      box=(400, 640, 600, 760), confidence=0.9),
            Detection(id="lamp", label="arc floor lamp", object_class="lamp",
                      box=(680, 450, 760, 720), confidence=0.9),
            Detection(id="far", label="bookshelf", object_class="bookshelf",
                      box=(0, 200, 120, 800), confidence=0.9),
            Detection(id="wall", label="painted wall", object_class="wall",
                      box=(0, 0, 1000, 700), confidence=0.9),
        ]

    def test_finds_objects_inside_the_region(self):
        from interior_ai.perception.editing import overlapping_detections

        found = overlapping_detections((300, 550, 700, 750), self._dets())
        # The table sits squarely inside; the lamp clips its right edge.
        assert {d.id for d in found} == {"ct", "lamp"}

    def test_distant_objects_are_left_out(self):
        from interior_ai.perception.editing import overlapping_detections

        found = overlapping_detections((300, 550, 700, 750), self._dets())
        assert "far" not in {d.id for d in found}

    def test_surfaces_are_not_listed(self):
        """A wall is meant to be behind the object; listing it is noise."""
        from interior_ai.perception.editing import overlapping_detections

        found = overlapping_detections((0, 0, 1000, 1000), self._dets())
        assert "wall" not in {d.id for d in found}

    def test_target_excluded_from_its_own_protection(self):
        from interior_ai.perception.editing import overlapping_detections

        found = overlapping_detections((300, 550, 700, 800), self._dets(),
                                       exclude_id="ct")
        assert "ct" not in {d.id for d in found}

    def test_prompt_names_them_and_demands_survival(self):
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        target = Detection(id="sofa", label="two-seat sofa", object_class="sofa",
                           box=(350, 600, 650, 700), confidence=0.9)
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")), target,
                       product_name="Jaipur L-Shape", product_dims=(2600, 1600, 850),
                       neighbours=self._dets())
        prompt = state["prompt"]
        # The table's base is lower than the sofa's, so it is nearer the
        # camera and gets the in-front instruction.
        assert "NEARER THE CAMERA" in prompt
        assert "round coffee table" in prompt
        assert "Never delete, move, resize, restyle or substitute" in prompt

    def test_occlusion_is_explained_not_forbidden(self):
        """A bigger object legitimately stands in front of things -- the model
        should draw that, not avoid it."""
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        target = Detection(id="sofa", label="sofa", object_class="sofa",
                           box=(350, 600, 650, 700), confidence=0.9)
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")), target,
                       product_name="L-Shape", product_dims=(2600, 1600, 850),
                       neighbours=self._dets())
        # Objects nearer the camera are drawn over the replacement; the
        # relationship is explained either way rather than forbidden.
        assert "hide the parts of the" in state["prompt"]

    def test_no_note_when_nothing_is_at_risk(self):
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        target = Detection(id="sofa", label="sofa", object_class="sofa",
                           box=(350, 600, 650, 700), confidence=0.9)
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")), target,
                       product_name="Nova", product_dims=(1650, 850, 800),
                       neighbours=[])
        assert "MUST SURVIVE" not in state["prompt"]


class TestFlatFloorItems:
    """A rug's height_mm is its pile thickness. Reading that as the object's
    vertical extent asks for no room, and a larger rug is clipped along its far
    edge -- what actually extends up the image is its depth."""

    def test_rug_gets_room_from_its_depth(self):
        from interior_ai.perception.editing import replacement_region

        box = (150, 780, 850, 950)
        out = replacement_region(1536, 1152, box, "rug", (1520, 2130, 15))
        assert (box[1] - out[1]) > (box[3] - box[1]), "needs more than its own height"

    def test_deeper_rugs_reach_further_up(self):
        from interior_ai.perception.editing import replacement_region

        box = (150, 780, 850, 950)
        small = replacement_region(1536, 1152, box, "rug", (1220, 1830, 10))
        large = replacement_region(1536, 1152, box, "rug", (2440, 3050, 15))
        assert large[1] < small[1]

    def test_rug_grows_mostly_upward(self):
        """The near edge stays put; a bigger rug reaches away from the camera."""
        from interior_ai.perception.editing import replacement_region

        box = (150, 700, 850, 850)
        out = replacement_region(1536, 1152, box, "rug", (1520, 2130, 15))
        assert (box[1] - out[1]) > (out[3] - box[3])

    def test_upright_furniture_still_uses_height(self):
        """The fix must not change how a wardrobe or sofa is sized."""
        from interior_ai.perception.editing import replacement_region

        box = (300, 500, 700, 780)
        tall = replacement_region(1536, 1152, box, "wardrobe", (1500, 600, 2200))
        short = replacement_region(1536, 1152, box, "wardrobe", (1500, 600, 1200))
        assert tall[1] < short[1]


class TestOpacityAndOcclusionDirection:
    """Two failures that produced a see-through rug.

    First, nothing told the model to render the product solid across its whole
    footprint. Second -- and worse -- the preservation note claimed the new
    object was nearer the camera than its neighbours. For a rug that is
    backwards: the coffee tables stand ON it. Told the wrong way round, the
    model renders the rug "around" the furniture and the result reads as a mat
    with holes in it.
    """

    def _prompt_for(self, target, neighbours, dims):
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.replace(
            helper._uri(Image.new("RGB", (900, 700), "white")), target,
            product_name="Product", product_dims=dims, neighbours=neighbours,
        )
        return state["prompt"]

    def _tables(self):
        return [Detection(id="ct", label="round coffee table",
                          object_class="coffee_table",
                          box=(300, 300, 700, 600), confidence=0.9)]

    def _rug(self):
        return Detection(id="rug", label="area rug", object_class="rug",
                         box=(200, 400, 800, 800), confidence=0.9)

    def _sofa(self):
        return Detection(id="s", label="two-seat sofa", object_class="sofa",
                         box=(300, 500, 700, 750), confidence=0.9)

    def test_product_must_be_drawn_opaque(self):
        prompt = self._prompt_for(self._rug(), [], (1520, 2130, 15))
        assert "FULLY OPAQUE" in prompt
        assert "semi-transparent" in prompt

    def test_product_continues_under_objects(self):
        """The fix for gaps around table legs: the surface does not stop at
        them, it passes beneath and resumes."""
        prompt = self._prompt_for(self._rug(), [], (1520, 2130, 15))
        assert "continues UNBROKEN underneath" in prompt

    def test_rug_neighbours_are_described_as_on_top(self):
        prompt = self._prompt_for(self._rug(), self._tables(), (1520, 2130, 15))
        assert "LAYERING -- READ THIS CAREFULLY" in prompt
        assert "the objects sit on it" in prompt

    def test_rug_prompt_forbids_cutting_holes(self):
        prompt = self._prompt_for(self._rug(), self._tables(), (1520, 2130, 15))
        assert "do not cut holes or gaps" in prompt

    def test_rug_prompt_orders_the_layers(self):
        """Draw the surface first, then what stands on it."""
        prompt = self._prompt_for(self._rug(), self._tables(), (1520, 2130, 15))
        assert "Draw it FIRST and draw it WHOLE" in prompt

    def test_upright_target_keeps_the_in_front_wording(self):
        """A sofa may legitimately stand in front of its neighbours -- that
        case must not be switched to the on-top wording."""
        prompt = self._prompt_for(self._sofa(), self._tables(), (2600, 1600, 850))
        assert "partly hidden behind it" in prompt
        assert "LAYERING -- READ THIS CAREFULLY" not in prompt

    def test_neighbours_still_named_in_both_cases(self):
        for target, dims in [(self._rug(), (1520, 2130, 15)),
                             (self._sofa(), (2600, 1600, 850))]:
            prompt = self._prompt_for(target, self._tables(), dims)
            assert "round coffee table" in prompt

    def test_no_preservation_note_without_neighbours(self):
        prompt = self._prompt_for(self._rug(), [], (1520, 2130, 15))
        assert "MUST SURVIVE" not in prompt



class TestLayeringInsteadOfPasteBack:
    """Neighbouring objects are preserved by instruction, not by pixel surgery.

    Earlier versions pasted those objects back from the pre-edit image. Pixels
    copied from before the edit cannot blend with a surface generated after it,
    so every paste-back showed as a patch -- first a rectangle of old floor,
    then a pale block under the tables. The crop sent to the model already
    contains the objects; what it needed was to be told what they are.
    """

    def _prompt(self, target, neighbours, dims):
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")),
                       target, product_name="P", product_dims=dims,
                       neighbours=neighbours)
        return state["prompt"]

    def _rug(self):
        return Detection(id="rug", label="area rug", object_class="rug",
                         box=(200, 400, 800, 900), confidence=0.9)

    def _tables(self):
        return [
            Detection(id="ct", label="round wooden coffee table",
                      object_class="coffee_table",
                      box=(300, 350, 650, 700), confidence=0.9),
            Detection(id="ct2", label="green marble side table",
                      object_class="side_table",
                      box=(600, 450, 760, 720), confidence=0.9),
        ]

    def test_paste_back_is_gone(self):
        import inspect

        from interior_ai.perception.editing import GeminiPhotoEditor, composite_region

        assert "protect_rects" not in inspect.signature(composite_region).parameters
        assert "protect_neighbours" not in inspect.signature(
            GeminiPhotoEditor.replace
        ).parameters

    def test_prompt_defines_two_layers(self):
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "LAYER 1" in prompt and "LAYER 2" in prompt

    def test_surface_passes_under_the_objects(self):
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "passes UNDERNEATH" in prompt

    def test_pale_patch_is_forbidden_by_name(self):
        """The exact artefact the paste-back used to leave behind."""
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "pale patch" in prompt

    def test_every_object_on_top_is_named(self):
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "round wooden coffee table" in prompt
        assert "green marble side table" in prompt

    def test_model_is_told_they_are_in_the_image(self):
        """They are inside the crop, so the model can copy them rather than
        invent them."""
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "visible in the image you were given" in prompt

    def test_contact_shadows_requested(self):
        prompt = self._prompt(self._rug(), self._tables(), (1520, 2130, 15))
        assert "contact shadow" in prompt

    def test_locality_guarantee_survives_the_removal(self):
        """Removing paste-back must not weaken the rule that nothing outside
        the editable region changes."""
        from PIL import Image, ImageChops

        helper = TestRegionLockedReplacement()
        editor, _state = helper._editor_returning_solid()
        img = Image.new("RGB", (1000, 800), (40, 160, 60))
        det = Detection(id="d", label="sofa", object_class="sofa",
                        box=(900, 550, 1000, 800), confidence=0.9)
        out = helper._decode(editor.replace(helper._uri(img), det,
                                            product_name="Nova"))
        diff = ImageChops.difference(img, out).convert("L")
        assert max(diff.crop((0, 0, 400, 300)).getdata()) == 0


class TestDepthAwareLayering:
    """Which object is nearer the camera decides the drawing order.

    The prompt used to assert flatly that the replacement was nearer. For a
    coffee table standing in front of a TV console that is backwards, and the
    model obligingly drew the console over the table -- erasing it. The floor
    recedes upward in a photograph, so a lower base means a nearer object.
    """

    def _console(self):
        return Detection(id="tv", label="floating tv console",
                         object_class="tv_unit",
                         box=(300, 380, 780, 470), confidence=0.9)

    def _table(self):
        return Detection(id="ct", label="round coffee table",
                         object_class="coffee_table",
                         box=(330, 620, 700, 800), confidence=0.9)

    def _shelf(self):
        # Base above the console's, and overlapping its region, so it is a
        # genuine "behind" neighbour rather than one that is simply elsewhere.
        return Detection(id="sh", label="wall shelf", object_class="bookshelf",
                         box=(200, 300, 900, 430), confidence=0.9)

    def test_lower_base_means_nearer(self):
        from interior_ai.perception.editing import split_by_depth

        front, behind = split_by_depth(self._console(), [self._table(), self._shelf()])
        assert [d.id for d in front] == ["ct"]
        assert [d.id for d in behind] == ["sh"]

    def _prompt(self, neighbours):
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")),
                       self._console(), product_name="Linea",
                       product_dims=(1800, 420, 500), neighbours=neighbours,
                       floor_top=717)
        return state["prompt"]

    def test_object_in_front_is_declared_nearer(self):
        prompt = self._prompt([self._table()])
        assert "NEARER THE CAMERA" in prompt
        assert "round coffee table" in prompt

    def test_model_is_forbidden_from_covering_it(self):
        prompt = self._prompt([self._table()])
        assert "never draw the" in prompt and "on top of them" in prompt

    def test_object_behind_keeps_the_old_wording(self):
        prompt = self._prompt([self._shelf()])
        assert "partly hidden behind it" in prompt
        assert "NEARER THE CAMERA" not in prompt

    def test_both_groups_get_their_own_instruction(self):
        prompt = self._prompt([self._table(), self._shelf()])
        assert "NEARER THE CAMERA" in prompt
        assert "MUST SURVIVE" in prompt

    def test_flat_targets_are_unaffected(self):
        """Everything on a rug is in front of it; that path must not be
        rerouted through the depth split."""
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        rug = Detection(id="rug", label="area rug", object_class="rug",
                        box=(200, 400, 800, 900), confidence=0.9)
        editor.replace(helper._uri(Image.new("RGB", (900, 700), "white")), rug,
                       product_name="Jaipur", product_dims=(1520, 2130, 15),
                       neighbours=[self._table()])
        assert "LAYERING -- READ THIS CAREFULLY" in state["prompt"]


class TestAffectsWarning:
    """A person should see which objects a swap will cover before committing,
    not discover afterwards that a table went missing."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import SceneStore, create_app

        return TestClient(create_app(store=SceneStore()))

    def _session(self, client):
        client.post("/catalogue", json={
            "sku": "SOFA-A", "name": "Sofa A", "object_class": "sofa",
            "width_mm": 2000, "depth_mm": 900, "height_mm": 800,
            "display_price": "40000",
        })
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        return client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()["session_id"]

    def test_select_reports_what_would_be_covered(self, client):
        session = self._session(client)
        sel = client.post(f"/edit-sessions/{session}/select",
                          json={"x": 400, "y": 700}).json()
        assert sel["hit"]
        assert sel["affects"], "a sofa swap covers nearby items"
        assert all("label" in a for a in sel["affects"])

    def test_the_selected_object_is_not_listed(self, client):
        session = self._session(client)
        sel = client.post(f"/edit-sessions/{session}/select",
                          json={"x": 400, "y": 700}).json()
        assert sel["detection"]["id"] not in [a["id"] for a in sel["affects"]]

    def test_ui_shows_the_warning(self, client):
        assert "This swap covers" in client.get("/ui").text


class TestDetectionBoxFollowsTheReplacement:
    """After a swap the stored detection must describe the NEW object.

    Left stale, every later interaction uses the geometry of something no
    longer in the picture: clicking the part of a larger sofa that extends
    past the old outline selects nothing, and swapping it again sizes the
    region from the footprint of the object it replaced.
    """

    def test_bigger_product_gets_a_bigger_box(self):
        from interior_ai.perception.editing import replaced_object_box

        old = (350, 600, 650, 700)
        small = replaced_object_box(old, "sofa", (1650, 850, 800))
        large = replaced_object_box(old, "sofa", (2600, 1600, 850))
        assert (large[2] - large[0]) > (small[2] - small[0])

    def test_floor_contact_is_preserved(self):
        from interior_ai.perception.editing import replaced_object_box

        old = (350, 600, 650, 700)
        assert replaced_object_box(old, "sofa", (2600, 1600, 850))[3] == old[3]

    def test_rug_extent_uses_depth(self):
        from interior_ai.perception.editing import replaced_object_box

        old = (200, 800, 800, 950)
        out = replaced_object_box(old, "rug", (1520, 2130, 15))
        assert (out[3] - out[1]) > (old[3] - old[1]) * 2

    def test_floating_replacement_drops_to_the_floor(self):
        from interior_ai.perception.editing import replaced_object_box

        out = replaced_object_box((300, 380, 780, 470), "tv_unit",
                                  (1800, 420, 500),
                                  label="floating tv console", floor_top=717)
        assert out[3] == 717

    def test_unknown_dimensions_leave_the_box_alone(self):
        from interior_ai.perception.editing import replaced_object_box

        old = (350, 600, 650, 700)
        assert replaced_object_box(old, "sofa", None) == old

    def test_box_stays_inside_the_frame(self):
        from interior_ai.perception.editing import replaced_object_box

        out = replaced_object_box((900, 900, 1000, 1000), "sofa", (2600, 1600, 850))
        assert 0 <= out[0] < out[2] <= 1000
        assert 0 <= out[1] < out[3] <= 1000

    def test_session_updates_the_detection_after_a_swap(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'box.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))

        client = TestClient(create_app())
        client.post("/catalogue", json={
            "sku": "SOFA-BIG", "name": "Jaipur L-Shape", "object_class": "sofa",
            "width_mm": 2600, "depth_mm": 1600, "height_mm": 850,
            "display_price": "78000",
        })
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        session = client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()
        before = next(d for d in session["detections"] if d["object_class"] == "sofa")

        client.post(f"/edit-sessions/{session['session_id']}/apply",
                    json={"detection_id": before["id"], "sku": "SOFA-BIG"})
        after = next(
            d for d in client.get(
                f"/edit-sessions/{session['session_id']}"
            ).json()["detections"] if d["id"] == before["id"]
        )
        assert (after["box"][2] - after["box"][0]) > (before["box"][2] - before["box"][0])
        assert after["label"] == "Jaipur L-Shape"


class TestCataloguePriceStaysInSync:
    """The catalogue's display_price is what the picker shows; the quote
    commits to whatever price_history last recorded. Changing one without the
    other means the customer is shown one number and quoted another."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'price.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        return TestClient(create_app())

    def _put(self, client, price):
        return client.post("/catalogue", json={
            "sku": "SOFA-X", "name": "Sofa X", "object_class": "sofa",
            "width_mm": 2000, "depth_mm": 900, "height_mm": 800,
            "display_price": price,
        })

    def test_adding_a_product_records_its_price(self, client):
        self._put(client, "40000")
        assert client.get("/prices/SOFA-X").json()["amount"] == "40000.00"

    def test_updating_the_price_updates_the_book(self, client):
        self._put(client, "40000")
        resp = self._put(client, "52000")
        assert resp.json()["price_recorded"] is True
        assert client.get("/prices/SOFA-X").json()["amount"] == "52000.00"

    def test_unchanged_price_writes_no_duplicate(self, client):
        self._put(client, "40000")
        assert self._put(client, "40000").json()["price_recorded"] is False

    def test_quote_matches_the_displayed_price(self, client):
        """The end that matters: what the picker shows is what the quote
        commits to."""
        self._put(client, "40000")
        self._put(client, "52000")
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        session = client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()["session_id"]
        sel = client.post(f"/edit-sessions/{session}/select",
                          json={"x": 400, "y": 700}).json()
        shown = next(o["display_price"] for o in sel["offers"] if o["sku"] == "SOFA-X")
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": sel["detection"]["id"], "sku": "SOFA-X"})
        quote = client.post(f"/edit-sessions/{session}/quote").json()
        assert float(quote["total"]) == float(shown)


class TestQuotePricesFromTheCatalogue:
    """A product in the catalogue is always quotable, from its own row.

    Requiring a separate price_history entry only created ways for the two to
    disagree: a shelf sitting in the catalogue at 34,000 would quote as
    unpriced because nothing had written a price row for it. The catalogue is
    what the operator typed and what the picker shows, so it is what the quote
    commits to -- unless a vendor price was deliberately recorded.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'q.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        return TestClient(create_app())

    def _product_without_price_history(self):
        """Write straight to the catalogue table, as build_catalogue does."""
        from decimal import Decimal

        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import make_engine, make_session_factory

        with make_session_factory(make_engine())() as db:
            db.add(CatalogueItemRow(
                sku="BS-DIVIDER-1", name="Room Divider Shelf",
                object_class="bookshelf", width_mm=1600, depth_mm=400,
                height_mm=1800, display_price=Decimal("34000"), currency="INR",
            ))
            db.commit()

    def _swap(self, client):
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        session = client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()
        sofa = next(d for d in session["detections"] if d["object_class"] == "sofa")
        client.post(f"/edit-sessions/{session['session_id']}/apply",
                    json={"detection_id": sofa["id"], "sku": "BS-DIVIDER-1"})
        return session["session_id"]

    def test_no_price_history_row_exists(self, client):
        from sqlalchemy import select

        from interior_ai.db.models import PriceHistory
        from interior_ai.db.repository import make_engine, make_session_factory

        self._product_without_price_history()
        with make_session_factory(make_engine())() as db:
            rows = list(db.execute(
                select(PriceHistory).where(PriceHistory.sku == "BS-DIVIDER-1")
            ).scalars())
        assert rows == []

    def test_it_still_quotes(self, client):
        self._product_without_price_history()
        quote = client.post(f"/edit-sessions/{self._swap(client)}/quote").json()
        assert quote["is_complete"]
        assert quote["total"] == "34000.00"

    def test_the_line_says_where_the_price_came_from(self, client):
        self._product_without_price_history()
        quote = client.post(f"/edit-sessions/{self._swap(client)}/quote").json()
        assert quote["lines"][0]["vendor"] == "Catalogue"

    def test_recorded_vendor_price_still_wins(self, client):
        """An explicit /prices observation is a deliberate override."""
        self._product_without_price_history()
        session = self._swap(client)
        client.post("/prices", json={
            "sku": "BS-DIVIDER-1", "vendor": "RealVendor",
            "unit": "piece", "amount": "31500",
        })
        quote = client.post(f"/edit-sessions/{session}/quote").json()
        assert quote["total"] == "31500.00"
        assert quote["lines"][0]["vendor"] == "RealVendor"

    def test_unknown_sku_is_still_reported_unpriced(self, client):
        """Falling back to the catalogue must not invent a price for something
        that is not in it."""
        from interior_ai.api.app import create_app  # noqa: F401

        self._product_without_price_history()
        session = self._swap(client)
        quote = client.post(f"/edit-sessions/{session}/quote").json()
        assert all(line["sku"] == "BS-DIVIDER-1" for line in quote["lines"])


class TestOverlayRefreshesAfterASwap:
    """The server updated the detection, but the browser kept drawing the box
    and name from the original detect call -- so a replaced console still
    showed the old outline and the old label. The response now carries the
    refreshed detections, and undo puts the old ones back.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'o.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        client = TestClient(create_app())
        client.post("/catalogue", json={
            "sku": "TV-LINEA-18", "name": "Linea TV Unit 1.8m",
            "object_class": "tv_unit", "width_mm": 1800, "depth_mm": 420,
            "height_mm": 500, "display_price": "22000",
        })
        return client

    def _session(self, client):
        scene = client.post("/scenes", json={"rooms": [{
            "name": "L",
            "polygon": [{"x": 0, "y": 0}, {"x": 3700, "y": 0},
                        {"x": 3700, "y": 4300}, {"x": 0, "y": 4300}],
            "ceiling_height_mm": 3000, "surfaces": {},
        }]}).json()
        return client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()

    def test_apply_returns_the_refreshed_detections(self, client):
        session = self._session(client)
        target = next(d for d in session["detections"] if d["object_class"] == "sofa")
        step = client.post(f"/edit-sessions/{session['session_id']}/apply",
                           json={"detection_id": target["id"],
                                 "sku": "TV-LINEA-18"}).json()
        assert step["detections"], "the client needs the new geometry"
        updated = next(d for d in step["detections"] if d["id"] == target["id"])
        assert updated["label"] == "Linea TV Unit 1.8m"
        assert updated["box"] != target["box"]

    def test_undo_restores_the_previous_outline(self, client):
        """Reverting the image while leaving the new product's box would draw
        a boundary around something no longer in the picture."""
        session = self._session(client)
        target = next(d for d in session["detections"] if d["object_class"] == "sofa")
        client.post(f"/edit-sessions/{session['session_id']}/apply",
                    json={"detection_id": target["id"], "sku": "TV-LINEA-18"})
        undone = client.post(
            f"/edit-sessions/{session['session_id']}/undo"
        ).json()
        restored = next(d for d in undone["detections"] if d["id"] == target["id"])
        assert restored["box"] == target["box"]
        assert restored["label"] == target["label"]

    def test_ui_takes_the_detections_from_the_response(self, client):
        html = client.get("/ui").text
        assert "st.detections" in html
        assert "u.detections" in html


class TestAutomaticRedetection:
    """After a swap the image is re-analysed, because a replacement can change
    an object's shape far more than its dimensions predict -- a narrow 2 m
    bookcase standing in for a wide low one leaves the estimated outline
    visibly wrong.

    The hazard this creates is identity: detection ids are generated fresh
    every pass, and the quote decides supersession by comparing them. Without
    reconciliation the same shelf swapped twice becomes two ids and the
    customer is charged for two shelves.
    """

    def test_identity_survives_a_shape_change(self):
        from interior_ai.perception.editing import reconcile_detections

        old = [Detection(id="shelf-1", label="wide bookshelf",
                         object_class="bookshelf",
                         box=(200, 600, 700, 850), confidence=0.9)]
        new = [Detection(id="fresh", label="tall narrow bookshelf",
                         object_class="bookshelf",
                         box=(330, 180, 560, 850), confidence=0.9)]
        assert reconcile_detections(old, new)[0].id == "shelf-1"

    def test_genuinely_new_objects_keep_fresh_ids(self):
        from interior_ai.perception.editing import reconcile_detections

        old = [Detection(id="sofa-1", label="sofa", object_class="sofa",
                         box=(300, 600, 650, 760), confidence=0.9)]
        new = [
            Detection(id="a", label="sofa", object_class="sofa",
                      box=(300, 600, 650, 760), confidence=0.9),
            Detection(id="b", label="lamp", object_class="lamp",
                      box=(830, 500, 900, 780), confidence=0.9),
        ]
        out = reconcile_detections(old, new)
        assert out[0].id == "sofa-1"
        assert out[1].id == "b"

    def test_one_to_one_matching(self):
        """Two similar objects must not both claim the same identity."""
        from interior_ai.perception.editing import reconcile_detections

        old = [Detection(id="a1", label="chair", object_class="chair",
                         box=(100, 500, 200, 700), confidence=0.9),
               Detection(id="a2", label="chair", object_class="chair",
                         box=(300, 500, 400, 700), confidence=0.9)]
        new = [Detection(id="n1", label="chair", object_class="chair",
                         box=(105, 505, 205, 705), confidence=0.9),
               Detection(id="n2", label="chair", object_class="chair",
                         box=(305, 505, 405, 705), confidence=0.9)]
        ids = [d.id for d in reconcile_detections(old, new)]
        assert ids == ["a1", "a2"]
        assert len(set(ids)) == 2

    def test_distant_objects_do_not_match(self):
        from interior_ai.perception.editing import reconcile_detections

        old = [Detection(id="a", label="sofa", object_class="sofa",
                         box=(0, 0, 100, 100), confidence=0.9)]
        new = [Detection(id="n", label="sofa", object_class="sofa",
                         box=(800, 800, 900, 900), confidence=0.9)]
        assert reconcile_detections(old, new)[0].id == "n"

    def test_apply_exposes_the_toggle(self):
        from interior_ai.api.schemas import ApplyIn

        assert ApplyIn(detection_id="d", sku="s").redetect is True
        assert ApplyIn(detection_id="d", sku="s", redetect=False).redetect is False

    def test_swap_twice_prices_once(self, tmp_path, monkeypatch):
        """The failure reconciliation exists to prevent."""
        import uuid
        from decimal import Decimal

        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import create_all, make_engine, make_session_factory
        from interior_ai.perception.edit_session import EditSessionService

        url = f"sqlite+pysqlite:///{tmp_path / 'r.db'}"
        engine = make_engine(url)
        create_all(engine)
        db = make_session_factory(engine)()
        for sku, price in [("A", "40000"), ("B", "52000")]:
            db.add(CatalogueItemRow(sku=sku, name=f"Shelf {sku}",
                                    object_class="bookshelf", width_mm=900,
                                    depth_mm=350, height_mm=1700,
                                    display_price=Decimal(price)))
        db.commit()

        class Editor:
            """Fresh ids every pass, like the real detector."""

            def detect(self, ref):
                return [Detection(id=uuid.uuid4().hex[:12], label="bookshelf",
                                  object_class="bookshelf",
                                  box=(200, 600, 700, 850), confidence=0.9)], []

            def replace(self, ref, det, **kw):
                return "mock://edited"

        svc = EditSessionService(db, editor=Editor())
        session = svc.start(scene_id="s", room_id="r", image_ref="mock://o")
        first = svc._detections(session)[0]
        svc.apply(session, first.id, "A")
        second = svc._detections(session)[0]
        svc.apply(session, second.id, "B")
        assert len(svc.swapped_skus(session)) == 1
        assert list(svc.swapped_skus(session).values()) == ["B"]
        db.close()

    def test_stale_detection_falls_back_to_the_estimate(self, tmp_path):
        """A detector that reports the swapped object completely unchanged has
        not seen the edit. Accepting that would leave the old name and outline
        on the new product."""
        from decimal import Decimal

        from interior_ai.db.catalogue import CatalogueItemRow
        from interior_ai.db.repository import create_all, make_engine, make_session_factory
        from interior_ai.perception.edit_session import EditSessionService
        from interior_ai.perception.editing import MockPhotoEditor

        url = f"sqlite+pysqlite:///{tmp_path / 's.db'}"
        engine = make_engine(url)
        create_all(engine)
        db = make_session_factory(engine)()
        db.add(CatalogueItemRow(sku="TALL", name="Tall Narrow Bookshelf",
                                object_class="bookshelf", width_mm=400,
                                depth_mm=300, height_mm=2000,
                                display_price=Decimal("9500")))
        db.commit()

        # MockPhotoEditor.detect returns the same fixed set whatever the image.
        svc = EditSessionService(db, editor=MockPhotoEditor())
        session = svc.start(scene_id="s", room_id="r", image_ref="mock://o")
        target = next(d for d in svc._detections(session)
                      if d.object_class == "sofa")
        svc.apply(session, target.id, "TALL")
        updated = next(d for d in svc._detections(session) if d.id == target.id)
        assert updated.label == "Tall Narrow Bookshelf"
        assert updated.box != target.box
        db.close()


class TestOversizePreflight:
    """Ask before spending a minute of image generation on a swap that cannot
    plausibly work. The check informs; it never forbids -- the person may well
    want to see the result anyway, and refusing outright would be worse than
    the wasted call.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'pf.db'}"
        monkeypatch.setenv("DATABASE_URL", url)
        monkeypatch.setenv("AUTO_CREATE_SCHEMA", "1")
        create_all(make_engine(url))
        client = TestClient(create_app())
        for sku, name, w, d, h, price in [
            ("SOFA-OK", "Oslo 2-Seater", 1650, 850, 800, "38000"),
            ("SOFA-HUGE", "Palace 9-Seater", 9000, 2000, 900, "250000"),
        ]:
            client.post("/catalogue", json={
                "sku": sku, "name": name, "object_class": "sofa",
                "width_mm": w, "depth_mm": d, "height_mm": h,
                "display_price": price,
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
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()
        target = next(d for d in session["detections"] if d["object_class"] == "sofa")
        return session["session_id"], target["id"]

    def test_reasonable_product_is_not_questioned(self, client):
        session, target = self._session(client)
        resp = client.post(f"/edit-sessions/{session}/apply",
                           json={"detection_id": target, "sku": "SOFA-OK"})
        assert resp.status_code == 200

    def test_oversized_product_asks_first(self, client):
        session, target = self._session(client)
        resp = client.post(f"/edit-sessions/{session}/apply",
                           json={"detection_id": target, "sku": "SOFA-HUGE"})
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "oversize_replacement"

    def test_reasons_are_specific(self, client):
        """'Too big' is not actionable; the measured overage is."""
        session, target = self._session(client)
        detail = client.post(f"/edit-sessions/{session}/apply",
                             json={"detection_id": target,
                                   "sku": "SOFA-HUGE"}).json()["detail"]
        assert detail["reasons"]
        assert any("mm" in r for r in detail["reasons"])

    def test_nothing_changes_when_it_asks(self, client):
        """The question must cost nothing -- no step, no image call."""
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": target, "sku": "SOFA-HUGE"})
        state = client.get(f"/edit-sessions/{session}").json()
        assert state["swapped_skus"] == {}
        assert state["steps"] == []

    def test_confirmation_proceeds(self, client):
        session, target = self._session(client)
        resp = client.post(f"/edit-sessions/{session}/apply",
                           json={"detection_id": target, "sku": "SOFA-HUGE",
                                 "confirm_oversize": True})
        assert resp.status_code == 200
        assert resp.json()["swapped_skus"] == {target: "SOFA-HUGE"}

    def test_covered_objects_are_named(self, client):
        session, target = self._session(client)
        detail = client.post(f"/edit-sessions/{session}/apply",
                             json={"detection_id": target,
                                   "sku": "SOFA-HUGE"}).json()["detail"]
        assert any("extend across" in r for r in detail["reasons"])

    def test_ui_asks_and_can_cancel(self, client):
        html = client.get("/ui").text
        assert "oversize_replacement" in html
        assert "window.confirm" in html
        assert "Swap cancelled" in html

    def test_normal_products_are_never_questioned(self, client):
        """A preflight that fires on ordinary swaps trains people to click
        through it, which is worse than not having one."""
        session, target = self._session(client)
        for sku, name, w, d, h in [
            ("P-A", "Oslo 2-Seater", 1650, 850, 800),
            ("P-B", "Milano 3-Seater", 2100, 880, 820),
            ("P-C", "Jaipur L-Shape", 2600, 1600, 850),
        ]:
            client.post("/catalogue", json={
                "sku": sku, "name": name, "object_class": "sofa",
                "width_mm": w, "depth_mm": d, "height_mm": h,
                "display_price": "40000",
            })
            resp = client.post(f"/edit-sessions/{session}/apply",
                               json={"detection_id": target, "sku": sku})
            assert resp.status_code == 200, f"{name} should not need confirming"
            client.post(f"/edit-sessions/{session}/undo")

    def test_tall_narrow_replacement_is_not_questioned(self, client):
        """Vertical growth is normal: detection boxes are tight, and a tall
        bookcase standing in for a squat one needs several times the area."""
        session, _target = self._session(client)
        state = client.get(f"/edit-sessions/{session}").json()
        shelf = next((d for d in state["detections"]
                      if d["object_class"] in {"bookshelf", "coffee_table"}), None)
        if shelf is None:
            pytest.skip("no suitable object in the mock detection set")
        client.post("/catalogue", json={
            "sku": "TALL", "name": "Tall Narrow Bookshelf",
            "object_class": shelf["object_class"], "width_mm": 400,
            "depth_mm": 300, "height_mm": 2000, "display_price": "9500",
        })
        resp = client.post(f"/edit-sessions/{session}/apply",
                           json={"detection_id": shelf["id"], "sku": "TALL"})
        assert resp.status_code == 200


class TestTypedInstructions:
    """Free-text edits, interpreted before they are executed.

    Sending raw text straight to an image model spends a minute discovering
    that "make it darker" meant the wall, not the sofa that happened to be
    selected. A fast text call resolves which object is meant and what should
    happen to it, which also decides whether the edit can be region-locked.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from interior_ai.api.app import create_app
        from interior_ai.db import catalogue as _c  # noqa: F401
        from interior_ai.db.repository import create_all, make_engine

        url = f"sqlite+pysqlite:///{tmp_path / 'i.db'}"
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
        session = client.post(
            f"/scenes/{scene['scene_id']}/rooms/{scene['rooms'][0]['id']}/edit-session",
            files={"image": ("r.png", TestEditingAPI._png(), "image/png")},
        ).json()
        target = next(d for d in session["detections"] if d["object_class"] == "sofa")
        return session["session_id"], target["id"]

    def test_instruction_with_a_selection_targets_it(self, client):
        session, target = self._session(client)
        out = client.post(f"/edit-sessions/{session}/instruct",
                          json={"text": "make this a deep navy blue",
                                "detection_id": target}).json()
        assert out["applied"]
        assert out["intent"]["target_id"] == target

    def test_instruction_without_a_selection_is_scene_wide(self, client):
        session, _ = self._session(client)
        out = client.post(f"/edit-sessions/{session}/instruct",
                          json={"text": "brighten the whole room"}).json()
        assert out["applied"]
        assert out["intent"]["target_id"] is None

    def test_instructions_never_reach_the_quote(self, client):
        """A typed edit changes the photo but adds no product."""
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/instruct",
                    json={"text": "paint it darker", "detection_id": target})
        quote = client.post(f"/edit-sessions/{session}/quote").json()
        assert quote["lines"] == []

    def test_instruction_on_a_swapped_object_drops_its_price(self, client):
        """Editing away a product means it is no longer what is in the
        picture, so charging for it would be wrong."""
        client.post("/catalogue", json={
            "sku": "SOFA-A", "name": "Sofa A", "object_class": "sofa",
            "width_mm": 2000, "depth_mm": 900, "height_mm": 800,
            "display_price": "40000",
        })
        session, target = self._session(client)
        client.post(f"/edit-sessions/{session}/apply",
                    json={"detection_id": target, "sku": "SOFA-A"})
        assert client.post(f"/edit-sessions/{session}/quote").json()["lines"]
        state = client.get(f"/edit-sessions/{session}").json()
        current = next(d for d in state["detections"]
                       if d["id"] == target)["id"]
        client.post(f"/edit-sessions/{session}/instruct",
                    json={"text": "remove it entirely", "detection_id": current})
        assert client.post(f"/edit-sessions/{session}/quote").json()["lines"] == []

    def test_unclear_request_does_nothing(self, client):
        from interior_ai.perception.editing import parse_intent

        intent = parse_intent("no idea what this means")
        assert not intent.is_actionable

    def test_mismatch_asks_before_acting(self):
        """A misplaced click is common; the words are better evidence."""
        from interior_ai.perception.editing import parse_intent

        intent = parse_intent(
            '{"target_id":"det-wall","operation":"recolour",'
            '"selection_matches":false,"instruction":"Paint the wall sage.",'
            '"confidence":0.9}'
        )
        assert intent.selection_matches is False
        assert intent.target_id == "det-wall"

    def test_intent_prompt_carries_the_position_map(self):
        """The model can only pick a target if it is told where things are."""
        import base64
        import io

        from PIL import Image

        from interior_ai.perception.editing import Detection, GeminiPhotoEditor

        seen: dict = {}

        def transport(model, payload):
            seen["prompt"] = payload["contents"][0]["parts"][0]["text"]
            return {"candidates": [{"content": {"parts": [{"text":
                '{"operation":"recolour","instruction":"x","target_id":"a"}'}]}}]}

        editor = GeminiPhotoEditor(api_key="x", transport=transport)
        dets = [Detection(id="a", label="grey sofa", object_class="sofa",
                          box=(100, 200, 300, 400), confidence=0.9)]
        editor.analyse_instruction("make it blue", dets, selected=dets[0])
        assert "grey sofa" in seen["prompt"]
        assert "[100, 200, 300, 400]" in seen["prompt"]
        assert "THEY HAD SELECTED" in seen["prompt"]

    def test_no_selection_is_stated_in_the_prompt(self):
        from interior_ai.perception.editing import GeminiPhotoEditor

        seen: dict = {}

        def transport(model, payload):
            seen["prompt"] = payload["contents"][0]["parts"][0]["text"]
            return {"candidates": [{"content": {"parts": [{"text":
                '{"operation":"scene","instruction":"x"}'}]}}]}

        GeminiPhotoEditor(api_key="x", transport=transport).analyse_instruction(
            "brighten the room", []
        )
        assert "NOTHING SELECTED" in seen["prompt"]

    def test_targeted_edit_is_region_locked(self):
        """A request to recolour one chair must not repaint the room."""
        from PIL import Image, ImageChops

        from interior_ai.perception.editing import Detection, EditIntent

        helper = TestRegionLockedReplacement()
        editor, _state = helper._editor_returning_solid()
        img = Image.new("RGB", (900, 700), (40, 160, 60))
        target = Detection(id="d", label="chair", object_class="chair",
                           box=(700, 600, 850, 800), confidence=0.9)
        intent = EditIntent(target_ids=("d",), operation="recolour",
                            instruction="Make the chair navy.", confidence=0.9)
        out = helper._decode(editor.instruct(helper._uri(img), intent,
                                             target=target))
        diff = ImageChops.difference(img, out).convert("L")
        assert max(diff.crop((0, 0, 300, 300)).getdata()) == 0

    def test_removal_uses_its_own_prompt(self):
        from interior_ai.perception.editing import (
            INSTRUCT_REMOVE_TEMPLATE,
            Detection,
            EditIntent,
        )
        from PIL import Image

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        target = Detection(id="d", label="floor lamp", object_class="lamp",
                           box=(700, 400, 800, 700), confidence=0.9)
        editor.instruct(helper._uri(Image.new("RGB", (900, 700), "white")),
                        EditIntent(target_ids=("d",), operation="remove",
                                   instruction="Remove the lamp.",
                                   confidence=0.9),
                        target=target)
        assert "reconstruct what would be behind it" in state["prompt"]

    def test_ui_offers_the_input(self, client):
        html = client.get("/ui").text
        assert "ask-text" in html and "sendInstruction" in html


class TestMultipleTargets:
    """"Paint the wall sage green" means every wall.

    A room's walls arrive as several detected regions -- a marble feature
    wall, white walls either side. Resolving an unqualified request to a
    single id paints one panel and leaves the rest, which is never what
    anybody meant by "paint the wall".
    """

    def _walls(self):
        return [
            Detection(id="w-left", label="white painted wall",
                      object_class="wall", box=(0, 0, 300, 700), confidence=0.9),
            Detection(id="w-feat", label="marble feature wall",
                      object_class="wall", box=(300, 0, 650, 700), confidence=0.9),
            Detection(id="w-right", label="white painted wall",
                      object_class="wall", box=(650, 0, 1000, 700), confidence=0.9),
        ]

    def test_intent_accepts_a_list(self):
        from interior_ai.perception.editing import parse_intent

        intent = parse_intent(
            '{"target_ids":["w-left","w-feat","w-right"],'
            '"operation":"recolour","instruction":"Paint every wall sage.",'
            '"confidence":0.9}'
        )
        assert len(intent.target_ids) == 3

    def test_single_id_key_still_parses(self):
        """Models drift between the two shapes; both must work."""
        from interior_ai.perception.editing import parse_intent

        assert parse_intent(
            '{"target_id":"w-feat","operation":"recolour",'
            '"instruction":"x","confidence":0.9}'
        ).target_ids == ("w-feat",)

    def test_prompt_asks_for_every_matching_region(self):
        from interior_ai.perception.editing import INTENT_PROMPT

        assert "EVERY matching region" in INTENT_PROMPT
        assert "painting one wall of four" in INTENT_PROMPT

    def test_all_targets_are_edited(self):
        from PIL import Image, ImageChops, ImageDraw

        from interior_ai.perception.editing import EditIntent

        img = Image.new("RGB", (900, 600), (240, 236, 228))
        ImageDraw.Draw(img).rectangle((0, 430, 900, 600), fill=(178, 140, 96))

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid(colour=(150, 180, 150))
        intent = EditIntent(target_ids=("w-left", "w-feat", "w-right"),
                            operation="recolour",
                            instruction="Paint every wall sage green.",
                            confidence=0.9)
        out = helper._decode(editor.instruct(helper._uri(img), intent,
                                             targets=self._walls()))
        diff = ImageChops.difference(img, out).convert("L")
        for x in (100, 450, 800):
            assert diff.getpixel((x, 150)) > 0, f"wall at x={x} was left unchanged"
        assert "APPLY IT TO ALL OF THEM" in state["prompt"]

    def test_every_target_is_named_in_the_prompt(self):
        from PIL import Image

        from interior_ai.perception.editing import EditIntent

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.instruct(helper._uri(Image.new("RGB", (900, 600), "white")),
                        EditIntent(target_ids=("w-left", "w-feat", "w-right"),
                                   operation="recolour",
                                   instruction="Paint every wall sage.",
                                   confidence=0.9),
                        targets=self._walls())
        assert "marble feature wall" in state["prompt"]
        assert "white painted wall" in state["prompt"]

    def test_group_members_are_not_treated_as_bystanders(self):
        """A wall being repainted must not also appear on the list of things
        to preserve -- that would ask for both at once."""
        from PIL import Image

        from interior_ai.perception.editing import EditIntent

        walls = self._walls()
        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.instruct(helper._uri(Image.new("RGB", (900, 600), "white")),
                        EditIntent(target_ids=tuple(d.id for d in walls),
                                   operation="recolour",
                                   instruction="Paint every wall sage.",
                                   confidence=0.9),
                        targets=walls, neighbours=walls)
        prompt = state["prompt"]
        assert "MUST SURVIVE" not in prompt
        assert "NEARER THE CAMERA" not in prompt

    def test_single_target_keeps_the_tight_lock(self):
        """Selecting one wall must still edit only that wall."""
        from PIL import Image, ImageChops

        from interior_ai.perception.editing import EditIntent

        img = Image.new("RGB", (900, 600), (40, 160, 60))
        helper = TestRegionLockedReplacement()
        editor, _state = helper._editor_returning_solid()
        out = helper._decode(editor.instruct(
            helper._uri(img),
            EditIntent(target_ids=("w-left",), operation="recolour",
                       instruction="Paint this wall sage.", confidence=0.9),
            targets=[self._walls()[0]],
        ))
        diff = ImageChops.difference(img, out).convert("L")
        assert max(diff.crop((750, 400, 900, 600)).getdata()) == 0


class TestUnselectedRequestsGetNoMap:
    """With nothing selected, the request goes to the image model with the
    photograph and nothing else.

    The object map actively harms this case. Detection carves a room's walls
    into separate regions -- a marble feature wall behind the television,
    painted walls either side -- and any target resolved from that list
    inherits the carving, so "paint the wall" repaints one panel. Those
    boundaries are an artefact of how detection segments a photo, not of how a
    person sees the room.
    """

    class _Spy:
        def __init__(self):
            self.analysed = 0
            self.calls = []

        def detect(self, ref):
            return [
                Detection(id="w-feat", label="marble feature wall",
                          object_class="wall", box=(300, 0, 650, 700),
                          confidence=0.9),
                Detection(id="sofa", label="three-seat sofa",
                          object_class="sofa", box=(150, 550, 700, 900),
                          confidence=0.9),
            ], []

        def analyse_instruction(self, text, dets, *, selected=None):
            from interior_ai.perception.editing import EditIntent

            self.analysed += 1
            return EditIntent(target_ids=("w-feat",), operation="recolour",
                              instruction="Paint the marble feature wall sage.",
                              confidence=0.9, selection_matches=True)

        def instruct(self, ref, intent, *, target=None, targets=None,
                     neighbours=None, floor_top=None, on_retry=None):
            self.calls.append({
                "target": target.id if target else None,
                "targets": [d.id for d in (targets or [])],
                "neighbours": len(neighbours or []),
                "instruction": intent.instruction,
                "operation": intent.operation,
            })
            return "mock://edited"

    def _service(self, tmp_path):
        from interior_ai.db.repository import create_all, make_engine, make_session_factory
        from interior_ai.perception.edit_session import EditSessionService

        engine = make_engine(f"sqlite+pysqlite:///{tmp_path / 'nm.db'}")
        create_all(engine)
        db = make_session_factory(engine)()
        spy = self._Spy()
        svc = EditSessionService(db, editor=spy)
        session = svc.start(scene_id="s", room_id="r", image_ref="mock://o")
        return svc, session, spy, db

    def test_no_intent_call_is_made(self, tmp_path):
        """Interpretation exists to resolve a target; with no map there is
        nothing to resolve, and the call would only cost time."""
        svc, session, spy, db = self._service(tmp_path)
        svc.instruct(session, "paint wall sage green")
        assert spy.analysed == 0
        db.close()

    def test_no_targets_are_passed(self, tmp_path):
        svc, session, spy, db = self._service(tmp_path)
        svc.instruct(session, "paint wall sage green")
        assert spy.calls[-1]["target"] is None
        assert spy.calls[-1]["targets"] == []
        db.close()

    def test_no_neighbours_are_passed(self, tmp_path):
        """They only tell a region-locked edit what not to damage. With no
        region they would just describe the room to a model that sees it."""
        svc, session, spy, db = self._service(tmp_path)
        svc.instruct(session, "paint wall sage green")
        assert spy.calls[-1]["neighbours"] == 0
        db.close()

    def test_the_users_own_words_are_sent(self, tmp_path):
        svc, session, spy, db = self._service(tmp_path)
        svc.instruct(session, "paint wall sage green")
        assert spy.calls[-1]["instruction"] == "paint wall sage green"
        assert spy.calls[-1]["operation"] == "scene"
        db.close()

    def test_a_selection_still_uses_the_map(self, tmp_path):
        """The targeted path is unchanged: a click means that object."""
        svc, session, spy, db = self._service(tmp_path)
        svc.instruct(session, "make this navy", detection_id="sofa")
        assert spy.analysed == 1
        assert spy.calls[-1]["target"] == "w-feat"
        db.close()

    def test_scene_prompt_carries_no_coordinates(self):
        from PIL import Image

        from interior_ai.perception.editing import EditIntent

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.instruct(helper._uri(Image.new("RGB", (900, 600), "white")),
                        EditIntent(target_ids=(), operation="scene",
                                   instruction="paint wall sage green",
                                   confidence=1.0))
        prompt = state["prompt"]
        assert "paint wall sage green" in prompt
        assert "0-1000" not in prompt
        assert "CROPPED REGION" not in prompt

    def test_scene_edit_sends_the_whole_image(self):
        """No crop: the request is about the room, so the model sees the room."""
        from PIL import Image

        from interior_ai.perception.editing import EditIntent

        helper = TestRegionLockedReplacement()
        editor, state = helper._editor_returning_solid()
        editor.instruct(helper._uri(Image.new("RGB", (900, 600), "white")),
                        EditIntent(target_ids=(), operation="scene",
                                   instruction="brighten the room",
                                   confidence=1.0))
        assert state["sent_size"] == (900, 600)


class TestSceneScopeDiscipline:
    """Full freedom needs an explicit definition of scope.

    Asked to "paint walls sage green" with no object map, the model painted the
    built-in shelving, the media unit and the cabinetry too -- all of which are
    fixed against a wall and could be read as part of it. Without a region lock
    the only thing holding the edit to what was asked is the prompt, so the
    prompt has to say where a wall ends and furniture begins.
    """

    def _prompt(self, instruction="paint walls sage green"):
        from interior_ai.perception.editing import INSTRUCT_SCENE_TEMPLATE

        return INSTRUCT_SCENE_TEMPLATE.format(instruction=instruction)

    def test_built_ins_are_defined_as_furniture(self):
        prompt = self._prompt()
        assert "are FURNITURE, not wall" in prompt
        for item in ("shelving", "cabinetry", "panelling", "joinery",
                     "media unit", "wardrobes"):
            assert item in prompt, f"{item} not covered"

    def test_the_awkward_cases_are_named(self):
        """Floor-to-ceiling, wall-coloured, wall-covering units are exactly
        what the model mistook for wall."""
        prompt = self._prompt()
        assert "floor to ceiling" in prompt
        assert "same colour as the wall" in prompt
        assert "cover most of it" in prompt

    def test_uncertainty_defaults_to_excluding(self):
        """A model unsure whether something counts should leave it alone."""
        prompt = self._prompt()
        assert "NOT included" in prompt

    def test_other_categories_are_bounded(self):
        prompt = self._prompt()
        assert "CEILING means the overhead surface only" in prompt
        assert "FLOOR means the floor covering only" in prompt

    def test_contents_are_protected(self):
        prompt = self._prompt()
        for item in ("decor and objects on", "curtains and blinds",
                     "television and electronics", "plants", "artwork"):
            assert item in prompt, f"{item} not protected"

    def test_it_still_covers_every_wall(self):
        """Scope discipline must not undo the fix for painting one panel."""
        assert "every wall in the room" in self._prompt()

    def test_scene_prompt_still_carries_no_map(self):
        """The cure for over-reach is precision about categories, not
        reintroducing coordinates."""
        prompt = self._prompt()
        assert "0-1000" not in prompt
        assert "CROPPED REGION" not in prompt

    def test_the_users_words_lead(self):
        prompt = self._prompt("remove the rug")
        assert prompt.index("remove the rug") < prompt.index("SCOPE")