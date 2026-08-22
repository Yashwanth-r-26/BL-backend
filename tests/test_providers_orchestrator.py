"""Providers and the end-to-end orchestrator."""

from __future__ import annotations

import pytest

from interior_ai.core.enums import ExecutionPath, ObjectClass, Phase, Tri
from interior_ai.core.scene import CatalogueItem, Footprint, Scene, SurfaceState, Vec2
from interior_ai.orchestrator import Orchestrator
from interior_ai.providers.base import ProviderError, RenderRequest
from interior_ai.providers.gemini import (
    GeminiPerceptionProvider,
    GeminiRenderProvider,
    parse_surface_response,
)
from interior_ai.providers.mock import MockPerceptionProvider, MockRenderProvider

COMPLETE_JSON = (
    '{"walls_painted":"yes","flooring_installed":"yes","ceiling_finished":"yes",'
    '"electrical_terminated":"yes","plumbing_terminated":"yes",'
    '"carpentry_installed":"yes","furniture_present":"no"}'
)


def gemini_returning(text: str) -> GeminiPerceptionProvider:
    def transport(payload):
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    return GeminiPerceptionProvider(api_key="test", transport=transport)


class TestMockProvider:
    def test_is_deterministic(self):
        provider = MockPerceptionProvider()
        assert provider.analyse("room.jpg").surfaces == provider.analyse("room.jpg").surfaces

    def test_different_inputs_differ(self):
        provider = MockPerceptionProvider()
        a = provider.analyse("kitchen.jpg").surfaces
        b = provider.analyse("bathroom.jpg").surfaces
        assert a != b

    def test_declares_itself_a_mock(self):
        result = MockPerceptionProvider().analyse("room.jpg")
        assert result.path is ExecutionPath.MOCK
        assert any("MOCK" in n for n in result.notes)

    def test_forced_state_overrides_hashing(self, finished_surfaces):
        provider = MockPerceptionProvider(forced=finished_surfaces)
        assert provider.analyse("anything.jpg").surfaces == finished_surfaces

    def test_render_is_stable(self):
        req = RenderRequest(prompt="a living room", room_id="r1")
        assert MockRenderProvider().render(req).image_ref == MockRenderProvider().render(req).image_ref


class TestGeminiParsing:
    def test_plain_json(self):
        state, notes = parse_surface_response(COMPLETE_JSON)
        assert state.walls_painted == "yes"
        assert notes == []

    def test_fenced_json(self):
        state, _ = parse_surface_response(f"```json\n{COMPLETE_JSON}\n```")
        assert state.flooring_installed == "yes"

    def test_json_with_surrounding_prose(self):
        state, _ = parse_surface_response(f"Here is my assessment:\n{COMPLETE_JSON}\nHope that helps.")
        assert state.ceiling_finished == "yes"

    def test_partial_is_preserved(self):
        text = COMPLETE_JSON.replace('"walls_painted":"yes"', '"walls_painted":"partial"')
        state, _ = parse_surface_response(text)
        assert state.walls_painted == Tri.PARTIAL.value

    def test_hedge_becomes_unknown_not_yes(self):
        """'mostly painted' silently upgraded to YES is how a half-painted room
        gets quoted as finished."""
        text = COMPLETE_JSON.replace('"walls_painted":"yes"', '"walls_painted":"mostly"')
        state, notes = parse_surface_response(text)
        assert state.walls_painted == Tri.UNKNOWN.value
        assert any("not a recognised value" in n for n in notes)

    def test_prose_response_degrades_to_all_unknown(self):
        state, notes = parse_surface_response("The room looks pretty good honestly")
        assert state.walls_painted == Tri.UNKNOWN.value
        assert notes

    def test_missing_keys_become_unknown(self):
        state, notes = parse_surface_response('{"walls_painted":"yes"}')
        assert state.flooring_installed == Tri.UNKNOWN.value
        assert any("omitted" in n for n in notes)

    def test_unexpected_keys_are_noted_not_fatal(self):
        text = COMPLETE_JSON[:-1] + ',"vibe":"cosy"}'
        state, notes = parse_surface_response(text)
        assert state.walls_painted == "yes"
        assert any("unexpected keys" in n for n in notes)

    def test_booleans_coerce(self):
        text = COMPLETE_JSON.replace('"walls_painted":"yes"', '"walls_painted":true')
        state, _ = parse_surface_response(text)
        assert state.walls_painted == Tri.YES.value


class TestGeminiProvider:
    def test_analyse_via_injected_transport(self):
        result = gemini_returning(COMPLETE_JSON).analyse("data:image/jpeg;base64,AAAA")
        assert result.path is ExecutionPath.CLOUD_API
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_drops_with_unknowns(self):
        partial = '{"walls_painted":"yes"}'
        assert gemini_returning(partial).analyse("x.jpg").confidence < 0.5

    def test_malformed_response_shape_raises(self):
        provider = GeminiPerceptionProvider(api_key="test", transport=lambda p: {"nope": 1})
        with pytest.raises(ProviderError):
            provider.analyse("x.jpg")

    def test_missing_key_raises_rather_than_silently_mocking(self, clean_env):
        provider = GeminiPerceptionProvider(api_key="")
        with pytest.raises(ProviderError):
            provider.analyse("x.jpg")

    def test_render_extracts_inline_image(self):
        def transport(payload):
            return {
                "candidates": [
                    {"content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": "QUJD"}}]}}
                ]
            }

        provider = GeminiRenderProvider(api_key="test", transport=transport)
        result = provider.render(RenderRequest(prompt="a room", room_id="r1"))
        assert result.image_ref.startswith("data:image/png;base64,")

    def test_render_without_image_raises(self):
        transport = lambda p: {"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]}
        provider = GeminiRenderProvider(api_key="test", transport=transport)
        with pytest.raises(ProviderError):
            provider.render(RenderRequest(prompt="a room", room_id="r1"))


SOFA = CatalogueItem(
    sku="SOFA-3S", name="Sofa", object_class=ObjectClass.SOFA,
    footprint=Footprint(width_mm=2200, depth_mm=900, height_mm=800),
)
TABLE = CatalogueItem(
    sku="CT-01", name="Table", object_class=ObjectClass.COFFEE_TABLE,
    footprint=Footprint(width_mm=1100, depth_mm=600, height_mm=400),
)


class TestOrchestratorPhaseGate:
    def test_unfinished_room_is_blocked(self, scene, mock_probe, price_book):
        unfinished = SurfaceState(
            walls_painted="no", flooring_installed="no", ceiling_finished="no",
            electrical_terminated="no", plumbing_terminated="no",
            carpentry_installed="no", furniture_present="no",
        )
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=unfinished),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=(SOFA,))
        assert not report.ok
        assert report.blocked_reason
        assert report.phase.phase is Phase.SURFACE_FINISHING

    def test_blocked_room_still_gets_a_surface_quote(self, scene, mock_probe, price_book):
        """The room is not ready for furniture, but the work it *does* need is
        exactly what should be priced."""
        unfinished = SurfaceState(
            walls_painted="no", flooring_installed="no", ceiling_finished="no",
            electrical_terminated="no", plumbing_terminated="no",
            carpentry_installed="no", furniture_present="no",
        )
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=unfinished),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=(SOFA,))
        assert report.quote is not None
        assert report.quote.total > 0

    def test_partial_work_blocks_the_pipeline(self, scene, mock_probe, price_book):
        half_painted = SurfaceState(
            walls_painted="partial", flooring_installed="yes", ceiling_finished="yes",
            electrical_terminated="yes", plumbing_terminated="yes",
            carpentry_installed="yes", furniture_present="no",
        )
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=half_painted),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=(SOFA,))
        assert not report.ok
        assert "walls_painted" in report.blocked_reason


@pytest.mark.slow
class TestOrchestratorFullRun:
    def test_finished_room_completes(self, scene, mock_probe, price_book, finished_surfaces):
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=finished_surfaces),
            price_book=price_book,
        )
        report = orch.run(
            scene, scene.rooms[0].id, catalogue=(SOFA, TABLE),
            focal_point=Vec2(x=2500, y=0), solve_time_limit_s=20,
        )
        assert report.ok, report.blocked_reason
        assert report.validation.ok
        assert report.new_scene is not None

    def test_commits_a_new_immutable_version(self, scene, mock_probe, price_book, finished_surfaces):
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=finished_surfaces),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=(SOFA,), solve_time_limit_s=20)
        assert report.new_scene.version == scene.version + 1
        assert report.new_scene.parent_version_id == scene.version_id
        # The input scene is untouched.
        assert scene.rooms[0].placements == ()

    def test_stored_surfaces_are_not_overwritten_by_perception(
        self, bare_room, mock_probe, price_book, finished_surfaces
    ):
        """Regression: the orchestrator re-perceived every run, silently
        overwriting an assessment a human may have corrected."""
        room = bare_room.model_copy(update={"surfaces": finished_surfaces})
        scene = Scene(rooms=(room,))
        # A provider that would report the room as unfinished if consulted.
        contradicting = MockPerceptionProvider(
            forced=SurfaceState(
                walls_painted="no", flooring_installed="no", ceiling_finished="no",
                electrical_terminated="no", plumbing_terminated="no",
                carpentry_installed="no", furniture_present="no",
            )
        )
        orch = Orchestrator(probe=mock_probe, perception=contradicting, price_book=price_book)
        report = orch.run(scene, room.id, catalogue=(SOFA,), solve_time_limit_s=20)
        assert report.phase.phase is Phase.STYLING_RESTRUCTURE
        assert any("already has known surfaces" in s for s in report.stages)

    def test_reperceive_flag_forces_a_fresh_look(
        self, bare_room, mock_probe, price_book, finished_surfaces
    ):
        room = bare_room.model_copy(update={"surfaces": finished_surfaces})
        scene = Scene(rooms=(room,))
        contradicting = MockPerceptionProvider(
            forced=SurfaceState(
                walls_painted="no", flooring_installed="no", ceiling_finished="no",
                electrical_terminated="no", plumbing_terminated="no",
                carpentry_installed="no", furniture_present="no",
            )
        )
        orch = Orchestrator(probe=mock_probe, perception=contradicting, price_book=price_book)
        report = orch.run(scene, room.id, catalogue=(SOFA,), reperceive=True)
        assert report.phase.phase is Phase.SURFACE_FINISHING

    def test_no_catalogue_is_reported_not_crashed(
        self, scene, mock_probe, price_book, finished_surfaces
    ):
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=finished_surfaces),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=())
        assert not report.ok
        assert "catalogue" in report.blocked_reason

    def test_every_stage_is_recorded(self, scene, mock_probe, price_book, finished_surfaces):
        orch = Orchestrator(
            probe=mock_probe,
            perception=MockPerceptionProvider(forced=finished_surfaces),
            price_book=price_book,
        )
        report = orch.run(scene, scene.rooms[0].id, catalogue=(SOFA,), solve_time_limit_s=20)
        joined = " ".join(report.stages)
        assert "probe" in joined
        assert "phase" in joined
        assert "solver" in joined
        assert "validation" in joined

    def test_unknown_room_raises(self, scene, mock_probe, price_book):
        orch = Orchestrator(probe=mock_probe, price_book=price_book)
        with pytest.raises(KeyError):
            orch.run(scene, "no-such-room")
