"""Phase rules and capability probing."""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from interior_ai.core.enums import ExecutionPath, Phase, Tri
from interior_ai.core.scene import SurfaceState
from interior_ai.perception.probe import CapabilityProbe, GpuInfo
from interior_ai.phase.rules import (
    REVIEW_THRESHOLD,
    can_progress,
    classify,
    restart_from_empty,
)

DONE = dict(
    walls_painted="yes",
    flooring_installed="yes",
    ceiling_finished="yes",
    electrical_terminated="yes",
    plumbing_terminated="yes",
    carpentry_installed="yes",
    furniture_present="yes",
)


def surfaces(**overrides) -> SurfaceState:
    return SurfaceState(**{**DONE, **overrides})


class TestPhaseRouting:
    def test_complete_room_reaches_styling(self):
        verdict = classify(surfaces())
        assert verdict.phase is Phase.STYLING_RESTRUCTURE
        assert verdict.confidence >= 0.9

    def test_missing_floor_holds_at_surface(self):
        assert classify(surfaces(flooring_installed="no")).phase is Phase.SURFACE_FINISHING

    def test_missing_carpentry_holds_at_fixtures(self):
        assert (
            classify(surfaces(carpentry_installed="no")).phase is Phase.FIXTURES_CARPENTRY
        )

    def test_surfaces_evaluated_before_fixtures(self):
        """A room with no floor is in SURFACE_FINISHING regardless of how many
        cabinets are already hanging -- work happens in build order."""
        verdict = classify(surfaces(flooring_installed="no", carpentry_installed="no"))
        assert verdict.phase is Phase.SURFACE_FINISHING


class TestPartialBlocksProgression:
    """The load-bearing rule: half-done work holds the room in place."""

    @pytest.mark.parametrize(
        "signal", ["walls_painted", "flooring_installed", "ceiling_finished"]
    )
    def test_partial_surface_blocks(self, signal):
        verdict = classify(surfaces(**{signal: "partial"}))
        assert verdict.phase is Phase.SURFACE_FINISHING
        assert signal in verdict.blocking_signals

    @pytest.mark.parametrize(
        "signal",
        ["electrical_terminated", "plumbing_terminated", "carpentry_installed"],
    )
    def test_partial_fixture_blocks(self, signal):
        verdict = classify(surfaces(**{signal: "partial"}))
        assert verdict.phase is Phase.FIXTURES_CARPENTRY
        assert signal in verdict.blocking_signals

    def test_partial_never_rounds_up_to_complete(self):
        """Rounding PARTIAL to YES would quote a half-painted room as finished
        and omit the remaining work."""
        assert classify(surfaces(walls_painted="partial")).phase is not Phase.STYLING_RESTRUCTURE

    def test_can_progress_reports_the_blocker(self):
        ok, blockers = can_progress(surfaces(walls_painted="partial"))
        assert not ok
        assert "walls_painted" in blockers

    def test_complete_room_can_progress(self):
        ok, blockers = can_progress(surfaces())
        assert ok
        assert blockers == ()


class TestUnknownYieldsLowConfidence:
    def test_all_unknown_routes_to_review(self):
        verdict = classify(SurfaceState())
        assert verdict.needs_review
        assert verdict.confidence < REVIEW_THRESHOLD

    def test_unknown_does_not_produce_a_confident_wrong_answer(self):
        verdict = classify(surfaces(flooring_installed="unknown", walls_painted="unknown"))
        assert verdict.confidence < REVIEW_THRESHOLD
        assert verdict.phase is Phase.SURFACE_FINISHING

    def test_unknown_signals_are_listed(self):
        verdict = classify(surfaces(plumbing_terminated="unknown"))
        assert "plumbing_terminated" in verdict.unknown_signals

    def test_definite_no_beats_unknown_for_confidence(self):
        """A known-absent floor is a certain verdict; an unassessed one is not."""
        certain = classify(surfaces(flooring_installed="no"))
        uncertain = classify(surfaces(flooring_installed="no", walls_painted="unknown"))
        assert certain.confidence > uncertain.confidence

    def test_verdict_always_explains_itself(self):
        assert classify(SurfaceState()).explain()
        assert classify(surfaces()).explain()


class TestRestartFromEmpty:
    def test_produces_a_confident_bare_shell(self):
        """'Gut it and start over' is a real decision, so it must yield a
        legitimate scene state -- not a room full of UNKNOWNs that routes to
        human review."""
        verdict = classify(restart_from_empty())
        assert verdict.phase is Phase.SURFACE_FINISHING
        assert not verdict.needs_review

    def test_every_signal_is_definite(self):
        state = restart_from_empty()
        for field in state.model_fields:
            assert getattr(state, field) == Tri.NO.value


class TestCapabilityProbe:
    def test_gpu_without_weights_cannot_run_locally(self, clean_env):
        """The failure this design exists to prevent: a 48 GB card with no
        weights on disk must not route to LOCAL_FULL."""
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(
                present=True, name="RTX A6000", vram_mb=49140, torch_cuda=True
            ),
            health_check=lambda key: False,
        )
        caps = probe.detect()
        assert caps.path is not ExecutionPath.LOCAL_FULL
        assert caps.path is ExecutionPath.MOCK
        assert any("NO weights" in r for r in caps.reasons)

    def test_gpu_without_weights_prefers_cloud_when_available(self, clean_env, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(present=True, name="A6000", vram_mb=49140),
            health_check=lambda key: True,
        )
        assert probe.detect().path is ExecutionPath.CLOUD_API

    def test_gpu_with_full_weights_routes_local(self, clean_env):
        d = tempfile.mkdtemp()
        pathlib.Path(d, "sdxl-turbo-fp16.safetensors").write_text("x")
        probe = CapabilityProbe(
            model_dir=d,
            gpu_detector=lambda: GpuInfo(present=True, name="A6000", vram_mb=49140),
            health_check=lambda key: False,
        )
        assert probe.detect().path is ExecutionPath.LOCAL_FULL

    def test_weights_matched_by_substring_not_exact_name(self, clean_env):
        """Weights ship under many names depending on who quantised them."""
        d = tempfile.mkdtemp()
        pathlib.Path(d, "my_custom_sdxl_merge.v3.safetensors").write_text("x")
        probe = CapabilityProbe(
            model_dir=d,
            gpu_detector=lambda: GpuInfo(present=True, vram_mb=49140),
            health_check=lambda key: False,
        )
        assert probe.detect().full_weights

    def test_light_weights_without_gpu(self, clean_env):
        d = tempfile.mkdtemp()
        pathlib.Path(d, "lcm-dreamshaper.safetensors").write_text("x")
        probe = CapabilityProbe(
            model_dir=d,
            gpu_detector=lambda: GpuInfo(present=False),
            health_check=lambda key: False,
        )
        assert probe.detect().path is ExecutionPath.LOCAL_LIGHT

    def test_insufficient_vram_falls_through(self, clean_env):
        d = tempfile.mkdtemp()
        pathlib.Path(d, "sdxl-base.safetensors").write_text("x")
        probe = CapabilityProbe(
            model_dir=d,
            gpu_detector=lambda: GpuInfo(present=True, name="GTX 1050", vram_mb=2048),
            health_check=lambda key: False,
        )
        assert probe.detect().path is not ExecutionPath.LOCAL_FULL

    def test_cloud_requires_a_healthy_endpoint(self, clean_env, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(present=False),
            health_check=lambda key: False,
        )
        caps = probe.detect()
        assert caps.api_key_present
        assert not caps.api_healthy
        assert caps.path is ExecutionPath.MOCK

    def test_force_override(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORCE_EXECUTION_PATH", "LOCAL_FULL")
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(present=False),
            health_check=lambda key: False,
        )
        caps = probe.detect()
        assert caps.path is ExecutionPath.LOCAL_FULL
        assert caps.forced

    def test_invalid_override_is_ignored_not_fatal(self, clean_env, monkeypatch):
        monkeypatch.setenv("FORCE_EXECUTION_PATH", "TELEPORT")
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(present=False),
            health_check=lambda key: False,
        )
        caps = probe.detect()
        assert caps.path is ExecutionPath.MOCK
        assert not caps.forced
        assert any("invalid" in r for r in caps.reasons)

    def test_skip_healthcheck_env(self, clean_env, monkeypatch):
        """Serverless cold starts pay the health ping on the first request of
        every container, which is where the latency budget is tightest."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("PROBE_SKIP_HEALTHCHECK", "1")
        from interior_ai.perception.probe import _default_health_check

        assert _default_health_check("any-key") is True

    def test_results_are_cached(self, clean_env):
        calls = {"n": 0}

        def counting_detector() -> GpuInfo:
            calls["n"] += 1
            return GpuInfo(present=False)

        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=counting_detector,
            health_check=lambda key: False,
        )
        probe.detect()
        probe.detect()
        probe.detect()
        assert calls["n"] == 1

    def test_invalidate_forces_a_reprobe(self, clean_env):
        calls = {"n": 0}

        def counting_detector() -> GpuInfo:
            calls["n"] += 1
            return GpuInfo(present=False)

        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=counting_detector,
            health_check=lambda key: False,
        )
        probe.detect()
        probe.invalidate()
        probe.detect()
        assert calls["n"] == 2

    def test_capabilities_always_explain(self, clean_env):
        probe = CapabilityProbe(
            model_dir=tempfile.mkdtemp(),
            gpu_detector=lambda: GpuInfo(present=False),
            health_check=lambda key: False,
        )
        assert probe.detect().explain()
