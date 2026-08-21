"""Tests for RoomGridTickIntegration — Metronome + Compiler + EventBus wiring.

Covers:
  1. Basic tick / tick_batch with and without integrations
  2. Compiler hot-swap hook fires correctly
  3. Metronome synchronization dispatch
  4. FleetEventBus metrics emission
  5. Graceful fallback when optional components are missing
  6. No breakage of existing RoomGrid API
"""

import sys
import numpy as np
import pytest

from nerve.room_grid import RoomGrid
from nerve.room_grid_tick_integration import RoomGridTickIntegration, TickMetrics


@pytest.fixture
def bus_fixture():
    """Return a FleetEventBus that records all emitted events."""
    from nexus.fleet_event_bus import FleetEventBus

    bus = FleetEventBus()
    bus._test_events: list[dict] = []

    def _capture(ev):
        bus._test_events.append(ev.to_dict())

    bus.on("grid_tick_metrics", _capture)
    bus.on("grid_tick_error", _capture)
    bus.on("compiler_hot_swap", _capture)
    bus.on("metronome_tick_error", _capture)
    return bus


@pytest.fixture
def grid_100():
    np.random.seed(42)
    return RoomGrid(100)


@pytest.fixture
def signal_64():
    np.random.seed(42)
    return np.random.randn(64).astype(np.float32)


@pytest.fixture
def signals_batch():
    np.random.seed(42)
    return np.random.randn(8, 64).astype(np.float32)


# ── Basics ──────────────────────────────────────────────────────


class TestTickIntegrationBasics:
    """Instantiation, enable/disable, status."""

    def test_integration_instantiates(self, grid_100):
        integration = RoomGridTickIntegration(grid_100)
        assert integration.grid is grid_100
        assert integration.metronome is None
        assert integration.compiler_swap is None
        assert integration.event_bus is None
        assert integration._enabled is True

    def test_disable_and_enable(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        integration.disable()
        assert integration._enabled is False
        # Should still tick (falls back to raw grid.tick)
        result = integration.tick(signal_64)
        assert "fired" in result

        integration.enable()
        assert integration._enabled is True
        result = integration.tick(signal_64)
        assert "fired" in result

    def test_get_status(self, grid_100):
        integration = RoomGridTickIntegration(grid_100)
        status = integration.get_status()
        assert status["enabled"] is True
        assert status["tick_count"] == 0
        assert status["has_metronome"] is False
        assert status["has_compiler_swap"] is False
        assert status["has_event_bus"] is False
        assert status["grid_n_rooms"] == 100

    def test_status_updates_after_ticks(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        integration.tick(signal_64)
        integration.tick(signal_64)
        status = integration.get_status()
        assert status["tick_count"] == 2
        assert status["avg_duration_ms"] > 0


# ── Single tick ─────────────────────────────────────────────────


class TestTickSingle:
    """integration.tick() with various component combinations."""

    def test_tick_no_components(self, grid_100, signal_64):
        """Plain tick: no metronome, no compiler, no bus."""
        integration = RoomGridTickIntegration(grid_100)
        result = integration.tick(signal_64)
        assert "fired" in result
        assert "ids" in result
        assert "tick" in result
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_tick_with_event_bus(self, grid_100, signal_64, bus_fixture):
        integration = RoomGridTickIntegration(grid_100, event_bus=bus_fixture)
        result = integration.tick(signal_64)
        assert result["fired"] >= 0

        # Bus should have received a metrics event
        metrics_events = [
            e for e in bus_fixture._test_events if e["type"] == "grid_tick_metrics"
        ]
        assert len(metrics_events) == 1
        payload = metrics_events[0]["payload"]
        assert payload["n_rooms"] == 100
        assert "thermal_pressure" in payload
        assert "active_ratio" in payload

    def test_tick_with_metronome(self, grid_100, signal_64):
        from nerve.metronome_integration import MetronomeIntegration

        metro = MetronomeIntegration(grid_100, devices=["cuda:0"])
        metro.enable()
        integration = RoomGridTickIntegration(grid_100, metronome=metro)
        result = integration.tick(signal_64)
        assert "fired" in result
        # Metronome status attached
        assert "metronome" in result
        assert "offline_devices" in result["metronome"]

    def test_tick_metronome_offline_device(self, grid_100, signal_64):
        from nerve.metronome_integration import MetronomeIntegration

        # Use a very short timeout so the device goes offline immediately
        metro = MetronomeIntegration(
            grid_100, devices=["test_dev"], heartbeat_timeout_sec=0.01
        )
        metro.enable()
        # Manually age the heartbeat so device appears offline
        metro._devices["test_dev"].last_heartbeat = 0.0
        integration = RoomGridTickIntegration(grid_100, metronome=metro)
        result = integration.tick(signal_64)
        assert "metronome" in result
        assert "test_dev" in result["metronome"].get("offline_devices", [])

    def test_tick_with_all_components(self, grid_100, signal_64, bus_fixture):
        from nerve.metronome_integration import MetronomeIntegration
        from compiler.hot_swap_integration import CompilerHotSwap

        metro = MetronomeIntegration(grid_100)
        swap = CompilerHotSwap(grid_100)
        swap.enable_auto_compile()

        integration = RoomGridTickIntegration(
            grid_100,
            metronome=metro,
            compiler_swap=swap,
            event_bus=bus_fixture,
        )
        result = integration.tick(signal_64)
        assert "fired" in result
        assert "duration_ms" in result

        # Bus should have at least metrics
        assert any(e["type"] == "grid_tick_metrics" for e in bus_fixture._test_events)


# ── Batch tick ────────────────────────────────────────────────────


class TestTickBatch:
    """integration.tick_batch() — metronome sync + aggregate metrics."""

    def test_batch_no_components(self, grid_100, signals_batch):
        integration = RoomGridTickIntegration(grid_100)
        results = integration.tick_batch(signals_batch)
        assert len(results) == len(signals_batch)
        for r in results:
            assert "fired" in r
            assert "tick" in r
            assert "duration_ms" in r

    def test_batch_with_event_bus(self, grid_100, signals_batch, bus_fixture):
        integration = RoomGridTickIntegration(grid_100, event_bus=bus_fixture)
        results = integration.tick_batch(signals_batch)
        assert len(results) == len(signals_batch)

        metrics_events = [
            e for e in bus_fixture._test_events if e["type"] == "grid_tick_metrics"
        ]
        assert len(metrics_events) == 1
        # Batch aggregate should report total fired
        total_fired = sum(r["fired"] for r in results)
        assert metrics_events[0]["payload"]["fired_count"] == total_fired

    def test_batch_with_metronome(self, grid_100, signals_batch):
        from nerve.metronome_integration import MetronomeIntegration

        metro = MetronomeIntegration(grid_100, devices=["cuda:0"])
        metro.enable()
        integration = RoomGridTickIntegration(grid_100, metronome=metro)
        results = integration.tick_batch(signals_batch)
        assert len(results) == len(signals_batch)
        # Every result should carry metronome metadata
        for r in results:
            assert "metronome" in r

    def test_batch_disabled(self, grid_100, signals_batch):
        integration = RoomGridTickIntegration(grid_100)
        integration.disable()
        results = integration.tick_batch(signals_batch)
        assert len(results) == len(signals_batch)
        # Should not have duration_ms injected when disabled
        # (raw grid.tick_batch doesn't add it)
        for r in results:
            assert "fired" in r


# ── Compiler hot-swap hook ──────────────────────────────────────


class TestCompilerHook:
    """Compiler check_and_compile fires at tick time."""

    def test_compiler_check_fires_on_tick(self, grid_100, signal_64):
        from compiler.hot_swap_integration import CompilerHotSwap

        swap = CompilerHotSwap(grid_100)
        swap.enable_auto_compile()
        # Manually mutate config hash so check_and_compile triggers
        swap._last_config_hash = "forced_old_hash"

        integration = RoomGridTickIntegration(grid_100, compiler_swap=swap)
        result = integration.tick(signal_64)
        assert "fired" in result
        # Compiler should have incremented compile_count
        assert swap._compile_count >= 1

    def test_compiler_no_compile_when_disabled(self, grid_100, signal_64):
        from compiler.hot_swap_integration import CompilerHotSwap

        swap = CompilerHotSwap(grid_100)
        swap.enable_auto_compile()
        swap.disable_auto_compile()

        integration = RoomGridTickIntegration(grid_100, compiler_swap=swap)
        before = swap._compile_count
        integration.tick(signal_64)
        assert swap._compile_count == before

    def test_compiler_non_fatal_failure(self, grid_100, signal_64, bus_fixture):
        """A broken compiler should not crash the tick."""

        class BrokenCompiler:
            def check_and_compile(self):
                raise RuntimeError("compile boom")

        integration = RoomGridTickIntegration(
            grid_100,
            compiler_swap=BrokenCompiler(),
            event_bus=bus_fixture,
        )
        result = integration.tick(signal_64)
        assert "fired" in result  # tick survived


# ── EventBus edge cases ───────────────────────────────────────────


class TestEventBusEdgeCases:
    """Graceful degradation when event bus is missing / broken."""

    def test_no_bus_no_crash(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        result = integration.tick(signal_64)
        assert "fired" in result

    def test_broken_bus_no_crash(self, grid_100, signal_64):
        class BrokenBus:
            def emit(self, ev, source=""):
                raise RuntimeError("bus boom")

        integration = RoomGridTickIntegration(grid_100, event_bus=BrokenBus())
        result = integration.tick(signal_64)
        assert "fired" in result

    def test_bus_without_emit_no_crash(self, grid_100, signal_64):
        class WeirdBus:
            pass

        integration = RoomGridTickIntegration(grid_100, event_bus=WeirdBus())
        result = integration.tick(signal_64)
        assert "fired" in result


# ── Metrics construction ──────────────────────────────────────────


class TestMetrics:
    """TickMetrics accuracy and edge cases."""

    def test_metrics_fields_present(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        result = integration.tick(signal_64)
        metrics = integration._build_metrics(result, 1.234)
        assert metrics.tick == result["tick"]
        assert metrics.n_rooms == 100
        assert metrics.duration_ms == 1.234
        assert 0.0 <= metrics.thermal_pressure <= 1.0
        assert 0.0 <= metrics.active_ratio <= 1.0

    def test_metrics_to_dict(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        result = integration.tick(signal_64)
        metrics = integration._build_metrics(result, 2.0)
        d = metrics.to_dict()
        assert set(d.keys()) == {
            "tick",
            "n_rooms",
            "fired_count",
            "active_ratio",
            "thermal_pressure",
            "backend",
            "duration_ms",
            "timestamp",
        }

    def test_metrics_backend_detection(self, grid_100, signal_64):
        integration = RoomGridTickIntegration(grid_100)
        result = integration.tick(signal_64)
        metrics = integration._build_metrics(result, 1.0)
        # Without CUDA or Rust, should fall back to numpy
        assert metrics.backend == "numpy"


# ── Existing API non-regression ───────────────────────────────────


class TestNoRegression:
    """Verify RoomGrid.tick() and tick_batch() still work standalone."""

    def test_standalone_grid_tick_unchanged(self, grid_100, signal_64):
        """RoomGrid.tick() must not be monkey-patched by integration."""
        result = grid_100.tick(signal_64)
        assert "duration_ms" not in result  # raw grid does not add this
        assert "fired" in result

    def test_standalone_grid_tick_batch_unchanged(self, grid_100, signals_batch):
        results = grid_100.tick_batch(signals_batch)
        assert len(results) == len(signals_batch)
        for r in results:
            assert "duration_ms" not in r  # raw grid does not add this

    def test_multiple_integrations_on_same_grid(self, grid_100, signal_64, bus_fixture):
        """Two integration wrappers on one grid should not interfere."""
        int1 = RoomGridTickIntegration(grid_100, event_bus=bus_fixture)
        int2 = RoomGridTickIntegration(grid_100)
        r1 = int1.tick(signal_64)
        r2 = int2.tick(signal_64)
        # Both should succeed; underlying grid advances independently
        assert "fired" in r1
        assert "fired" in r2
        assert r2["tick"] > r1["tick"]  # grid has advanced


# ── Global cleanup ───────────────────────────────────────────────


def pytest_sessionfinish(session, exitstatus):
    """Restore any lingering compiler hot-swaps."""
    mod = sys.modules.get("nerve.room_grid")
    if mod is None:
        return
    for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
        obj = getattr(mod, attr, None)
        if obj is not None and hasattr(obj, "_sunset_original"):
            setattr(mod, attr, obj._sunset_original)
        if attr == "_tick_routing_compiled" and hasattr(mod, attr):
            delattr(mod, attr)
