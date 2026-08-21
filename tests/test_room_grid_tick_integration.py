"""Tests for room_grid_tick_integration.py — RoomGrid tick orchestration."""

import numpy as np
import pytest
from unittest.mock import MagicMock, call

from nerve.room_grid_tick_integration import RoomGridTickIntegration, TickMetrics


class MockGrid:
    """Minimal stand-in for RoomGrid."""

    def __init__(self, n=10):
        self.n = n
        self.ticks = 0
        self.activity = np.zeros(n, dtype=np.float32)
        self.chaos = np.zeros(n, dtype=np.float32)

    def tick(self, x):
        self.ticks += 1
        self.activity[0] = 1.0  # fire room 0
        return {"tick": self.ticks, "fired": 1, "ids": [0]}

    def tick_batch(self, signals):
        self.ticks += 1
        return [
            {"tick": self.ticks, "fired": 1, "ids": [i]} for i in range(len(signals))
        ]


class TestTickMetrics:
    def test_to_dict(self):
        m = TickMetrics(
            tick=1,
            n_rooms=10,
            fired_count=3,
            active_ratio=0.3,
            thermal_pressure=0.5,
            backend="numpy",
            duration_ms=1.23,
        )
        d = m.to_dict()
        assert d["tick"] == 1
        assert d["n_rooms"] == 10
        assert d["backend"] == "numpy"
        assert "timestamp" in d


class TestRoomGridTickIntegration:
    def test_init_defaults(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        assert integration.grid is grid
        assert integration._enabled is True
        assert integration.metronome is None

    def test_enable_disable(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        integration.disable()
        assert integration._enabled is False
        integration.enable()
        assert integration._enabled is True

    def test_tick_disabled_falls_back(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        integration.disable()
        result = integration.tick(np.zeros(64, dtype=np.float32))
        assert result["fired"] == 1
        assert "duration_ms" not in result  # raw grid tick

    def test_tick_enabled(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        result = integration.tick(np.zeros(64, dtype=np.float32))
        assert "duration_ms" in result
        assert result["fired"] == 1
        assert integration._tick_count == 1

    def test_tick_with_metronome(self):
        grid = MockGrid(10)
        metro = MagicMock()
        metro.check_devices.return_value = []
        metro.get_drift_correction.return_value = 0.5
        metro._enabled = True

        integration = RoomGridTickIntegration(grid, metronome=metro)
        result = integration.tick(np.zeros(64, dtype=np.float32))
        assert "metronome" in result
        assert result["metronome"]["drift_correction_ms"] == 0.5
        metro.check_devices.assert_called_once()

    def test_tick_metronome_no_methods(self):
        grid = MockGrid(10)
        metro = MagicMock()
        del metro.check_devices
        del metro.get_drift_correction

        integration = RoomGridTickIntegration(grid, metronome=metro)
        result = integration.tick(np.zeros(64, dtype=np.float32))
        # metronome key may or may not be present depending on status
        # but should not crash
        assert "duration_ms" in result

    def test_tick_batch_disabled(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        integration.disable()
        results = integration.tick_batch(np.zeros((3, 64), dtype=np.float32))
        assert len(results) == 3
        assert "duration_ms" not in results[0]

    def test_tick_batch_enabled(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        results = integration.tick_batch(np.zeros((3, 64), dtype=np.float32))
        assert len(results) == 3
        assert "duration_ms" in results[0]
        assert integration._tick_count == 1

    def test_tick_batch_with_metronome(self):
        grid = MockGrid(10)
        metro = MagicMock()
        metro.tick.return_value = {"sync": True}

        integration = RoomGridTickIntegration(grid, metronome=metro)
        results = integration.tick_batch(np.zeros((2, 64), dtype=np.float32))
        assert len(results) == 2
        assert results[0]["metronome"]["sync"] is True
        metro.tick.assert_called_once()

    def test_tick_batch_metronome_error(self):
        grid = MockGrid(10)
        metro = MagicMock()
        metro.tick.side_effect = RuntimeError("boom")

        bus = MagicMock()
        integration = RoomGridTickIntegration(grid, metronome=metro, event_bus=bus)
        results = integration.tick_batch(np.zeros((2, 64), dtype=np.float32))
        assert len(results) == 2
        # Event bus should have received error event
        assert bus.emit.called

    def test_compiler_swap(self):
        grid = MockGrid(10)
        compiler = MagicMock()
        compiler.check_and_compile.return_value = MagicMock(
            success=True, compile_time_ms=42.0
        )

        bus = MagicMock()
        integration = RoomGridTickIntegration(
            grid, compiler_swap=compiler, event_bus=bus
        )
        integration.tick(np.zeros(64, dtype=np.float32))
        compiler.check_and_compile.assert_called_once()
        assert bus.emit.called

    def test_compiler_swap_no_method(self):
        grid = MockGrid(10)
        compiler = MagicMock()
        del compiler.check_and_compile

        integration = RoomGridTickIntegration(grid, compiler_swap=compiler)
        integration.tick(np.zeros(64, dtype=np.float32))
        # should not crash

    def test_compiler_swap_exception(self):
        grid = MockGrid(10)
        compiler = MagicMock()
        compiler.check_and_compile.side_effect = RuntimeError("compile failed")

        integration = RoomGridTickIntegration(grid, compiler_swap=compiler)
        integration.tick(np.zeros(64, dtype=np.float32))
        # should not crash

    def test_event_bus(self):
        grid = MockGrid(10)
        bus = MagicMock()
        integration = RoomGridTickIntegration(grid, event_bus=bus)
        integration.tick(np.zeros(64, dtype=np.float32))
        assert bus.emit.called

    def test_event_bus_no_emit(self):
        grid = MockGrid(10)
        bus = MagicMock()
        del bus.emit

        integration = RoomGridTickIntegration(grid, event_bus=bus)
        integration.tick(np.zeros(64, dtype=np.float32))
        # should not crash

    def test_event_bus_exception(self):
        grid = MockGrid(10)
        bus = MagicMock()
        bus.emit.side_effect = RuntimeError("emit failed")

        integration = RoomGridTickIntegration(grid, event_bus=bus)
        integration.tick(np.zeros(64, dtype=np.float32))
        # should not crash

    def test_grid_tick_error(self):
        grid = MockGrid(10)
        grid.tick = MagicMock(side_effect=RuntimeError("tick failed"))

        bus = MagicMock()
        integration = RoomGridTickIntegration(grid, event_bus=bus)
        with pytest.raises(RuntimeError, match="tick failed"):
            integration.tick(np.zeros(64, dtype=np.float32))
        assert bus.emit.called

    def test_grid_tick_batch_error(self):
        grid = MockGrid(10)
        grid.tick_batch = MagicMock(side_effect=RuntimeError("batch failed"))

        bus = MagicMock()
        integration = RoomGridTickIntegration(grid, event_bus=bus)
        with pytest.raises(RuntimeError, match="batch failed"):
            integration.tick_batch(np.zeros((2, 64), dtype=np.float32))
        assert bus.emit.called

    def test_get_status(self):
        grid = MockGrid(10)
        metro = MagicMock()
        compiler = MagicMock()
        bus = MagicMock()

        integration = RoomGridTickIntegration(
            grid, metronome=metro, compiler_swap=compiler, event_bus=bus
        )
        integration.tick(np.zeros(64, dtype=np.float32))

        status = integration.get_status()
        assert status["enabled"] is True
        assert status["tick_count"] == 1
        assert status["has_metronome"] is True
        assert status["has_compiler_swap"] is True
        assert status["has_event_bus"] is True
        assert status["grid_n_rooms"] == 10

    def test_backend_detection_numpy(self):
        grid = MockGrid(10)
        integration = RoomGridTickIntegration(grid)
        result = integration.tick(np.zeros(64, dtype=np.float32))
        assert result["duration_ms"] > 0
        # _build_metrics should detect numpy backend
        status = integration.get_status()
        assert status["grid_n_rooms"] == 10

    def test_backend_detection_cuda(self):
        grid = MockGrid(10)
        grid._cuda_grid = True  # fake CUDA backend
        integration = RoomGridTickIntegration(grid)
        metrics = integration._build_metrics({"fired": 1}, 1.0)
        assert metrics.backend == "cuda"

    def test_backend_detection_rust(self):
        grid = MockGrid(10)
        grid._rust_grid = True  # fake Rust backend
        integration = RoomGridTickIntegration(grid)
        metrics = integration._build_metrics({"fired": 1}, 1.0)
        assert metrics.backend == "rust_persistent"

    def test_build_metrics_no_grid_attrs(self):
        """If grid lacks n/activity/chaos, metrics should still build."""
        grid = MagicMock()
        grid.n = 0
        del grid.activity
        del grid.chaos

        integration = RoomGridTickIntegration(grid)
        metrics = integration._build_metrics({"fired": 0}, 1.0)
        assert metrics.n_rooms == 0
        assert metrics.active_ratio == 0.0
        assert metrics.thermal_pressure == 0.0
