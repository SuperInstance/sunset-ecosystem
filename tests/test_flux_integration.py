"""Tests for FLUX constraint checker — Python backend, presets, RoomGrid hook."""
import numpy as np
import pytest

from sunset.flux_integration import FluxConstraintChecker, PRESETS


class TestPythonBackend:
    """Pure-Python constraint checking."""

    def test_detects_bounds_violation(self):
        """Values outside bounds are flagged."""
        checker = FluxConstraintChecker(preset="neural_bounds")
        latents = np.array([[0.0] * 16, [20.0] + [0.0] * 15], dtype=np.float32)
        violations = checker.check_batch(latents)
        assert violations[0] == False, "Clean room flagged"
        assert violations[1] == True, "Violating room not flagged"

    def test_detects_l2_violation(self):
        """L2 norm exceeding threshold is flagged."""
        checker = FluxConstraintChecker(preset="neural_bounds")
        # Room with all dims = 10 → L2 = sqrt(16 * 100) = 40
        latents = np.full((2, 16), 10.0, dtype=np.float32)
        latents[0] = 0.0  # Clean room
        violations = checker.check_batch(latents)
        assert violations[0] == False
        assert violations[1] == True

    def test_all_pass(self):
        """Zeros pass all constraints."""
        checker = FluxConstraintChecker(preset="safe_mode")
        latents = np.zeros((10, 16), dtype=np.float32)
        violations = checker.check_batch(latents)
        assert not violations.any(), f"All-pass failed: {violations.sum()}/{len(violations)}"


class TestPresets:
    """Preset strictness ordering."""

    def test_safe_mode_stricter(self):
        """safe_mode catches more violations than neural_bounds."""
        neural = FluxConstraintChecker(preset="neural_bounds")
        safe = FluxConstraintChecker(preset="safe_mode")

        # Values that pass neural but fail safe
        latents = np.full((1, 16), 6.0, dtype=np.float32)
        v_neural = neural.check_batch(latents)
        v_safe = safe.check_batch(latents)
        assert not v_neural[0], "Should pass neural_bounds"
        assert v_safe[0], "Should fail safe_mode"

    def test_exploration_loose(self):
        """exploration preset allows very large values."""
        exploration = FluxConstraintChecker(preset="exploration")
        latents = np.full((1, 16), 20.0, dtype=np.float32)
        violations = exploration.check_batch(latents)
        assert not violations[0], "exploration should allow value=20"


class TestRoomGridIntegration:
    """FLUX checker attached to RoomGrid."""

    def test_attach_checker(self, room_grid_100, flux_checker):
        """attach_flux_checker stores checker."""
        room_grid_100.attach_flux_checker(flux_checker)
        assert room_grid_100._flux_checker is not None

    def test_tick_with_checker(self, room_grid_100, flux_checker, signal_64):
        """Tick works with checker attached."""
        room_grid_100.attach_flux_checker(flux_checker)
        result = room_grid_100.tick(signal_64)
        assert "fired" in result

    def test_feedback_runs_clean(self, room_grid_100, flux_checker):
        """Feedback code path executes even with no violations (clean run)."""
        room_grid_100.attach_flux_checker(flux_checker)
        # Zeros should produce no violations — but code path must still run
        zero_signal = np.zeros(64, dtype=np.float32)
        # Should not raise
        result = room_grid_100.tick(zero_signal)
        assert "fired" in result
        # Chaos may change slightly due to random firing, but should stay in bounds
        assert (room_grid_100.chaos >= 0.01).all()
        assert (room_grid_100.chaos <= 1.0).all()


class TestDetailedViolations:
    """get_violations() returns structured data."""

    def test_returns_violation_objects(self):
        """get_violations returns list of ConstraintViolation."""
        checker = FluxConstraintChecker(preset="neural_bounds")
        latents = np.array([[20.0] + [0.0] * 15], dtype=np.float32)
        room_ids = ["room-0"]
        violations = checker.get_violations(latents, room_ids)
        assert len(violations) == 1
        assert violations[0].room_idx == 0
        assert violations[0].room_id == "room-0"
        assert "bounds" in violations[0].constraint_name

    def test_empty_for_clean(self):
        """get_violations returns empty list for clean rooms."""
        checker = FluxConstraintChecker(preset="neural_bounds")
        latents = np.zeros((5, 16), dtype=np.float32)
        room_ids = [f"room-{i}" for i in range(5)]
        violations = checker.get_violations(latents, room_ids)
        assert len(violations) == 0


# ── New top-level unit tests (Task requirements) ──

def test_python_backend_detects_violations():
    """Large latent values are flagged by the Python backend."""
    from sunset.flux_integration import _PythonBackend, PRESETS
    backend = _PythonBackend()
    latents = np.zeros((10, 16), dtype=np.float32)
    latents[0, 0] = 15.0  # exceeds safe_mode bound of 5
    violations = backend.check_batch(latents, PRESETS["safe_mode"])
    assert violations[0]


def test_safe_mode_stricter():
    """safe_mode catches more violations than neural_bounds."""
    checker = FluxConstraintChecker(preset="neural_bounds")
    latents = np.zeros((10, 16), dtype=np.float32)
    latents[:, 0] = 7.0  # >5 but <10
    neural = checker.check_batch(latents, "neural_bounds")
    safe = checker.check_batch(latents, "safe_mode")
    assert safe.sum() > neural.sum()


def test_room_grid_integration():
    """RoomGrid with attached checker shows increased chaos for violations."""
    from nerve.room_grid import RoomGrid
    np.random.seed(42)
    grid = RoomGrid(10)
    x = np.random.randn(64).astype(np.float32)
    grid.tick(x)
    baseline = grid.chaos.copy()

    checker = FluxConstraintChecker(preset="safe_mode")
    # Force every room to be flagged as violating
    checker.check_batch = lambda latents, preset=None: np.ones(len(latents), dtype=bool)
    grid.attach_flux_checker(checker)
    grid.tick(x)
    after = grid.chaos.copy()
    assert (after > baseline).sum() > 0


def test_rust_backend_when_available():
    """Use Rust backend if libflux_vm.so is present."""
    import os
    so_paths = [
        os.path.join(os.path.dirname(__file__), "..", "flux-vm-v3-temp", "target", "release", "libflux_vm.so"),
        "/usr/local/lib/libflux_vm.so",
    ]
    found = any(os.path.exists(p) for p in so_paths)
    if not found:
        pytest.skip("libflux_vm.so not available")
    from sunset.flux_integration import _RustBackend
    idx = [i for i, p in enumerate(so_paths) if os.path.exists(p)][0]
    backend = _RustBackend(so_paths[idx])
    assert backend.available
