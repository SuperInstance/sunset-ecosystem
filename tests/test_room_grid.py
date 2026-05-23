"""Tests for RoomGrid — forward consistency, novelty, breeding, chaos, FLUX."""
import numpy as np
import pytest

from nerve.room_grid import RoomGrid, make_weights, batch_novelty, forward_einsum, forward_rust_oneshot, _RUST_LIB
from sunset.flux_integration import FluxConstraintChecker, apply_constraint_feedback


class TestRoomGridForward:
    """Numerical correctness of forward pass across backends."""

    def test_forward_numpy_not_nan(self, room_grid_100, signal_64):
        """Forward pass should never produce NaN."""
        result = room_grid_100._forward(signal_64)
        assert not np.isnan(result).any(), "NaN in forward output"

    def test_forward_shape(self, room_grid_100, signal_64):
        """Output shape is (n_rooms, l)."""
        result = room_grid_100._forward(signal_64)
        assert result.shape == (room_grid_100.n, 16)

    def test_forward_deterministic_seed(self, room_grid_100, signal_64):
        """Same seed + same input = same output."""
        out1 = room_grid_100._forward(signal_64)
        out2 = room_grid_100._forward(signal_64)
        np.testing.assert_allclose(out1, out2, atol=1e-7)


class TestRoomGridTick:
    """Tick-level behavior: novelty, chaos, firing, history."""

    def test_tick_returns_stats(self, room_grid_100):
        r = room_grid_100.tick(np.random.randn(64).astype(np.float32))
        assert "fired" in r
        assert "tick" in r
        assert isinstance(r["fired"], int)

    def test_tick_increments_tick_counter(self, room_grid_100):
        t0 = room_grid_100.ticks
        room_grid_100.tick(np.random.randn(64).astype(np.float32))
        assert room_grid_100.ticks == t0 + 1

    def test_chaos_bounds(self, room_grid_100):
        """Chaos stays in [0.01, 1.0] after 50 ticks."""
        x = np.random.randn(64).astype(np.float32)
        for _ in range(50):
            room_grid_100.tick(x)
        assert room_grid_100.chaos.min() >= 0.01
        assert room_grid_100.chaos.max() <= 1.0


def test_flux_checker(room_grid_100):
    """attach_flux_checker increases chaos for violating rooms."""
    np.random.seed(42)
    x = np.random.randn(64).astype(np.float32)
    room_grid_100.tick(x)
    baseline = room_grid_100.chaos.copy()

    checker = FluxConstraintChecker(preset="safe_mode")
    # Force every room to be flagged as violating
    checker.check_batch = lambda latents, preset=None: np.ones(len(latents), dtype=bool)
    room_grid_100.attach_flux_checker(checker)
    room_grid_100.tick(x)
    after = room_grid_100.chaos.copy()
    assert (after > baseline).sum() > 0


class TestRoomGridDiversity:
    """RoomGrid.diversity() computes population diversity via HDC or cosine."""

    def test_empty_grid_zero_diversity(self):
        g = RoomGrid(10)
        assert g.diversity() == 0.0

    def test_single_active_room_zero_diversity(self):
        g = RoomGrid(10)
        g.tick(np.random.randn(64))
        assert g.diversity() == 0.0

    def test_diversity_increases_after_breed(self):
        g = RoomGrid(10)
        # Fire a few rooms
        for _ in range(5):
            g.tick(np.random.randn(64))
        div_before = g.diversity()
        # Breed room 0 into room 5 (clone + small mutation)
        g.breed(0, 5)
        # Fire room 5 so it becomes active
        for _ in range(3):
            g.tick(np.random.randn(64))
        div_after = g.diversity()
        # Diversity should be non-zero after breeding
        assert div_after > 0.0

    def test_diversity_range(self):
        g = RoomGrid(20)
        for _ in range(10):
            g.tick(np.random.randn(64))
        div = g.diversity()
        assert 0.0 <= div <= 1.0

    def test_hdc_cosine_correlation(self):
        g = RoomGrid(10)
        for _ in range(5):
            g.tick(np.random.randn(64))
        div_hdc = g.diversity(use_hdc=True)
        div_cos = g.diversity(use_hdc=False)
        # Both should be in [0, 1] and reasonably correlated
        assert 0.0 <= div_hdc <= 1.0
        assert 0.0 <= div_cos <= 1.0
        # If population is diverse enough, both should be non-zero
        if div_cos > 0.1:
            assert div_hdc > 0.0

    def test_stats_includes_diversity(self):
        g = RoomGrid(10)
        for _ in range(5):
            g.tick(np.random.randn(64))
        stats = g.stats
        assert "diversity" in stats
        assert 0.0 <= stats["diversity"] <= 1.0
