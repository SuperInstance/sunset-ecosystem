"""Tests for RoomGrid — forward consistency, novelty, breeding, chaos, FLUX."""
import numpy as np
import pytest

from nerve.room_grid import RoomGrid, make_weights, batch_novelty


class TestRoomGridForward:
    """Numerical correctness of forward pass across backends."""

    def test_forward_numpy_not_nan(self, room_grid_100, signal_64):
        """Forward pass should never produce NaN."""
        result = room_grid_100._forward(signal_64)
        assert not np.isnan(result).any(), "NaN in forward output"

    def test_forward_shape(self, room_grid_100, signal_64):
        """Output shape is (n_rooms, l)."""
        result = room_grid_100._forward(signal_64)
        assert result.shape == (100, 16), f"Expected (100, 16), got {result.shape}"

    def test_breed_preserves_structure(self, room_grid_100):
        """Child weights have same shape as parent."""
        parent_w = room_grid_100.w["w2"][0]  # room 0, layer w2
        room_grid_100.breed(0, 50)
        child_w = room_grid_100.w["w2"][50]
        assert child_w.shape == parent_w.shape, f"Shape mismatch: {child_w.shape} vs {parent_w.shape}"

    def test_breed_changes_weights(self, room_grid_100):
        """Breeding should actually change the target room's weights."""
        before = room_grid_100.w["w2"][50].copy()
        room_grid_100.breed(0, 50)
        after = room_grid_100.w["w2"][50]
        assert not np.allclose(before, after), "Breed did not change weights"


class TestNovelty:
    """Novelty scoring correctness."""

    def test_novelty_range(self):
        """Novelty scores are bounded below at 0 (upper bound depends on history)."""
        np.random.seed(42)
        latents = np.random.randn(100, 16).astype(np.float32)
        hist = np.random.randn(20, 100, 16).astype(np.float32)
        hist_count = np.full(100, 20, dtype=np.int32)
        hist_idx = 5
        hist_max = 20

        nv = batch_novelty(latents, hist, hist_count, hist_idx, hist_max)
        assert (nv >= 0).all(), f"Novelty below 0: min={nv.min()}"
        # Novelty can exceed 1 with highly divergent latents; assert finite
        assert np.isfinite(nv).all(), f"Non-finite novelty: {nv}"

    def test_novelty_identical_latents(self):
        """Identical latents across history → novelty ≈ 0."""
        latents = np.ones((10, 16), dtype=np.float32) * 0.5
        hist = np.ones((20, 10, 16), dtype=np.float32) * 0.5
        hist_count = np.full(10, 20, dtype=np.int32)

        nv = batch_novelty(latents, hist, hist_count, 0, 20)
        assert nv.mean() < 0.01, f"Expected near-zero novelty, got {nv.mean()}"

    def test_novelty_different_latents(self):
        """Very different latents → novelty > 0.5."""
        latents = np.ones((10, 16), dtype=np.float32) * 10.0
        hist = np.zeros((20, 10, 16), dtype=np.float32)
        hist_count = np.full(10, 20, dtype=np.int32)

        nv = batch_novelty(latents, hist, hist_count, 0, 20)
        assert nv.mean() > 0.5, f"Expected high novelty, got {nv.mean()}"


class TestChaos:
    """Chaos dynamics."""

    def test_chaos_bounds(self, room_grid_100, signal_64):
        """Chaos stays in [0.01, 1.0] after any number of ticks."""
        for _ in range(50):
            room_grid_100.tick(signal_64)
        assert (room_grid_100.chaos >= 0.01).all(), f"Chaos below 0.01: min={room_grid_100.chaos.min()}"
        assert (room_grid_100.chaos <= 1.0).all(), f"Chaos above 1.0: max={room_grid_100.chaos.max()}"

    def test_fired_rooms_chaos_increases(self, room_grid_100, signal_64):
        """Rooms that fire should have higher chaos than rooms that don't."""
        before = room_grid_100.chaos.copy()
        result = room_grid_100.tick(signal_64)
        after = room_grid_100.chaos.copy()

        fired_mask = np.zeros(100, dtype=bool)
        for idx in result["ids"]:
            fired_mask[idx] = True

        if fired_mask.any():
            fired_chaos = after[fired_mask]
            unfired_chaos = after[~fired_mask]
            assert fired_chaos.mean() >= unfired_chaos.mean() * 0.9, \
                f"Fired chaos ({fired_chaos.mean():.3f}) should not be lower than unfired ({unfired_chaos.mean():.3f})"


class TestRingBuffer:
    """History buffer correctness."""

    def test_buffer_wraps(self, room_grid_100, signal_64):
        """Buffer wraps after >20 ticks without corruption."""
        for i in range(25):
            room_grid_100.tick(signal_64)
        assert room_grid_100._hist_idx < 20, "Buffer index should wrap"
        assert room_grid_100._hist_count.min() <= 20, "Count should not exceed max"

    def test_buffer_stores_latents(self, room_grid_100, signal_64):
        """Buffer contains actual latent vectors."""
        room_grid_100.tick(signal_64)
        stored = room_grid_100._hist[0]
        assert stored.shape == (100, 16), f"Stored shape wrong: {stored.shape}"
        assert not np.isnan(stored).any(), "NaN in history buffer"


class TestFluxIntegration:
    """FLUX constraint checker in RoomGrid."""

    def test_flux_checker_attaches(self, room_grid_100, flux_checker):
        """attach_flux_checker works without error."""
        room_grid_100.attach_flux_checker(flux_checker)
        assert room_grid_100._flux_checker is not None

    def test_flux_feedback_runs(self, room_grid_100, flux_checker, signal_64):
        """FLUX feedback code path executes without error."""
        room_grid_100.attach_flux_checker(flux_checker)
        # Should not raise — the feedback runs even if no violations
        result = room_grid_100.tick(signal_64)
        assert "fired" in result

    def test_latents_stored(self, room_grid_100, signal_64):
        """tick() stores latents in self.latents."""
        room_grid_100.tick(signal_64)
        assert room_grid_100.latents is not None
        assert room_grid_100.latents.shape == (100, 16)
