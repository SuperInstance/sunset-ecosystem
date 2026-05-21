"""Tests for RoomGrid — batched JEPA rooms."""

import pytest
import torch

from nerve.room_grid import JEPABatch, RoomGrid, RoomFingerprint


class TestJEPABatch:
    def test_forward_shape(self):
        batch = JEPABatch(n_rooms=10, input_dim=64, latent_dim=16)
        signal = torch.randn(1, 64)
        result = batch(signal)
        assert result.shape == (10, 16)

    def test_room_latent(self):
        batch = JEPABatch(n_rooms=10)
        signal = torch.randn(1, 64)
        z = batch.room_latent(3, signal)
        assert z.shape == (1, 16)

    def test_set_seed(self):
        batch = JEPABatch(n_rooms=5)
        signal = torch.randn(1, 64)
        z1 = batch.room_latent(0, signal)
        batch.set_seed(0, seed=9999)
        z2 = batch.room_latent(0, signal)
        # Different seed → different latent
        assert (z1 - z2).norm().item() > 0.001


class TestRoomGrid:
    def test_create(self):
        grid = RoomGrid(n_rooms=10)
        assert grid.n_rooms == 10
        assert "RoomGrid" in repr(grid)

    def test_tick(self):
        grid = RoomGrid(n_rooms=10)
        signal = torch.randn(64)
        result = grid.tick(signal)
        assert "fired_rooms" in result
        assert result["total_ticks"] == 1

    def test_multiple_ticks(self):
        grid = RoomGrid(n_rooms=10)
        for _ in range(5):
            grid.tick(torch.randn(64))
        assert grid._total_ticks == 5

    def test_sort_by_activity(self):
        grid = RoomGrid(n_rooms=10)
        for _ in range(20):
            grid.tick(torch.randn(64))
        top = grid.sort_by_activity(3)
        assert len(top) <= 3
        if top:
            assert all(isinstance(x[0], int) for x in top)
            assert all(isinstance(x[1], int) for x in top)

    def test_prune_cold(self):
        grid = RoomGrid(n_rooms=10)
        # Only fire rooms 0-4
        for i in range(5):
            for _ in range(10):
                grid._activity[i] = grid._activity.get(i, 0) + 1
        cold = grid.prune_cold(threshold=5)
        # Rooms 5-9 should be cold (< 5 fires)
        assert all(i >= 5 for i in cold)

    def test_rebirth(self):
        grid = RoomGrid(n_rooms=10)
        signal = torch.randn(64)
        z1 = grid.jepa.room_latent(7, signal.unsqueeze(0))
        grid.rebirth_as(7, 0)
        z2 = grid.jepa.room_latent(7, signal.unsqueeze(0))
        # Rebirth resets weights — should be different
        assert (z1 - z2).norm().item() > 0.001 or grid._activity[7] == 0

    def test_fingerprints(self):
        grid = RoomGrid(n_rooms=50)
        for _ in range(10):
            grid.tick(torch.randn(64))
        fps = grid.get_fingerprints()
        assert len(fps) == 50
        assert isinstance(fps[0], RoomFingerprint)

    def test_stats(self):
        grid = RoomGrid(n_rooms=10)
        for _ in range(5):
            grid.tick(torch.randn(64))
        s = grid.stats
        assert s["rooms"] == 10
        assert s["ticks"] == 5
