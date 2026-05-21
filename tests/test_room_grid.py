"""Tests for pure-numpy room grid."""

import numpy as np
from nerve.room_grid import JEPAGrid, RoomFingerprint, make_jepa_weights, forward_batch, forward_one, compute_novelty


class TestWeights:
    def test_shapes(self):
        w = make_jepa_weights(10)
        assert w["W1"].shape == (10, 64, 32)
        assert w["b1"].shape == (10, 32)
        assert w["W2"].shape == (10, 32, 16)
        assert w["W3"].shape == (10, 16, 16)

    def test_output_rank(self):
        # Each room should produce different latent for same input
        w = make_jepa_weights(10, seed=42)
        z = forward_batch(w, np.random.randn(64))
        assert z.shape == (10, 16)
        assert not np.allclose(z[0], z[1])  # Different rooms, different latents


class TestForward:
    def test_forward_batch(self):
        w = make_jepa_weights(5)
        z = forward_batch(w, np.ones(64))
        assert z.shape == (5, 16)

    def test_forward_one(self):
        w = make_jepa_weights(10)
        z = forward_one(w, 3, np.ones(64))
        assert z.shape == (16,)

    def test_known_shape(self):
        w = make_jepa_weights(100, input_dim=64, latent_dim=32)
        z = forward_batch(w, np.ones(64))
        assert z.shape == (100, 32)

    def test_different_seeds_different_latents(self):
        w1 = make_jepa_weights(1, seed=42)
        w2 = make_jepa_weights(1, seed=99)
        z1 = forward_batch(w1, np.ones(64))
        z2 = forward_batch(w2, np.ones(64))
        # Small weights * single signal = may be close. Check they're not identical.
        assert not np.allclose(z1, z2, atol=1e-12), "Different seeds should differ"


class TestNovelty:
    def test_known_signal(self):
        z = np.ones(16)
        history = [np.ones(16), np.ones(16), np.ones(16)]
        n = compute_novelty(z, history)
        assert n < 0.5  # identical signals = low novelty

    def test_novel_signal(self):
        z = np.ones(16)
        history = [np.zeros(16), np.zeros(16), np.zeros(16)]
        n = compute_novelty(z, history)
        assert n > 0.3  # different signals = high novelty


class TestRoomGrid:
    def test_create(self):
        g = JEPAGrid(n_rooms=10)
        assert g.n_rooms == 10
        assert "JEPAGrid" in repr(g)

    def test_tick(self):
        g = JEPAGrid(n_rooms=10)
        r = g.tick(np.random.randn(64))
        assert "fired_rooms" in r
        assert r["total_ticks"] == 1

    def test_heatmap(self):
        g = JEPAGrid(n_rooms=10)
        for _ in range(20):
            g.tick(np.random.randn(64))
        hm = g.heatmap()
        assert hm.shape == (10,)
        assert hm.sum() > 0

    def test_top_rooms(self):
        g = JEPAGrid(n_rooms=10)
        for _ in range(20):
            g.tick(np.random.randn(64))
        top = g.top_rooms(3)
        assert len(top) <= 3

    def test_cold_rooms(self):
        g = JEPAGrid(n_rooms=10)
        # Fire some, leave others cold
        for i in range(3):
            for _ in range(10):
                g._activity[i] += 1
        cold = g.cold_rooms(threshold=5)
        for c in cold:
            assert g._activity[c] < 5

    def test_rebirth(self):
        g = JEPAGrid(n_rooms=10)
        w_before = g.weights["W1"][5].copy()
        g.rebirth(5)
        w_after = g.weights["W1"][5]
        assert np.linalg.norm(w_before - w_after) > 0.001
        assert g._activity[5] == 0

    def test_fingerprints(self):
        g = JEPAGrid(n_rooms=50)
        for _ in range(5):
            g.tick(np.random.randn(64))
        fps = g.fingerprints()
        assert len(fps) == 50
        assert all(isinstance(f, RoomFingerprint) for f in fps)

    def test_stats(self):
        g = JEPAGrid(n_rooms=10)
        for _ in range(5):
            g.tick(np.random.randn(64))
        s = g.stats
        assert s["rooms"] == 10
        assert s["ticks"] == 5
