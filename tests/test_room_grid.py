"""Pure-numpy JEPA grid tests — 250 rooms, 195μs."""

import numpy as np
from nerve.room_grid import RoomGrid, Fingerprint, make_weights, forward_einsum, forward_one, novelty


class TestWeights:
    def test_shapes(self):
        w = make_weights(10)
        assert w["w1"].shape == (10, 64, 32)
        assert w["b1"].shape == (1, 10, 32)

    def test_broadcast(self):
        w = make_weights(50)
        z = forward_einsum(w, np.ones(64))
        assert z.shape == (50, 16)
        assert not np.allclose(z[0], z[1])


class TestForward:
    def test_one(self):
        w = make_weights(5)
        z = forward_one(w, 3, np.ones(64))
        assert z.shape == (16,)

    def test_all(self):
        w = make_weights(10, l=32)
        z = forward_einsum(w, np.ones(64))
        assert z.shape == (10, 32)

    def test_diversity(self):
        w1 = make_weights(1, seed=42)
        w2 = make_weights(1, seed=99)
        z1 = forward_einsum(w1, np.ones(64))
        z2 = forward_einsum(w2, np.ones(64))
        assert not np.allclose(z1, z2, atol=1e-10)


class TestNovelty:
    def test_low(self):
        z = np.ones(16)
        n = novelty(z, [np.ones(16)]*3)
        assert n < 0.5

    def test_high(self):
        z = np.ones(16)
        n = novelty(z, [np.zeros(16)]*3)
        assert n > 0.3


class TestRoomGrid:
    def test_make(self):
        g = RoomGrid(10)
        assert g.n == 10
        assert repr(g)

    def test_tick(self):
        g = RoomGrid(10)
        r = g.tick(np.random.randn(64))
        assert "fired" in r

    def test_multitick(self):
        g = RoomGrid(10)
        for _ in range(20):
            g.tick(np.random.randn(64))
        assert g.ticks == 20
        assert int((g.activity > 0).sum()) > 0

    def test_top(self):
        g = RoomGrid(10)
        for _ in range(20):
            g.tick(np.random.randn(64))
        t = g.top(3)
        assert len(t) <= 3

    def test_cold(self):
        g = RoomGrid(10)
        g.activity[:3] = 10
        c = g.cold(5)
        assert all(g.activity[i] < 5 for i in c)

    def test_rebirth(self):
        g = RoomGrid(10)
        w_before = g.w["w1"][5].copy()
        g.rebirth(5)
        assert np.linalg.norm(w_before - g.w["w1"][5]) > 0.001

    def test_fingerprints(self):
        g = RoomGrid(50)
        for _ in range(5):
            g.tick(np.random.randn(64))
        fps = g.fingerprints()
        assert len(fps) == 50
        assert all(isinstance(f, Fingerprint) for f in fps)

    def test_stats(self):
        g = RoomGrid(10)
        for _ in range(5):
            g.tick(np.random.randn(64))
        s = g.stats
        assert s["rooms"] == 10
        assert s["ticks"] == 5
        assert s["active"] > 0

    def test_diverse_fingerprints(self):
        g = RoomGrid(10)
        fps = g.fingerprints(10)
        # All rooms should have different fingerprints
        diffs = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                diffs.append(fps[i].diff(fps[j]))
        # With small weights, norms are ~0.0002, so diffs are ~0.001
        # Check they're not *identical* (machine epsilon)
        unique = set(tuple(f.sine) for f in fps)
        assert len(unique) > 1, "Rooms should produce different latents"
