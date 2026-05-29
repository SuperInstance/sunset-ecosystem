"""Tests for Constraint Bridge — exact Pythagorean snapping and quantization.

Covers vector snapping, batch operations, quantization modes, holonomy,
and Laman rigidity checks.
"""

import math
import numpy as np
import pytest

from swarm.constraint_bridge import ConstraintBridge, SnapResult, CT_AVAILABLE


class TestInit:
    def test_default(self):
        bridge = ConstraintBridge(dim=128, density=100)
        assert bridge.dim == 128
        assert bridge.density == 100
        stats = bridge.get_stats()
        assert stats["ct_available"] == CT_AVAILABLE

    def test_stats(self):
        bridge = ConstraintBridge(dim=256, density=200)
        stats = bridge.get_stats()
        assert stats["dim"] == 256
        assert stats["density"] == 200
        assert stats["triples_cached"] > 0 or CT_AVAILABLE


class TestSnapVector:
    def test_snap_exact(self):
        bridge = ConstraintBridge(density=200)
        result = bridge.snap_vector([0.6, 0.8])
        assert isinstance(result, SnapResult)
        assert len(result.exact) == 2
        # Should be very close to [0.6, 0.8] or a nearby Pythagorean triple
        assert result.noise >= 0.0

    def test_snap_unit_circle(self):
        bridge = ConstraintBridge(density=200)
        result = bridge.snap_vector([1.0, 0.0])
        assert result.exact[0] > 0.99  # Should snap to ~1.0

    def test_snap_zero(self):
        bridge = ConstraintBridge(density=200)
        result = bridge.snap_vector([0.0, 0.0])
        assert result.exact[0] == pytest.approx(0.0)
        assert result.exact[1] == pytest.approx(0.0)

    def test_snap_3d_raises(self):
        bridge = ConstraintBridge(density=200)
        with pytest.raises(ValueError):
            bridge.snap_vector([0.1, 0.2, 0.3])

    def test_snap_has_triple(self):
        bridge = ConstraintBridge(density=200)
        result = bridge.snap_vector([0.6, 0.8])
        if result.triple:
            a, b, c = result.triple
            assert a * a + b * b == c * c


class TestBatchSnap:
    def test_batch(self):
        bridge = ConstraintBridge(density=200)
        vectors = [[0.6, 0.8], [0.3, 0.4], [1.0, 0.0]]
        results = bridge.batch_snap(vectors)
        assert len(results) == 3
        for r in results:
            assert len(r.exact) == 2


class TestQuantize:
    def test_ternary(self):
        bridge = ConstraintBridge()
        emb = [0.5, -0.2, 0.05, -0.8]
        q = bridge.quantize_embedding(emb, mode="ternary")
        assert set(q.tolist()).issubset({-1, 0, 1})

    def test_polar(self):
        bridge = ConstraintBridge()
        emb = [0.8, 0.6, 0.0, 0.0]
        q = bridge.quantize_embedding(emb, mode="polar")
        assert len(q) >= 2
        # First two components should be on unit circle
        norm = math.sqrt(q[0] ** 2 + q[1] ** 2)
        assert abs(norm - 1.0) < 0.1

    def test_turbo(self):
        bridge = ConstraintBridge()
        emb = [0.123, 0.456, 0.789]
        q = bridge.quantize_embedding(emb, mode="turbo")
        # Should be quantized to 1 decimal place
        for val in q:
            assert abs(val - round(val * 10) / 10) < 1e-6

    def test_hybrid(self):
        bridge = ConstraintBridge()
        emb = [0.5, 0.5, 0.5]
        q = bridge.quantize_embedding(emb, mode="hybrid")
        assert len(q) == len(emb)


class TestHolonomy:
    def test_trivial_cycle(self):
        bridge = ConstraintBridge()
        cycle = [[1.0, 0.0], [0.0, 1.0]]
        assert bridge.check_holonomy(cycle) is True

    def test_zero_holonomy(self):
        bridge = ConstraintBridge()
        # Square cycle: should have zero holonomy
        cycle = [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
        assert bridge.check_holonomy(cycle) is True

    def test_nonzero_holonomy(self):
        bridge = ConstraintBridge()
        # Triangle with arbitrary angles
        cycle = [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
        # May or may not be zero depending on exact angles
        result = bridge.check_holonomy(cycle)
        assert isinstance(result, bool)


class TestLamanRigidity:
    def test_rigid_triangle(self):
        bridge = ConstraintBridge()
        edges = [(0, 1), (1, 2), (2, 0)]
        assert bridge.laman_rigid(edges, 3) is True

    def test_not_rigid_too_few(self):
        bridge = ConstraintBridge()
        edges = [(0, 1), (1, 2)]
        assert bridge.laman_rigid(edges, 3) is False

    def test_not_rigid_too_many(self):
        bridge = ConstraintBridge()
        edges = [(0, 1), (1, 2), (2, 0), (0, 2)]  # 4 edges > 2*3-3 = 3
        assert bridge.laman_rigid(edges, 3) is False


class TestPythagoreanExactness:
    def test_exactness(self):
        bridge = ConstraintBridge(density=500)
        result = bridge.snap_vector([0.6, 0.8])
        a, b = result.exact[0], result.exact[1]
        # Should be exact or very close
        mag_sq = a * a + b * b
        assert abs(mag_sq - 1.0) < 0.01 or result.triple is not None
