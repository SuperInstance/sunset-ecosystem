"""tests/test_constraint_bridge.py — Tests for upgraded ConstraintBridge with FFI."""

import pytest
import numpy as np
import math

from swarm.constraint_bridge import ConstraintBridge, SnapResult, FFI_AVAILABLE


class TestSnapVector:
    """Pythagorean vector snapping."""

    def test_snap_unit_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([0.577, 0.816])
        assert isinstance(result, SnapResult)
        assert result.exact.shape == (2,)
        assert abs(np.linalg.norm(result.exact) - 1.0) < 0.01

    def test_snap_exact_on_circle(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 3-4-5 triangle normalized: (0.6, 0.8)
        result = bridge.snap_vector([0.6, 0.8])
        assert result.triple is not None
        a, b, c = result.triple
        assert a * a + b * b == c * c

    def test_snap_zero_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([0.0, 0.0])
        assert np.allclose(result.exact, [0.0, 0.0])
        assert result.noise == 0.0

    def test_snap_non_2d_raises(self):
        bridge = ConstraintBridge(dim=256, density=50)
        with pytest.raises(ValueError, match="Only 2D vectors"):
            bridge.snap_vector([0.1, 0.2, 0.3])

    def test_snap_noise_nonnegative(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([0.1, 0.9])
        assert result.noise >= 0.0

    def test_snap_idempotent_on_triple(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Find a triple and snap it
        triples = bridge._triples[:5]
        for a, b, c in triples:
            vec = [a / c, b / c]
            result = bridge.snap_vector(vec)
            assert result.noise < 0.01


class TestBatchSnap:
    """Batch snapping operations."""

    def test_batch_snap_returns_list(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
        results = bridge.batch_snap(vectors)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SnapResult)

    def test_batch_snap_empty(self):
        bridge = ConstraintBridge(dim=256, density=50)
        results = bridge.batch_snap([])
        assert results == []

    def test_batch_snap_noise_values(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.1, 0.9], [0.5, 0.5]]
        results = bridge.batch_snap(vectors)
        for r in results:
            assert r.noise >= 0.0


class TestQuantization:
    """Embedding quantization modes."""

    def test_quantize_ternary(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.2, -0.2, 0.05, -0.05, 0.0]
        q = bridge.quantize_embedding(emb, mode="ternary")
        assert q.dtype == np.int8
        assert list(q) == [1, -1, 0, 0, 0]

    def test_quantize_polar(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.6, 0.8, 0.1]
        q = bridge.quantize_embedding(emb, mode="polar")
        assert q.shape == (2,)
        assert abs(np.linalg.norm(q) - 1.0) < 0.01

    def test_quantize_turbo(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.123, 0.456, 0.789]
        q = bridge.quantize_embedding(emb, mode="turbo")
        # Should be rounded to 1 decimal place
        assert np.allclose(q, [0.1, 0.5, 0.8], atol=0.05)

    def test_quantize_auto_high_norm(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [2.0, 3.0, 1.0]  # norm > 1
        q = bridge.quantize_embedding(emb, mode="auto")
        # Should select polar
        assert q.shape == (2,)

    def test_quantize_auto_low_norm(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.1, 0.1, 0.1]  # norm < 1
        q = bridge.quantize_embedding(emb, mode="auto")
        # Should select ternary
        assert q.dtype == np.int8


class TestHolonomy:
    """Holonomy (cycle consistency) checks."""

    def test_holonomy_consistent_cycle(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Three vectors that sum to zero rotation
        cycle = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        assert bridge.check_holonomy(cycle) is True

    def test_holonomy_trivial(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.check_holonomy([]) is True
        assert bridge.check_holonomy([[1.0, 0.0]]) is True
        assert bridge.check_holonomy([[1.0, 0.0], [0.0, 1.0]]) is True

    def test_holonomy_inconsistent(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Cycle with repeated vectors: angles are 90°, 0°, 90°, 0° = 180°
        # 180° mod 360° = 180°, not 0° or 360° → inconsistent
        cycle = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        result = bridge.check_holonomy(cycle)
        assert result is False

    def test_holonomy_perfect_square(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Square: should have zero holonomy
        cycle = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        assert bridge.check_holonomy(cycle) is True


class TestFFIFunctions:
    """FFI-accelerated or pure Python fallback functions."""

    def test_eisenstein_norm(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.eisenstein_norm(1, 0) == 1
        assert bridge.eisenstein_norm(2, 1) == 3
        assert bridge.eisenstein_norm(0, 1) == 1
        assert bridge.eisenstein_norm(3, 2) == 7

    def test_eisenstein_norm_symmetry(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # N(a,b) = a² - ab + b² should be symmetric
        for a in range(-5, 6):
            for b in range(-5, 6):
                assert bridge.eisenstein_norm(a, b) == bridge.eisenstein_norm(b, a)

    def test_laman_is_rigid_triangle(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 3 vertices, 3 edges = triangle → rigid
        assert bridge.laman_is_rigid(3, 3) is True

    def test_laman_is_rigid_not_rigid(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 4 vertices, 4 edges ≠ 2*4-3 = 5 → not rigid
        assert bridge.laman_is_rigid(4, 4) is False

    def test_laman_is_rigid_exactly_5(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 4 vertices, 5 edges = 2*4-3 → rigid
        assert bridge.laman_is_rigid(4, 5) is True

    def test_holonomy_check_consistent(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.holonomy_check([0.0, 0.0, 0.0], 1e-6)
        assert result == 1.0

    def test_holonomy_check_inconsistent(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.holonomy_check([0.0, 1.0, 2.0], 0.1)
        assert result == 0.0

    def test_constraint_check_inside(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_check(0.5, 0.0, 1.0) is True

    def test_constraint_check_outside(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_check(1.1, 0.0, 1.0) is False
        assert bridge.constraint_check(-0.1, 0.0, 1.0) is False

    def test_constraint_violation_inside(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_violation(0.5, 0.0, 1.0) == 0.0

    def test_constraint_violation_above(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_violation(1.2, 0.0, 1.0) == pytest.approx(0.2)

    def test_constraint_violation_below(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_violation(-0.3, 0.0, 1.0) == pytest.approx(0.3)

    def test_spline_interpolate(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Simple line from 0 to 1, zero tangents
        result = bridge.spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.5)
        assert abs(result - 0.5) < 1e-9

    def test_spline_interpolate_endpoints(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.spline_interpolate(0.0, 1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
        assert bridge.spline_interpolate(0.0, 1.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_deadband_filter_no_change(self):
        bridge = ConstraintBridge(dim=256, density=50)
        val, last = bridge.deadband_filter(0.05, 0.0, 0.1)
        assert val == 0.0
        assert last == 0.0

    def test_deadband_filter_change(self):
        bridge = ConstraintBridge(dim=256, density=50)
        val, last = bridge.deadband_filter(0.2, 0.0, 0.1)
        assert val == 0.2
        assert last == 0.2

    def test_manhattan_distance(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.manhattan_distance([1.0, 2.0, 3.0], [4.0, 0.0, 3.0]) == 5.0

    def test_manhattan_distance_same(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.manhattan_distance([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_cascade_match_found(self):
        bridge = ConstraintBridge(dim=256, density=50)
        query = [1.0, 1.0, 1.0]
        candidates = [[0.0, 0.0, 0.0], [1.1, 1.0, 1.0], [2.0, 2.0, 2.0]]
        thresholds = [0.5, 1.5]
        idx = bridge.cascade_match(query, candidates, thresholds)
        assert idx == 1  # Second candidate matches at tier 0

    def test_cascade_match_not_found(self):
        bridge = ConstraintBridge(dim=256, density=50)
        query = [10.0, 10.0, 10.0]
        candidates = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        thresholds = [0.5]
        idx = bridge.cascade_match(query, candidates, thresholds)
        assert idx == -1

    def test_pythagorean48_encode(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 2:1 ratio = octave = 12 semitones
        assert bridge.pythagorean48_encode(2, 1) == 12
        # 3:2 ratio = perfect fifth = 7 semitones
        assert bridge.pythagorean48_encode(3, 2) == 7

    def test_pythagorean48_encode_unison(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.pythagorean48_encode(1, 1) == 0


class TestLamanRigidity:
    """Graph rigidity checks."""

    def test_laman_rigid_triangle(self):
        bridge = ConstraintBridge(dim=256, density=50)
        edges = [(0, 1), (1, 2), (0, 2)]
        assert bridge.laman_rigid(edges, 3) is True

    def test_laman_rigid_not_enough_edges(self):
        bridge = ConstraintBridge(dim=256, density=50)
        edges = [(0, 1), (1, 2)]  # 2 edges, need 3 for 3 nodes
        assert bridge.laman_rigid(edges, 3) is False

    def test_laman_rigid_too_many_edges(self):
        bridge = ConstraintBridge(dim=256, density=50)
        edges = [(0, 1), (0, 1), (1, 2)]  # Duplicate edge
        assert bridge.laman_rigid(edges, 3) is False

    def test_laman_rigid_quadrilateral_with_diagonal(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # 4 nodes, 5 edges = 2*4-3 = 5 → rigid
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        assert bridge.laman_rigid(edges, 4) is True


class TestStats:
    """Bridge introspection."""

    def test_get_stats(self):
        bridge = ConstraintBridge(dim=256, density=50)
        stats = bridge.get_stats()
        assert stats["dim"] == 256
        assert stats["density"] == 50
        assert stats["ct_available"] is False  # Not available in test env
        assert stats["triples_cached"] > 0

    def test_stats_ffi_flag(self):
        bridge = ConstraintBridge(dim=256, density=50)
        stats = bridge.get_stats()
        assert "ffi_available" in stats
        # Should be False or True depending on env
        assert isinstance(stats["ffi_available"], bool)


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_snap_large_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([100.0, 100.0])
        assert abs(np.linalg.norm(result.exact) - 1.0) < 0.01

    def test_snap_negative(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([-0.5, -0.5])
        assert result.noise >= 0.0
        assert result.exact.shape == (2,)

    def test_quantize_empty(self):
        bridge = ConstraintBridge(dim=256, density=50)
        q = bridge.quantize_embedding([], mode="turbo")
        assert q.shape == (0,)

    def test_holonomy_single_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Single vector repeated — trivially consistent
        cycle = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert bridge.check_holonomy(cycle) is True

    def test_batch_snap_with_zero(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.0, 0.0], [1.0, 0.0]]
        results = bridge.batch_snap(vectors)
        assert len(results) == 2
        assert np.allclose(results[0].exact, [0.0, 0.0])
