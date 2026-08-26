"""tests/test_constraint_bridge_upgrade.py — Tests for ConstraintBridge upgrades.

Covers:
  * exact_pythagorean_snap correctness & quadrant handling
  * batch_snap vectorised performance
  * LRU triple-cache behaviour
  * "exact" quantization mode
  * FFI fallback paths (mock works when .so absent)
  * round-trip snap → unscale fidelity
"""

import math
import time

import numpy as np
import pytest

from swarm.constraint_bridge import (
    ConstraintBridge,
    SnapResult,
    _generate_triples_cached,
    FFI_SO_EXISTS,
)


# ── Exact Pythagorean Snap ─────────────────────────────────────────


class TestExactSnap:
    """exact_pythagorean_snap single-vector tests."""

    def test_returns_tuple_of_four(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.exact_pythagorean_snap(0.5, 0.5)
        assert len(result) == 4
        sx, sy, triple, noise = result
        assert isinstance(sx, float)
        assert isinstance(sy, float)
        assert isinstance(noise, float)

    def test_zero_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(0.0, 0.0)
        assert sx == pytest.approx(0.0)
        assert sy == pytest.approx(0.0)
        assert triple == (0, 0, 1)
        assert noise == pytest.approx(0.0)

    def test_pythagorean_property(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(0.577, 0.816)
        assert triple is not None
        a, b, c = triple
        assert a * a + b * b == c * c

    def test_first_quadrant_positive(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(0.6, 0.8)
        assert sx > 0
        assert sy > 0

    def test_second_quadrant_negative_x(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(-0.6, 0.8)
        assert sx < 0
        assert sy > 0

    def test_third_quadrant_both_negative(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(-0.6, -0.8)
        assert sx < 0
        assert sy < 0

    def test_fourth_quadrant_negative_y(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(0.6, -0.8)
        assert sx > 0
        assert sy < 0

    def test_near_known_triple(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # (0.6, 0.8) is exactly 3-4-5
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(0.6, 0.8)
        assert triple is not None
        a, b, c = triple
        assert (a, b, c) == (3, 4, 5)
        assert noise < 1e-6

    def test_noise_is_nonnegative(self):
        bridge = ConstraintBridge(dim=256, density=50)
        for vec in [(1.0, 0.0), (0.0, 1.0), (-0.5, 0.5), (0.3, -0.9)]:
            sx, sy, triple, noise = bridge.exact_pythagorean_snap(*vec)
            assert noise >= 0.0

    def test_large_coordinates(self):
        bridge = ConstraintBridge(dim=256, density=50)
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(1e6, 2e6)
        assert triple is not None
        a, b, c = triple
        assert a * a + b * b == c * c


# ── Batch Snap ─────────────────────────────────────────────────────


class TestBatchSnap:
    """Vectorised batch_snap tests."""

    def test_returns_list_of_snap_results(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
        results = bridge.batch_snap(vectors)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SnapResult)

    def test_all_triples_exact(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.1, 0.9], [-0.5, 0.5], [0.8, -0.2]]
        results = bridge.batch_snap(vectors)
        for r in results:
            if r.triple is not None and r.triple != (0, 0, 1):
                a, b, c = r.triple
                assert a * a + b * b == c * c

    def test_empty_batch(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.batch_snap([]) == []

    def test_batch_with_zero_vectors(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        results = bridge.batch_snap(vectors)
        assert len(results) == 3
        assert np.allclose(results[0].exact, [0.0, 0.0])
        assert results[0].triple == (0, 0, 1)
        assert results[0].noise == pytest.approx(0.0)

    def test_batch_quadrant_preservation(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[-0.6, 0.8], [-0.6, -0.8], [0.6, -0.8]]
        results = bridge.batch_snap(vectors)
        assert results[0].exact[0] < 0 and results[0].exact[1] > 0
        assert results[1].exact[0] < 0 and results[1].exact[1] < 0
        assert results[2].exact[0] > 0 and results[2].exact[1] < 0

    def test_batch_performance_1000_vectors(self):
        bridge = ConstraintBridge(dim=256, density=50)
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((1000, 2)).tolist()
        t0 = time.time()
        results = bridge.batch_snap(vectors)
        t1 = time.time()
        assert len(results) == 1000
        assert t1 - t0 < 1.0  # generous on shared VM

    def test_batch_performance_5000_vectors(self):
        bridge = ConstraintBridge(dim=256, density=50)
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((5000, 2)).tolist()
        t0 = time.time()
        results = bridge.batch_snap(vectors)
        t1 = time.time()
        assert len(results) == 5000
        assert t1 - t0 < 3.0  # generous on shared VM

    def test_batch_snap_noise_nonnegative(self):
        bridge = ConstraintBridge(dim=256, density=50)
        vectors = [[0.1, 0.9], [0.5, 0.5], [-0.8, 0.2]]
        results = bridge.batch_snap(vectors)
        for r in results:
            assert r.noise >= 0.0

    def test_batch_snap_idempotent_on_triples(self):
        bridge = ConstraintBridge(dim=256, density=50)
        triples = bridge._triples[:5]
        vectors = [[a / c, b / c] for a, b, c in triples]
        results = bridge.batch_snap(vectors)
        for r in results:
            assert r.noise < 1e-6


# ── Cache ──────────────────────────────────────────────────────────


class TestTripleCache:
    """LRU-cache behaviour for Pythagorean triples."""

    def test_cache_miss_on_first_access(self):
        _generate_triples_cached.cache_clear()
        info_before = _generate_triples_cached.cache_info()
        bridge = ConstraintBridge(dim=256, density=50)
        # Accessing bridge triggers triple generation
        _ = bridge._triples
        info_after = _generate_triples_cached.cache_info()
        assert info_after.misses == info_before.misses + 1

    def test_cache_hit_on_same_density(self):
        _generate_triples_cached.cache_clear()
        b1 = ConstraintBridge(dim=256, density=50)
        _ = b1._triples
        info1 = _generate_triples_cached.cache_info()
        assert info1.misses == 1
        b2 = ConstraintBridge(dim=256, density=50)
        _ = b2._triples
        info2 = _generate_triples_cached.cache_info()
        assert info2.hits == info1.hits + 1
        assert info2.misses == info1.misses  # no new miss

    def test_cache_miss_on_different_density(self):
        _generate_triples_cached.cache_clear()
        b1 = ConstraintBridge(dim=256, density=50)
        _ = b1._triples
        info1 = _generate_triples_cached.cache_info()
        b2 = ConstraintBridge(dim=256, density=60)
        _ = b2._triples
        info2 = _generate_triples_cached.cache_info()
        assert info2.misses == info1.misses + 1

    def test_cache_info_nonzero_after_use(self):
        _generate_triples_cached.cache_clear()
        for d in (30, 40, 50):
            _ = ConstraintBridge(dim=256, density=d)
        info = _generate_triples_cached.cache_info()
        assert info.currsize > 0


# ── Exact Quantization ─────────────────────────────────────────────


class TestExactQuantization:
    """quantize_embedding(mode='exact') tests."""

    def test_exact_mode_returns_float32(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.1, 0.9, 0.2, 0.8]
        q = bridge.quantize_embedding(emb, mode="exact")
        assert q.dtype == np.float32

    def test_exact_mode_even_dimension(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.6, 0.8, 0.3, 0.4]
        q = bridge.quantize_embedding(emb, mode="exact")
        assert len(q) == 4
        # Each pair should be near-unit (projected onto Pythagorean direction)
        for i in range(0, 4, 2):
            norm = float(np.linalg.norm(q[i : i + 2]))
            assert abs(norm - 1.0) < 0.01

    def test_exact_mode_odd_dimension_pad(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.6, 0.8, 0.3]  # odd length
        q = bridge.quantize_embedding(emb, mode="exact")
        assert len(q) == 3

    def test_exact_mode_empty(self):
        bridge = ConstraintBridge(dim=256, density=50)
        q = bridge.quantize_embedding([], mode="exact")
        assert len(q) == 0

    def test_exact_mode_zero_vector(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [0.0, 0.0, 0.0, 0.0]
        q = bridge.quantize_embedding(emb, mode="exact")
        assert np.allclose(q, [0.0, 0.0, 0.0, 0.0])

    def test_exact_mode_preserves_shape(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = list(np.random.randn(64))
        q = bridge.quantize_embedding(emb, mode="exact")
        assert q.shape == (64,)

    def test_exact_mode_negative_values(self):
        bridge = ConstraintBridge(dim=256, density=50)
        emb = [-0.6, -0.8, -0.3, 0.4]
        q = bridge.quantize_embedding(emb, mode="exact")
        assert q[0] < 0
        assert q[1] < 0


# ── FFI Fallback ───────────────────────────────────────────────────


class TestFFIFallback:
    """When libsuperinstance_ffi.so is absent, mocks / pure Python work."""

    def test_so_not_present(self):
        # In this test environment the .so is expected to be missing
        assert FFI_SO_EXISTS is False

    def test_eisenstein_norm_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.eisenstein_norm(2, 1) == 3

    def test_laman_is_rigid_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.laman_is_rigid(3, 3) is True
        assert bridge.laman_is_rigid(4, 4) is False

    def test_holonomy_check_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.holonomy_check([0.0, 0.0, 0.0], 1e-6) == 1.0

    def test_constraint_check_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.constraint_check(0.5, 0.0, 1.0) is True
        assert bridge.constraint_check(1.1, 0.0, 1.0) is False

    def test_manhattan_distance_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        assert bridge.manhattan_distance([1.0, 2.0, 3.0], [4.0, 0.0, 3.0]) == 5.0

    def test_cascade_match_via_fallback(self):
        bridge = ConstraintBridge(dim=256, density=50)
        idx = bridge.cascade_match(
            [1.0, 1.0, 1.0],
            [[0.0, 0.0, 0.0], [1.1, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [0.5, 1.5],
        )
        assert idx == 1


# ── Round-trip fidelity ────────────────────────────────────────────


class TestRoundTrip:
    """snap → unscale should stay close to the original direction."""

    def test_single_vector_roundtrip(self):
        bridge = ConstraintBridge(dim=256, density=50)
        orig = np.array([0.3, 0.9])
        sx, sy, triple, noise = bridge.exact_pythagorean_snap(*orig)
        # Re-scale snapped direction by original norm
        orig_norm = float(np.linalg.norm(orig))
        unscaled = np.array([sx, sy]) * orig_norm
        # Direction should be very close; magnitude exact
        cos_sim = float(
            np.dot(orig, unscaled)
            / (np.linalg.norm(orig) * np.linalg.norm(unscaled) + 1e-8)
        )
        assert abs(cos_sim - 1.0) < 0.05  # within ~13°

    def test_batch_roundtrip(self):
        bridge = ConstraintBridge(dim=256, density=50)
        rng = np.random.default_rng(42)
        origs = rng.standard_normal((100, 2))
        results = bridge.batch_snap(origs.tolist())
        cos_sims = []
        for orig, r in zip(origs, results):
            if np.allclose(orig, 0):
                continue
            unscaled = r.exact * float(np.linalg.norm(orig))
            cos_sim = float(
                np.dot(orig, unscaled)
                / (np.linalg.norm(orig) * np.linalg.norm(unscaled) + 1e-8)
            )
            cos_sims.append(cos_sim)
        assert min(cos_sims) > 0.90  # worst case still within ~26°

    def test_idempotent_snap(self):
        bridge = ConstraintBridge(dim=256, density=50)
        # Snapping a triple coordinate twice should give the same result
        triples = bridge._triples[:10]
        for a, b, c in triples:
            vec = [a / c, b / c]
            sx1, sy1, t1, n1 = bridge.exact_pythagorean_snap(*vec)
            sx2, sy2, t2, n2 = bridge.exact_pythagorean_snap(sx1, sy1)
            assert sx1 == pytest.approx(sx2, abs=1e-9)
            assert sy1 == pytest.approx(sy2, abs=1e-9)
            assert n2 < 1e-9


# ── Stats & Introspection ──────────────────────────────────────────


class TestStats:
    def test_stats_contains_ffi_so_exists(self):
        bridge = ConstraintBridge(dim=256, density=50)
        stats = bridge.get_stats()
        assert "ffi_so_exists" in stats
        assert isinstance(stats["ffi_so_exists"], bool)

    def test_triples_cached_nonzero(self):
        bridge = ConstraintBridge(dim=256, density=50)
        stats = bridge.get_stats()
        assert stats["triples_cached"] > 0


# ── Backward compatibility ───────────────────────────────────────


class TestBackwardCompatibility:
    def test_old_snap_vector_api_unchanged(self):
        bridge = ConstraintBridge(dim=256, density=50)
        result = bridge.snap_vector([0.6, 0.8])
        assert isinstance(result, SnapResult)
        assert result.triple == (3, 4, 5)

    def test_old_batch_snap_api_unchanged(self):
        bridge = ConstraintBridge(dim=256, density=50)
        results = bridge.batch_snap([[0.6, 0.8], [0.0, 0.0]])
        assert len(results) == 2
        assert isinstance(results[0], SnapResult)
