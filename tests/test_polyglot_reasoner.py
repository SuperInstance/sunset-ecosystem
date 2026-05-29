"""Tests for Polyglot Reasoner Python bridge.

Covers Python fallback, Rust/C++ FFI loading, and Mercury verification.
"""

import numpy as np
import pytest

from reasoning.python_bridge import PolyglotReasoner


class TestPolyglotReasoner:
    def test_init(self):
        r = PolyglotReasoner(dim=128)
        assert r.dim == 128
        assert r._backend in ("rust", "cpp", "python")

    def test_add_tile(self):
        r = PolyglotReasoner(dim=3)
        r.add_tile(1, [1.0, 0.0, 0.0])
        assert 1 in r._tiles
        assert len(r._tiles[1]) == 3

    def test_add_tile_wrong_dim(self):
        r = PolyglotReasoner(dim=3)
        with pytest.raises(ValueError):
            r.add_tile(1, [1.0, 0.0])

    def test_find_similar_python(self):
        r = PolyglotReasoner(dim=3, backend="python")
        r.add_tile(1, [1.0, 0.0, 0.0])
        r.add_tile(2, [0.0, 1.0, 0.0])
        r.add_tile(3, [0.5, 0.5, 0.0])
        
        results = r.find_similar([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1  # Most similar
        assert results[0][1] > 0.99

    def test_find_similar_empty(self):
        r = PolyglotReasoner(dim=3, backend="python")
        results = r.find_similar([1.0, 0.0, 0.0])
        assert results == []

    def test_find_similar_orthogonal(self):
        r = PolyglotReasoner(dim=3, backend="python")
        r.add_tile(1, [1.0, 0.0, 0.0])
        r.add_tile(2, [0.0, 1.0, 0.0])
        
        results = r.find_similar([1.0, 0.0, 0.0])
        assert results[0][0] == 1
        assert results[1][1] < 0.01  # Orthogonal should be ~0

    def test_stats(self):
        r = PolyglotReasoner(dim=128)
        r.add_tile(1, [1.0] + [0.0] * 127)
        stats = r.get_stats()
        assert stats["tile_count"] == 1
        assert stats["dim"] == 128

    def test_backend_selection(self):
        r = PolyglotReasoner(dim=128)
        assert r._backend in ("rust", "cpp", "python")

    def test_cosine_sim_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        score = PolyglotReasoner._cosine_sim(a, b)
        assert score == pytest.approx(1.0)

    def test_cosine_sim_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        score = PolyglotReasoner._cosine_sim(a, b)
        assert score == pytest.approx(0.0)

    def test_mercury_verify_unavailable(self):
        r = PolyglotReasoner(dim=3)
        r.add_tile(1, [1.0, 0.0, 0.0])
        result = r.verify_with_mercury([1.0, 0.0, 0.0], 1)
        assert result["verified"] is False
        assert "Mercury not available" in result["reason"]
