"""Tests for reasoning.python_bridge — polyglot reasoner with Python fallback."""

import numpy as np

import pytest

from reasoning.python_bridge import PolyglotReasoner


class TestPolyglotReasoner:
    def test_create(self):
        r = PolyglotReasoner(dim=3)
        assert r.dim == 3
        assert r.get_stats()["tile_count"] == 0

    def test_add_tile(self):
        r = PolyglotReasoner(dim=3)
        r.add_tile(1, [1.0, 0.0, 0.0])
        assert r.get_stats()["tile_count"] == 1

    def test_add_tile_wrong_dim(self):
        r = PolyglotReasoner(dim=3)
        with pytest.raises(ValueError, match="dim"):
            r.add_tile(1, [1.0, 0.0])

    def test_find_similar(self):
        r = PolyglotReasoner(dim=3)
        r.add_tile(1, [1.0, 0.0, 0.0])
        r.add_tile(2, [0.0, 1.0, 0.0])
        results = r.find_similar([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_find_similar_top_k(self):
        r = PolyglotReasoner(dim=3)
        r.add_tile(1, [1.0, 0.0, 0.0])
        r.add_tile(2, [0.0, 1.0, 0.0])
        r.add_tile(3, [0.0, 0.0, 1.0])
        results = r.find_similar([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_find_similar_empty(self):
        r = PolyglotReasoner(dim=3)
        results = r.find_similar([1.0, 0.0, 0.0], top_k=2)
        assert results == []

    def test_normalize(self):
        r = PolyglotReasoner(dim=2)
        r.add_tile(1, [2.0, 0.0])
        results = r.find_similar([1.0, 0.0], top_k=1)
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_backend_selection(self):
        r = PolyglotReasoner(dim=3, backend="python")
        assert r._backend == "python"

    def test_stats(self):
        r = PolyglotReasoner(dim=3, backend="python")
        r.add_tile(1, [1.0, 0.0, 0.0])
        stats = r.get_stats()
        assert stats["backend"] == "python"
        assert stats["dim"] == 3
        assert stats["tile_count"] == 1

    def test_cosine_sim_identical(self):
        r = PolyglotReasoner(dim=3)
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert r._cosine_sim(a, a) == pytest.approx(1.0)

    def test_cosine_sim_orthogonal(self):
        r = PolyglotReasoner(dim=2)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert r._cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_sim_zero(self):
        r = PolyglotReasoner(dim=2)
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert r._cosine_sim(a, b) == 0.0
