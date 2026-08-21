"""Tests for weighted_router.py — Weighted routing with health-adjusted weights.

Run: python3 -m pytest tests/test_weighted_router.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.weighted_router import WeightedRouter


class TestWeightedRouter:
    def test_create(self):
        router = WeightedRouter()
        assert router.stats()["backends"] == 0

    def test_add_backend(self):
        router = WeightedRouter()
        assert router.add_backend("svc-a", weight=5) is True
        assert "svc-a" in router.backends()

    def test_add_duplicate(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=5)
        assert router.add_backend("svc-a", weight=3) is False

    def test_remove_backend(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=5)
        assert router.remove_backend("svc-a") is True
        assert router.remove_backend("missing") is False

    def test_select(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=10)
        router.add_backend("svc-b", weight=0)
        selected = router.select()
        assert selected == "svc-a"

    def test_select_weighted(self):
        router = WeightedRouter()
        router.add_backend("a", weight=10)
        router.add_backend("b", weight=0)
        counts = {"a": 0, "b": 0}
        for _ in range(100):
            s = router.select()
            if s:
                counts[s] += 1
        assert counts["a"] == 100
        assert counts["b"] == 0

    def test_select_none(self):
        router = WeightedRouter()
        assert router.select() is None

    def test_sticky_session(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=10)
        router.add_backend("svc-b", weight=10)
        selected = router.select(session_id="sess-1")
        assert router.select(session_id="sess-1") == selected

    def test_sticky_clear(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=10)
        router.select(session_id="sess-1")
        assert router.sticky_clear("sess-1") is True
        assert router.sticky_clear("missing") is False

    def test_update_health(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=10, health="healthy")
        assert router.update_health("svc-a", "degraded") is True
        assert router.get_backend("svc-a")["health"] == "degraded"
        assert router.update_health("missing", "healthy") is False

    def test_update_weight(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=5)
        assert router.update_weight("svc-a", 10) is True
        assert router.get_backend("svc-a")["weight"] == 10

    def test_health_adjusted_weights(self):
        router = WeightedRouter()
        router.add_backend("a", weight=10, health="healthy")
        router.add_backend("b", weight=10, health="degraded")
        weights = router.effective_weights()
        assert weights["a"] == 10.0
        assert weights["b"] == 5.0

    def test_critical_backend_zero_weight(self):
        router = WeightedRouter()
        router.add_backend("a", weight=10, health="critical")
        assert router.effective_weights()["a"] == 0.0

    def test_remove_backend_clears_sticky(self):
        router = WeightedRouter()
        router.add_backend("svc-a", weight=10)
        router.select(session_id="sess-1")
        router.remove_backend("svc-a")
        assert router.select(session_id="sess-1") is None

    def test_stats(self):
        router = WeightedRouter()
        router.add_backend("a", weight=5, health="healthy")
        router.add_backend("b", weight=3, health="degraded")
        router.select(session_id="sess-1")
        stats = router.stats()
        assert stats["backends"] == 2
        assert stats["sticky_sessions"] == 1
        assert stats["total_effective_weight"] == 6.5
        assert stats["health_distribution"] == {"healthy": 1, "degraded": 1}

    def test_repr(self):
        router = WeightedRouter()
        assert "WeightedRouter" in repr(router)
