"""Tests for cache_warming.py — Preemptive cache warming.

Run: python3 -m pytest tests/test_cache_warming.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.cache_warming import CacheWarmer


class TestCacheWarmer:
    def test_create(self):
        w = CacheWarmer(max_workers=2)
        assert w.stats()["warmed"] == 0

    def test_warm_single(self):
        w = CacheWarmer()
        results = []
        w.warm(["a"], lambda key: results.append(key))
        assert results == ["a"]
        assert w.stats()["warmed"] == 1

    def test_warm_batch(self):
        w = CacheWarmer()
        warmed = []
        w.warm(["a", "b", "c"], lambda key: warmed.append(key))
        assert sorted(warmed) == ["a", "b", "c"]

    def test_prefetch(self):
        w = CacheWarmer()
        warmed = []
        w.prefetch(["x", "y"], loader=lambda key: warmed.append(key))
        assert sorted(warmed) == ["x", "y"]

    def test_schedule_and_run(self):
        w = CacheWarmer()
        warmed = []
        w.schedule("a", lambda key: warmed.append(key))
        w.run_scheduled()
        assert "a" in warmed

    def test_clear_queue(self):
        w = CacheWarmer()
        w.schedule("a", lambda k: None)
        w.clear_queue()
        assert w.stats()["queued"] == 0

    def test_error_not_fatal(self):
        w = CacheWarmer()
        warmed = []
        w.warm(
            ["a", "b"],
            lambda key: (
                (_ for _ in ()).throw(ValueError("boom"))
                if key == "a"
                else warmed.append(key)
            ),
        )
        assert warmed == ["b"]
        assert w.stats()["errors"] == 1

    def test_empty_keys(self):
        w = CacheWarmer()
        w.warm([], lambda k: None)
        assert w.stats()["warmed"] == 0

    def test_repr(self):
        w = CacheWarmer()
        assert "CacheWarmer" in repr(w)
