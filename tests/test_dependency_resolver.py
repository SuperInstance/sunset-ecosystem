"""Tests for dependency_resolver.py — Task dependency resolver.

Run: python3 -m pytest tests/test_dependency_resolver.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.dependency_resolver import DependencyResolver


class TestDependencyResolver:
    def test_create(self):
        resolver = DependencyResolver()
        assert resolver.stats()["tasks"] == 0

    def test_add_task(self):
        resolver = DependencyResolver()
        resolver.add_task("a", deps=["b"])
        assert resolver.tasks() == ["a"]
        assert resolver.dependencies("a") == ["b"]

    def test_remove_task(self):
        resolver = DependencyResolver()
        resolver.add_task("a", deps=["b"])
        assert resolver.remove_task("a") is True
        assert resolver.remove_task("missing") is False
        assert resolver.tasks() == []

    def test_resolve_simple(self):
        resolver = DependencyResolver()
        resolver.add_task("a", deps=["b"])
        resolver.add_task("b")
        order = resolver.resolve()
        assert order.index("b") < order.index("a")

    def test_resolve_chain(self):
        resolver = DependencyResolver()
        resolver.add_task("app", deps=["db"])
        resolver.add_task("db", deps=["network"])
        resolver.add_task("network")
        order = resolver.resolve()
        assert order == ["network", "db", "app"]

    def test_resolve_parallel(self):
        resolver = DependencyResolver()
        resolver.add_task("a")
        resolver.add_task("b")
        resolver.add_task("c", deps=["a", "b"])
        order = resolver.resolve()
        assert order.index("c") > order.index("a")
        assert order.index("c") > order.index("b")

    def test_detect_cycle(self):
        resolver = DependencyResolver()
        resolver.add_task("a", deps=["b"])
        resolver.add_task("b", deps=["a"])
        with pytest.raises(ValueError, match="cycle"):
            resolver.resolve()

    def test_parallel_groups(self):
        resolver = DependencyResolver()
        resolver.add_task("a")
        resolver.add_task("b")
        resolver.add_task("c", deps=["a", "b"])
        groups = resolver.parallel_groups()
        # All tasks should appear in groups
        flat = [t for g in groups for t in g]
        assert sorted(flat) == ["a", "b", "c"]

    def test_dependents(self):
        resolver = DependencyResolver()
        resolver.add_task("a")
        resolver.add_task("b", deps=["a"])
        resolver.add_task("c", deps=["a"])
        assert sorted(resolver.dependents("a")) == ["b", "c"]

    def test_stats(self):
        resolver = DependencyResolver()
        resolver.add_task("a", deps=["b"])
        resolver.add_task("b")
        stats = resolver.stats()
        assert stats["tasks"] == 2
        assert stats["dependencies"] == 1

    def test_repr(self):
        resolver = DependencyResolver()
        assert "DependencyResolver" in repr(resolver)
