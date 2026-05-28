"""Tests for dependency_graph.py — DAG dependency resolution.

Run: python3 -m pytest tests/test_dependency_graph.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.dependency_graph import DependencyGraph, CycleError


class TestDependencyGraph:
    def test_create(self):
        g = DependencyGraph()
        assert g.nodes() == []
        assert g.edge_count() == 0

    def test_add_node(self):
        g = DependencyGraph()
        g.add_node("a")
        assert g.has_node("a") is True

    def test_add_edge(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        assert g.has_node("a")
        assert g.has_node("b")
        assert g.dependencies("a") == {"b"}
        assert g.dependents("b") == {"a"}

    def test_self_dependency(self):
        g = DependencyGraph()
        with pytest.raises(CycleError):
            g.add_edge("a", "a")

    def test_topological_sort_linear(self):
        g = DependencyGraph()
        g.add_edge("c", "b")
        g.add_edge("b", "a")
        order = g.topological_sort()
        assert order == ["a", "b", "c"]

    def test_topological_sort_diamond(self):
        g = DependencyGraph()
        g.add_edge("c", "a")
        g.add_edge("c", "b")
        g.add_edge("b", "a")
        order = g.topological_sort()
        assert order[0] == "a"
        assert order[-1] == "c"

    def test_topological_sort_cycle(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        with pytest.raises(CycleError):
            g.topological_sort()

    def test_parallel_batches(self):
        g = DependencyGraph()
        g.add_edge("c", "a")
        g.add_edge("c", "b")
        batches = g.resolve_parallel_batches()
        assert batches[0] == ["a", "b"]
        assert batches[1] == ["c"]

    def test_parallel_batches_cycle(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(CycleError):
            g.resolve_parallel_batches()

    def test_find_cycle_exists(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        cycle = g.find_cycle()
        assert cycle is not None
        assert "a" in cycle and "b" in cycle and "c" in cycle

    def test_find_cycle_none(self):
        g = DependencyGraph()
        g.add_edge("b", "a")
        g.add_edge("c", "b")
        assert g.find_cycle() is None

    def test_remove_edge(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        assert g.remove_edge("a", "b") is True
        assert g.dependencies("a") == set()
        assert g.remove_edge("a", "b") is False

    def test_remove_node(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        g.add_edge("c", "b")
        g.remove_node("b")
        assert g.has_node("b") is False
        assert g.dependencies("a") == set()
        assert g.dependencies("c") == set()

    def test_is_leaf(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        assert g.is_leaf("b") is True
        assert g.is_leaf("a") is False

    def test_is_root(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        assert g.is_root("a") is True
        assert g.is_root("b") is False

    def test_repr(self):
        g = DependencyGraph()
        g.add_edge("a", "b")
        assert "DependencyGraph" in repr(g)
