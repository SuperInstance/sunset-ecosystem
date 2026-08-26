"""Tests for task_dependency_graph.py — Task DAG with cycle detection.

Run: python3 -m pytest tests/test_task_dependency_graph.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.task_dependency_graph import TaskDependencyGraph, CycleError


class TestTaskDependencyGraph:
    def test_create(self):
        g = TaskDependencyGraph()
        assert g.task_count() == 0

    def test_add_task_and_dep(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_dep("b", "a")
        assert g.deps("b") == ["a"]

    def test_execution_order(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_task("c")
        g.add_dep("b", "a")
        g.add_dep("c", "b")
        order = g.execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_cycle_detection(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_dep("a", "b")
        with pytest.raises(CycleError):
            g.add_dep("b", "a")

    def test_parallel_groups(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_task("c")
        g.add_dep("c", "a")
        g.add_dep("c", "b")
        groups = g.parallel_groups()
        assert len(groups) == 2
        assert set(groups[0]) == {"a", "b"}
        assert groups[1] == ["c"]

    def test_is_ready(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_dep("b", "a")
        assert g.is_ready("a") is True
        assert g.is_ready("b") is False

    def test_mark_done(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_dep("b", "a")
        g.mark_done("a")
        assert g.is_ready("b") is True

    def test_remove_task(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        g.add_task("b")
        g.add_dep("b", "a")
        g.remove_task("a")
        assert g.is_ready("b") is True

    def test_repr(self):
        g = TaskDependencyGraph()
        g.add_task("a")
        assert "TaskDependencyGraph" in repr(g)
