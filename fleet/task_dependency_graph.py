"""Task dependency graph with cycle detection and parallel groups.

Models tasks as a DAG with dependencies. Supports topological ordering,
parallel group detection, and cycle prevention. Used for fleet job
scheduling and workflow execution.

Usage:
    g = TaskDependencyGraph()
    g.add_task("compile")
    g.add_task("test", deps=["compile"])
    g.add_task("deploy", deps=["test"])
    order = g.execution_order()
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set


class CycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""
    pass


class TaskDependencyGraph:
    """
    Directed acyclic graph for task dependencies.
    """

    def __init__(self):
        self._tasks: Set[str] = set()
        self._deps: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        self._done: Set[str] = set()

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(self, task: str, deps: Optional[List[str]] = None) -> None:
        """Add a task with optional dependencies."""
        self._tasks.add(task)
        if deps:
            for dep in deps:
                self.add_dep(task, dep)

    def add_dep(self, task: str, dep: str) -> None:
        """Add a dependency edge (task depends on dep)."""
        if task == dep:
            raise CycleError(f"Self-dependency: {task}")
        self._tasks.add(task)
        self._tasks.add(dep)
        self._deps[task].add(dep)
        self._reverse[dep].add(task)
        if self._has_cycle(task):
            self._deps[task].discard(dep)
            self._reverse[dep].discard(task)
            raise CycleError(f"Cycle detected: {task} -> {dep}")

    def remove_task(self, task: str) -> bool:
        """Remove a task and its edges."""
        if task not in self._tasks:
            return False
        self._tasks.discard(task)
        for dep in list(self._deps.get(task, [])):
            self._reverse[dep].discard(task)
        self._deps.pop(task, None)
        for dependent in list(self._reverse.get(task, [])):
            self._deps[dependent].discard(task)
        self._reverse.pop(task, None)
        self._done.discard(task)
        return True

    def deps(self, task: str) -> List[str]:
        """Get direct dependencies of a task."""
        return sorted(self._deps.get(task, set()))

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def execution_order(self) -> List[str]:
        """Return topological sort (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {t: 0 for t in self._tasks}
        for task, deps in self._deps.items():
            for dep in deps:
                in_degree[task] += 1
        queue = deque([t for t, d in in_degree.items() if d == 0])
        order: List[str] = []
        while queue:
            task = queue.popleft()
            order.append(task)
            for dependent in self._reverse.get(task, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        return order

    def parallel_groups(self) -> List[List[str]]:
        """
        Return groups of tasks that can run in parallel.

        Each group contains tasks with all dependencies satisfied
        by previous groups.
        """
        in_degree: Dict[str, int] = {t: 0 for t in self._tasks}
        for task, deps in self._deps.items():
            for dep in deps:
                in_degree[task] += 1
        remaining = set(self._tasks)
        groups: List[List[str]] = []
        while remaining:
            group = [t for t in remaining if in_degree[t] == 0]
            if not group:
                break
            groups.append(group)
            for task in group:
                remaining.discard(task)
                for dependent in self._reverse.get(task, set()):
                    in_degree[dependent] -= 1
        return groups

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def is_ready(self, task: str) -> bool:
        """Check if all dependencies of a task are done."""
        return all(dep in self._done for dep in self._deps.get(task, set()))

    def mark_done(self, task: str) -> None:
        """Mark a task as completed."""
        self._done.add(task)

    def reset_done(self) -> None:
        """Clear done state."""
        self._done.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def task_count(self) -> int:
        return len(self._tasks)

    def _has_cycle(self, start: str) -> bool:
        """DFS cycle detection."""
        visited: Set[str] = set()
        stack: Set[str] = set()
        def visit(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for dep in self._deps.get(node, set()):
                if visit(dep):
                    return True
            stack.discard(node)
            return False
        return visit(start)

    def __repr__(self) -> str:
        return f"<TaskDependencyGraph tasks={len(self._tasks)}>"
