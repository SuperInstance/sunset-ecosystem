"""Task dependency resolver with cycle detection.

Resolves task execution order from dependency declarations. Detects
cycles, supports parallel-ready grouping, and handles optional/required
dependencies. Used for fleet workflow orchestration, build pipelines,
and deployment ordering.

Usage:
    resolver = DependencyResolver()
    resolver.add_task("deploy-db", deps=["provision-network"])
    resolver.add_task("deploy-app", deps=["deploy-db"])
    order = resolver.resolve()
    assert order == ["provision-network", "deploy-db", "deploy-app"]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class DependencyResolver:
    """
    Task dependency resolver with cycle detection.
    """

    def __init__(self):
        self._tasks: Set[str] = set()
        self._deps: Dict[str, List[str]] = {}  # task -> [dependencies]
        self._optional: Set[str] = set()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def add_task(
        self, task: str, deps: Optional[List[str]] = None, optional: bool = False
    ) -> None:
        """
        Register a task with dependencies.

        :param task: Task identifier.
        :param deps: List of task identifiers this task depends on.
        :param optional: If True, missing dependencies are ignored.
        """
        self._tasks.add(task)
        self._deps[task] = list(deps or [])
        if optional:
            self._optional.add(task)

    def remove_task(self, task: str) -> bool:
        """Remove a task and its dependencies."""
        if task not in self._tasks:
            return False
        self._tasks.discard(task)
        del self._deps[task]
        self._optional.discard(task)
        # Remove from other tasks' deps
        for t in self._deps:
            if task in self._deps[t]:
                self._deps[t].remove(task)
        return True

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> List[str]:
        """
        Resolve execution order via topological sort.

        :returns: Ordered list of task identifiers.
        :raises ValueError: If a cycle is detected.
        """
        # Build in-degree map
        in_degree: Dict[str, int] = {t: 0 for t in self._tasks}
        for task, deps in self._deps.items():
            for dep in deps:
                if dep in self._tasks or dep not in self._optional:
                    in_degree[task] = in_degree.get(task, 0) + 1

        # Kahn's algorithm
        queue = [t for t, d in in_degree.items() if d == 0]
        result: List[str] = []
        visited = set()

        while queue:
            current = queue.pop(0)
            result.append(current)
            visited.add(current)

            # Find tasks that depend on current
            for task, deps in self._deps.items():
                if current in deps and task not in visited:
                    in_degree[task] -= 1
                    if in_degree[task] <= 0:
                        queue.append(task)

        if len(result) != len(self._tasks):
            # Detect cycle
            cycle = self._find_cycle()
            raise ValueError(f"Dependency cycle detected: {' -> '.join(cycle)}")

        return result

    def _find_cycle(self) -> List[str]:
        """Find a cycle in the dependency graph."""
        visited: Set[str] = set()
        path: List[str] = []
        path_set: Set[str] = set()

        def dfs(node: str) -> Optional[List[str]]:
            if node in path_set:
                idx = path.index(node)
                return path[idx:] + [node]
            if node in visited:
                return None
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for dep in self._deps.get(node, []):
                if dep in self._tasks:
                    cycle = dfs(dep)
                    if cycle:
                        return cycle
            path.pop()
            path_set.discard(node)
            return None

        for task in self._tasks:
            cycle = dfs(task)
            if cycle:
                return cycle
        return []

    def parallel_groups(self) -> List[List[str]]:
        """
        Group tasks by execution wave (parallel-ready).

        :returns: List of task groups, each group can execute in parallel.
        """
        order = self.resolve()
        # Simple grouping: tasks with no deps between them in same group
        groups: List[List[str]] = []
        current_group: List[str] = []
        completed: Set[str] = set()

        for task in order:
            deps = set(self._deps.get(task, []))
            if deps.issubset(completed):
                current_group.append(task)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [task]
            completed.add(task)

        if current_group:
            groups.append(current_group)

        return groups

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def tasks(self) -> List[str]:
        return list(self._tasks)

    def dependencies(self, task: str) -> List[str]:
        return list(self._deps.get(task, []))

    def dependents(self, task: str) -> List[str]:
        """Get tasks that depend on this task."""
        return [t for t, deps in self._deps.items() if task in deps]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "tasks": len(self._tasks),
            "dependencies": sum(len(d) for d in self._deps.values()),
        }

    def __repr__(self) -> str:
        return f"<DependencyResolver tasks={len(self._tasks)}>"
