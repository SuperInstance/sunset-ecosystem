"""DAG dependency resolution with topological sort and cycle detection.

Resolves task dependencies, service startup order, and config loading
sequences. Used by the fleet conductor for ordered initialization.

Usage:
    g = DependencyGraph()
    g.add_edge("conductor", "health-monitor")
    g.add_edge("breeder", "conductor")
    order = g.topological_sort()
    # -> ["health-monitor", "conductor", "breeder"]
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CycleError(Exception):
    pass


class DependencyGraph:
    """
    Directed acyclic graph for dependency resolution.

    Nodes are strings (task names, service names, etc.).
    Edge A -> B means "A depends on B" (B must run before A).
    """

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependencies
        self._reverse: Dict[str, Set[str]] = defaultdict(set)  # node -> dependents

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def add_edge(self, node: str, dependency: str) -> None:
        """Add dependency edge: *node* depends on *dependency*."""
        self._nodes.add(node)
        self._nodes.add(dependency)
        if node == dependency:
            raise CycleError(f"Self-dependency: {node}")
        self._edges[node].add(dependency)
        self._reverse[dependency].add(node)

    def remove_edge(self, node: str, dependency: str) -> bool:
        if dependency in self._edges[node]:
            self._edges[node].discard(dependency)
            self._reverse[dependency].discard(node)
            return True
        return False

    def remove_node(self, node: str) -> None:
        self._nodes.discard(node)
        for dep in list(self._edges[node]):
            self.remove_edge(node, dep)
        for dependent in list(self._reverse[node]):
            self.remove_edge(dependent, node)
        self._edges.pop(node, None)
        self._reverse.pop(node, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_node(self, node: str) -> bool:
        return node in self._nodes

    def dependencies(self, node: str) -> Set[str]:
        return set(self._edges.get(node, set()))

    def dependents(self, node: str) -> Set[str]:
        return set(self._reverse.get(node, set()))

    def is_leaf(self, node: str) -> bool:
        return len(self._edges.get(node, set())) == 0

    def is_root(self, node: str) -> bool:
        return len(self._reverse.get(node, set())) == 0

    def nodes(self) -> List[str]:
        return sorted(self._nodes)

    def edge_count(self) -> int:
        return sum(len(deps) for deps in self._edges.values())

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm. Raises CycleError if cycles detected."""
        in_degree: Dict[str, int] = {node: 0 for node in self._nodes}
        for node, deps in self._edges.items():
            for dep in deps:
                in_degree[node] = in_degree.get(node, 0) + 1

        queue = deque([n for n in self._nodes if in_degree[n] == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in self._reverse.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            raise CycleError("Cycle detected in dependency graph")
        return result

    def resolve_parallel_batches(self) -> List[List[str]]:
        """
        Group nodes into parallel-executable batches.

        Each batch contains nodes whose dependencies are all in previous batches.
        """
        in_degree: Dict[str, int] = {node: 0 for node in self._nodes}
        for node, deps in self._edges.items():
            for dep in deps:
                in_degree[node] = in_degree.get(node, 0) + 1

        queue = deque([n for n in self._nodes if in_degree[n] == 0])
        batches: List[List[str]] = []
        current_batch: List[str] = []
        next_queue: deque = deque()

        while queue:
            node = queue.popleft()
            current_batch.append(node)
            for dependent in self._reverse.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_queue.append(dependent)

            if not queue:
                if current_batch:
                    batches.append(current_batch)
                queue = next_queue
                next_queue = deque()
                current_batch = []

        if sum(len(b) for b in batches) != len(self._nodes):
            raise CycleError("Cycle detected in dependency graph")
        return batches

    def find_cycle(self) -> Optional[List[str]]:
        """Return a cycle if one exists, else None. DFS-based."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        parent: Dict[str, Optional[str]] = {}

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            for dep in self._edges.get(node, set()):
                if dep not in visited:
                    parent[dep] = node
                    cycle = dfs(dep)
                    if cycle is not None:
                        return cycle
                elif dep in rec_stack:
                    # Reconstruct cycle
                    cycle = [dep]
                    cur = node
                    while cur != dep:
                        cycle.append(cur)
                        cur = parent.get(cur)
                        if cur is None:
                            break
                    cycle.append(dep)
                    return list(reversed(cycle))
            rec_stack.remove(node)
            return None

        for node in self._nodes:
            if node not in visited:
                parent[node] = None
                cycle = dfs(node)
                if cycle is not None:
                    return cycle
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<DependencyGraph nodes={len(self._nodes)} edges={self.edge_count()}>"
