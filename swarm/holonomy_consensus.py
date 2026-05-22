"""Holonomy-consensus engine — cycle verification and H¹ cohomology.

Implements:
    - Graph cycle space computation (β₁ = E - V + 1 for connected graphs)
    - Holonomy consistency checks around cycles
    - Emergence detection via persistent β₁ tracking

See Task 8 in docs/STRATEGIC-ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── data structures ───────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CycleReport:
    """Result of verifying one cycle."""

    cycle: tuple[str, ...]
    consistent: bool
    holonomy_error: float
    threshold: float


@dataclass
class CohomologySnapshot:
    """H¹ cohomology state at a point in time."""

    timestamp: float
    nodes: int
    edges: int
    betti_1: int
    independent_cycles: list[list[str]] = field(default_factory=list)


@dataclass
class EmergenceEvent:
    """An emergence event — a new topological feature detected."""

    timestamp: float
    previous_betti_1: int
    current_betti_1: int
    new_cycles: list[list[str]]
    description: str


# ── core math ─────────────────────────────────────────────────

class HolonomyConsensus:
    """Consensus engine for fleet-wide cycle verification.

    Maintains an undirected graph of fleet nodes and verifies that
    cycles are holonomically consistent (state agrees around the loop).
    """

    def __init__(self, consistency_threshold: float = 1e-6) -> None:
        self._adjacency: dict[str, set[str]] = {}
        self._node_state: dict[str, float] = {}
        self._threshold = consistency_threshold
        self._history: list[CohomologySnapshot] = []

    # ── graph construction ──────────────────────────────────

    def add_node(self, node_id: str, state: float = 0.0) -> None:
        """Register a fleet node with its consensus state."""
        if node_id not in self._adjacency:
            self._adjacency[node_id] = set()
            self._node_state[node_id] = state

    def add_edge(self, a: str, b: str) -> None:
        """Add an undirected edge between two fleet nodes."""
        self.add_node(a)
        self.add_node(b)
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    def remove_edge(self, a: str, b: str) -> None:
        """Remove an edge."""
        self._adjacency[a].discard(b)
        self._adjacency[b].discard(a)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        for neighbor in list(self._adjacency.get(node_id, [])):
            self._adjacency[neighbor].discard(node_id)
        self._adjacency.pop(node_id, None)
        self._node_state.pop(node_id, None)

    @property
    def node_count(self) -> int:
        return len(self._adjacency)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self._adjacency.values()) // 2

    # ── cycle verification ──────────────────────────────────

    def verify_cycle(self, cycle: list[str]) -> CycleReport:
        """Verify holonomic consistency around a cycle.

        A cycle is consistent if the cumulative state change around
        the loop is below the threshold. We model this as the sum
        of absolute state differences along the cycle.

        Args:
            cycle: Ordered list of node IDs forming a closed loop.
                The last node must connect back to the first.

        Returns:
            :class:`CycleReport` with consistency verdict.
        """
        if len(cycle) < 3:
            return CycleReport(
                cycle=tuple(cycle),
                consistent=True,
                holonomy_error=0.0,
                threshold=self._threshold,
            )

        total_drift = 0.0
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            if b not in self._adjacency.get(a, set()):
                return CycleReport(
                    cycle=tuple(cycle),
                    consistent=False,
                    holonomy_error=float("inf"),
                    threshold=self._threshold,
                )
            drift = abs(self._node_state.get(a, 0.0) - self._node_state.get(b, 0.0))
            total_drift += drift

        # Normalize by cycle length
        avg_drift = total_drift / len(cycle)
        consistent = avg_drift <= self._threshold

        return CycleReport(
            cycle=tuple(cycle),
            consistent=consistent,
            holonomy_error=avg_drift,
            threshold=self._threshold,
        )

    def verify_all_cycles(self) -> list[CycleReport]:
        """Find and verify all independent cycles in the graph."""
        cycles = self._find_independent_cycles()
        return [self.verify_cycle(c) for c in cycles]

    # ── H¹ cohomology ───────────────────────────────────────

    def h1_cohomology(self) -> CohomologySnapshot:
        """Compute H¹ cohomology (Betti-1) of the current graph.

        For a connected graph:
            β₀ = 1  (one connected component)
            β₁ = E - V + β₀ = E - V + 1

        For a disconnected graph with C components:
            β₁ = E - V + C

        Returns:
            :class:`CohomologySnapshot` with current topological state.
        """
        import time

        v = self.node_count
        e = self.edge_count
        c = self._count_components()
        betti_1 = max(0, e - v + c)
        cycles = self._find_independent_cycles()

        snap = CohomologySnapshot(
            timestamp=time.time(),
            nodes=v,
            edges=e,
            betti_1=betti_1,
            independent_cycles=cycles,
        )
        self._history.append(snap)
        return snap

    def detect_emergence(self) -> Optional[EmergenceEvent]:
        """Detect if a new topological feature (hole) has emerged.

        Compares the current β₁ to the previous snapshot. If β₁
        increased, new independent cycles have appeared.

        Returns:
            :class:`EmergenceEvent` if emergence detected, else None.
        """
        import time

        if len(self._history) < 2:
            return None

        prev = self._history[-2]
        curr = self._history[-1]

        if curr.betti_1 > prev.betti_1:
            # Identify truly new cycles (heuristic: cycles not in prev)
            prev_cycles = {tuple(c) for c in prev.independent_cycles}
            new_cycles = [c for c in curr.independent_cycles if tuple(c) not in prev_cycles]

            return EmergenceEvent(
                timestamp=time.time(),
                previous_betti_1=prev.betti_1,
                current_betti_1=curr.betti_1,
                new_cycles=new_cycles,
                description=(
                    f"H¹ emergence: β₁ {prev.betti_1} → {curr.betti_1} "
                    f"({len(new_cycles)} new independent cycle(s))"
                ),
            )
        return None

    # ── internals ───────────────────────────────────────────

    def _count_components(self) -> int:
        """Count connected components via DFS."""
        visited: set[str] = set()
        components = 0
        for node in self._adjacency:
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    stack.extend(self._adjacency[n] - visited)
        return components

    def _find_independent_cycles(self) -> list[list[str]]:
        """Find a basis for the cycle space.

        Uses a simple DFS-based approach:
            1. Build a spanning tree via DFS.
            2. Every non-tree edge creates one independent cycle.

        Returns:
            List of cycles (each cycle is a list of node IDs).
        """
        if not self._adjacency:
            return []

        visited: set[str] = set()
        parent: dict[str, Optional[str]] = {}
        tree_edges: set[tuple[str, str]] = set()

        # Build spanning forest
        for start in self._adjacency:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            parent[start] = None
            while stack:
                node = stack.pop()
                for neighbor in self._adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        tree_edges.add(self._norm_edge(node, neighbor))
                        stack.append(neighbor)

        # Find non-tree edges and build cycles
        cycles: list[list[str]] = []
        seen_cycles: set[tuple[str, ...]] = set()

        for node in self._adjacency:
            for neighbor in self._adjacency[node]:
                edge = self._norm_edge(node, neighbor)
                if edge not in tree_edges:
                    # This non-tree edge forms a cycle with tree path
                    cycle = self._tree_path(node, neighbor, parent)
                    if cycle:
                        key = tuple(cycle)
                        rev = tuple(reversed(cycle))
                        if key not in seen_cycles and rev not in seen_cycles:
                            seen_cycles.add(key)
                            cycles.append(cycle)

        return cycles

    @staticmethod
    def _norm_edge(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _tree_path(self, start: str, end: str, parent: dict[str, Optional[str]]) -> list[str]:
        """Return the unique tree path from start to end."""
        # Build path from start to root
        path_start: list[str] = []
        node: Optional[str] = start
        while node is not None:
            path_start.append(node)
            node = parent.get(node)

        # Build path from end to root
        path_end: list[str] = []
        node = end
        while node is not None:
            path_end.append(node)
            node = parent.get(node)

        # Find LCA
        lca_idx_start = -1
        lca_idx_end = -1
        for i, n in enumerate(path_start):
            if n in path_end:
                lca_idx_start = i
                lca_idx_end = path_end.index(n)
                break

        if lca_idx_start < 0 or lca_idx_end < 0:
            return []

        # path: start → ... → LCA → ... → end
        cycle = path_start[:lca_idx_start + 1] + list(reversed(path_end[:lca_idx_end]))
        return cycle
