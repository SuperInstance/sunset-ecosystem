"""Fleet-agent ↔ holonomy-consensus bridge.

Fleet-agent delegates cycle verification to holonomy-consensus instead of
reimplementing it. Emergence detection uses H¹ cohomology from
holonomy-consensus.

See Task 8 in docs/STRATEGIC-ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from swarm.holonomy_consensus import (
    CohomologySnapshot,
    CycleReport,
    EmergenceEvent,
    HolonomyConsensus,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BridgeReport:
    """Unified report from the holonomy bridge."""

    node_count: int
    edge_count: int
    betti_1: int
    cycles_verified: int
    cycles_consistent: int
    emergence_detected: bool
    emergence_description: str = ""
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "errors", self.errors or [])


class HolonomyBridge:
    """Adapter that wires fleet-agent to holonomy-consensus.

    Usage::

        bridge = HolonomyBridge()
        bridge.add_fleet_node("node-1", state=0.0)
        bridge.add_fleet_node("node-2", state=0.0)
        bridge.link("node-1", "node-2")

        report = bridge.check()
        if report.emergence_detected:
            logger.info("Emergence: %s", report.emergence_description)
    """

    def __init__(self, consistency_threshold: float = 1e-6) -> None:
        self._consensus = HolonomyConsensus(consistency_threshold=consistency_threshold)

    # ── fleet graph construction ────────────────────────────

    def add_fleet_node(self, node_id: str, state: float = 0.0) -> None:
        """Register a fleet node.

        Args:
            node_id: Unique node identifier (e.g. nexus node_id).
            state: Consensus state value (e.g. epoch timestamp hash).
        """
        self._consensus.add_node(node_id, state=state)

    def remove_fleet_node(self, node_id: str) -> None:
        """Remove a fleet node and all its links."""
        self._consensus.remove_node(node_id)

    def link(self, a: str, b: str) -> None:
        """Add a bidirectional link between two fleet nodes."""
        self._consensus.add_edge(a, b)

    def unlink(self, a: str, b: str) -> None:
        """Remove a link."""
        self._consensus.remove_edge(a, b)

    def update_state(self, node_id: str, state: float) -> None:
        """Update the consensus state of a fleet node."""
        self._consensus._node_state[node_id] = state

    # ── cycle verification ──────────────────────────────────

    def verify_cycle(self, cycle: list[str]) -> CycleReport:
        """Delegate cycle verification to holonomy-consensus.

        Args:
            cycle: Ordered list of node IDs forming a closed loop.

        Returns:
            :class:`CycleReport` with consistency verdict.
        """
        return self._consensus.verify_cycle(cycle)

    # ── H¹ cohomology & emergence ───────────────────────────

    def h1_snapshot(self) -> CohomologySnapshot:
        """Compute current H¹ cohomology state."""
        return self._consensus.h1_cohomology()

    def detect_emergence(self) -> Optional[EmergenceEvent]:
        """Detect topological emergence using H¹ cohomology."""
        return self._consensus.detect_emergence()

    # ── unified check ───────────────────────────────────────

    def check(self) -> BridgeReport:
        """Run full verification: cycles + cohomology + emergence.

        Returns:
            :class:`BridgeReport` summarizing fleet topology health.
        """
        errors: list[str] = []

        # Verify all independent cycles
        cycle_reports = self._consensus.verify_all_cycles()
        consistent_count = sum(1 for r in cycle_reports if r.consistent)

        # H¹ + emergence
        snap = self._consensus.h1_cohomology()
        emergence = self._consensus.detect_emergence()

        if emergence:
            logger.info("H¹ emergence detected: %s", emergence.description)

        return BridgeReport(
            node_count=snap.nodes,
            edge_count=snap.edges,
            betti_1=snap.betti_1,
            cycles_verified=len(cycle_reports),
            cycles_consistent=consistent_count,
            emergence_detected=emergence is not None,
            emergence_description=emergence.description if emergence else "",
            errors=errors,
        )

    # ── convenience factories ───────────────────────────────

    @classmethod
    def from_fleet_edges(
        cls,
        edges: list[tuple[str, str]],
        node_states: Optional[dict[str, float]] = None,
    ) -> "HolonomyBridge":
        """Build a bridge from a list of fleet edges.

        Args:
            edges: List of (node_a, node_b) tuples.
            node_states: Optional dict mapping node_id → state.

        Returns:
            Pre-populated :class:`HolonomyBridge`.
        """
        bridge = cls()
        node_states = node_states or {}
        for a, b in edges:
            bridge.add_fleet_node(a, state=node_states.get(a, 0.0))
            bridge.add_fleet_node(b, state=node_states.get(b, 0.0))
            bridge.link(a, b)
        return bridge
