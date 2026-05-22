"""Tests for fleet-agent ↔ holonomy-consensus bridge.

Covers:
    - Graph construction (nodes, edges)
    - Cycle verification (consistent vs inconsistent)
    - H¹ cohomology computation (β₁)
    - Emergence detection (β₁ increase)
    - Bridge unified check report
    - Factory method from_fleet_edges
"""

from __future__ import annotations

import pytest

from nexus.holonomy_bridge import BridgeReport, HolonomyBridge
from swarm.holonomy_consensus import HolonomyConsensus


# ═══════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════

class TestGraphConstruction:
    def test_add_node(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("n1", state=1.0)
        assert bridge._consensus.node_count == 1

    def test_add_edge(self):
        bridge = HolonomyBridge()
        bridge.link("n1", "n2")
        assert bridge._consensus.edge_count == 1

    def test_remove_node_cascades(self):
        bridge = HolonomyBridge()
        bridge.link("n1", "n2")
        bridge.remove_fleet_node("n1")
        assert bridge._consensus.node_count == 1
        assert bridge._consensus.edge_count == 0

    def test_remove_edge(self):
        bridge = HolonomyBridge()
        bridge.link("n1", "n2")
        bridge.unlink("n1", "n2")
        assert bridge._consensus.edge_count == 0


# ═══════════════════════════════════════════════════════════════
# Cycle verification
# ═══════════════════════════════════════════════════════════════

class TestCycleVerification:
    def test_consistent_cycle(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a", state=0.0)
        bridge.add_fleet_node("b", state=0.0)
        bridge.add_fleet_node("c", state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")

        report = bridge.verify_cycle(["a", "b", "c"])
        assert report.consistent
        assert report.holonomy_error == pytest.approx(0.0)

    def test_inconsistent_cycle(self):
        bridge = HolonomyBridge(consistency_threshold=0.1)
        bridge.add_fleet_node("a", state=0.0)
        bridge.add_fleet_node("b", state=1.0)
        bridge.add_fleet_node("c", state=2.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")

        report = bridge.verify_cycle(["a", "b", "c"])
        assert not report.consistent
        assert report.holonomy_error > 0.1

    def test_cycle_with_missing_edge(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a")
        bridge.add_fleet_node("b")
        bridge.add_fleet_node("c")
        # Only a-b edge exists
        bridge.link("a", "b")

        report = bridge.verify_cycle(["a", "b", "c"])
        assert not report.consistent
        assert report.holonomy_error == float("inf")

    def test_short_cycle_is_trivially_consistent(self):
        bridge = HolonomyBridge()
        report = bridge.verify_cycle(["a", "b"])
        assert report.consistent


# ═══════════════════════════════════════════════════════════════
# H¹ cohomology
# ═══════════════════════════════════════════════════════════════

class TestH1Cohomology:
    def test_tree_has_beta_1_zero(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        snap = bridge.h1_snapshot()
        assert snap.betti_1 == 0

    def test_triangle_has_beta_1_one(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        snap = bridge.h1_snapshot()
        assert snap.betti_1 == 1

    def test_square_with_diagonal_has_beta_1_two(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "d")
        bridge.link("d", "a")
        bridge.link("a", "c")  # diagonal
        snap = bridge.h1_snapshot()
        # 5 edges, 4 nodes, 1 component → β₁ = 5 - 4 + 1 = 2
        assert snap.betti_1 == 2

    def test_disconnected_graph(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("c", "d")
        snap = bridge.h1_snapshot()
        # 2 edges, 4 nodes, 2 components → β₁ = 2 - 4 + 2 = 0
        assert snap.betti_1 == 0


# ═══════════════════════════════════════════════════════════════
# Emergence detection
# ═══════════════════════════════════════════════════════════════

class TestEmergenceDetection:
    def test_no_emergence_on_first_snapshot(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.h1_snapshot()
        assert bridge.detect_emergence() is None

    def test_emergence_when_cycle_appears(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.h1_snapshot()

        # Add edge that creates a cycle
        bridge.link("c", "a")
        bridge.h1_snapshot()

        event = bridge.detect_emergence()
        assert event is not None
        assert event.previous_betti_1 == 0
        assert event.current_betti_1 == 1
        assert len(event.new_cycles) == 1

    def test_no_emergence_when_stable(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        bridge.h1_snapshot()
        bridge.h1_snapshot()
        assert bridge.detect_emergence() is None


# ═══════════════════════════════════════════════════════════════
# Unified bridge check
# ═══════════════════════════════════════════════════════════════

class TestBridgeCheck:
    def test_check_returns_report(self):
        bridge = HolonomyBridge()
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.check()
        assert isinstance(report, BridgeReport)
        assert report.node_count == 3
        assert report.edge_count == 3
        assert report.betti_1 == 1
        assert report.cycles_verified == 1
        assert report.cycles_consistent == 1

    def test_factory_from_fleet_edges(self):
        edges = [("n1", "n2"), ("n2", "n3"), ("n3", "n1")]
        states = {"n1": 0.0, "n2": 0.0, "n3": 0.0}
        bridge = HolonomyBridge.from_fleet_edges(edges, node_states=states)
        assert bridge._consensus.node_count == 3
        assert bridge._consensus.edge_count == 3

    def test_inconsistent_cycle_in_report(self):
        bridge = HolonomyBridge(consistency_threshold=0.1)
        bridge.add_fleet_node("a", state=0.0)
        bridge.add_fleet_node("b", state=1.0)
        bridge.add_fleet_node("c", state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.check()
        assert report.cycles_verified == 1
        assert report.cycles_consistent == 0


# ═══════════════════════════════════════════════════════════════
# HolonomyConsensus direct tests
# ═══════════════════════════════════════════════════════════════

class TestHolonomyConsensus:
    def test_find_independent_cycles_triangle(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        cycles = hc._find_independent_cycles()
        assert len(cycles) == 1
        # Cycle should contain a, b, c in some order
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_find_independent_cycles_square(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "d")
        hc.add_edge("d", "a")
        cycles = hc._find_independent_cycles()
        # Square has one independent cycle
        assert len(cycles) == 1
