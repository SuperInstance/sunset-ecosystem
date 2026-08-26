"""Tests for HolonomyBridge — fleet-agent ↔ holonomy-consensus adapter.

Covers graph construction, cycle verification, H¹ snapshot, emergence,
unified check, and convenience factories.
"""

import pytest

from nexus.holonomy_bridge import BridgeReport, HolonomyBridge


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_empty(self):
        bridge = HolonomyBridge()
        report = bridge.check()
        assert report.node_count == 0

    def test_add_fleet_node(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("node-1", state=1.0)
        report = bridge.check()
        assert report.node_count == 1
        assert report.edge_count == 0

    def test_link_and_unlink(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a")
        bridge.add_fleet_node("b")
        bridge.link("a", "b")
        report = bridge.check()
        assert report.edge_count == 1
        bridge.unlink("a", "b")
        report = bridge.check()
        assert report.edge_count == 0

    def test_remove_fleet_node(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a")
        bridge.add_fleet_node("b")
        bridge.link("a", "b")
        bridge.remove_fleet_node("a")
        report = bridge.check()
        assert report.node_count == 1
        assert report.edge_count == 0

    def test_update_state(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a", state=0.0)
        bridge.update_state("a", 5.0)
        # internal state updated; verify via cycle
        bridge.add_fleet_node("b", state=5.0)
        bridge.link("a", "b")
        bridge.add_fleet_node("c", state=0.0)
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.verify_cycle(["a", "b", "c", "a"])
        assert report.consistent is False  # drift 5 + 5 + 0 = 10, avg=3.33 > 1e-6


# ---------------------------------------------------------------------------
# Cycle verification
# ---------------------------------------------------------------------------


class TestVerifyCycle:
    def test_consistent(self):
        bridge = HolonomyBridge()
        for n in ["a", "b", "c"]:
            bridge.add_fleet_node(n, state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.verify_cycle(["a", "b", "c"])
        assert report.consistent is True

    def test_inconsistent(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a", state=0.0)
        bridge.add_fleet_node("b", state=1.0)
        bridge.add_fleet_node("c", state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.verify_cycle(["a", "b", "c", "a"])
        assert report.consistent is False


# ---------------------------------------------------------------------------
# H¹ snapshot & emergence
# ---------------------------------------------------------------------------


class TestH1AndEmergence:
    def test_snapshot(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a")
        bridge.add_fleet_node("b")
        bridge.link("a", "b")
        snap = bridge.h1_snapshot()
        assert snap.betti_1 == 0

    def test_emergence_none(self):
        bridge = HolonomyBridge()
        assert bridge.detect_emergence() is None

    def test_emergence_detected(self):
        bridge = HolonomyBridge()
        bridge.add_fleet_node("a")
        bridge.add_fleet_node("b")
        bridge.link("a", "b")
        bridge.h1_snapshot()
        bridge.add_fleet_node("c")
        bridge.link("b", "c")
        bridge.link("c", "a")
        bridge.h1_snapshot()
        event = bridge.detect_emergence()
        assert event is not None
        assert event.current_betti_1 == 1


# ---------------------------------------------------------------------------
# Unified check
# ---------------------------------------------------------------------------


class TestCheck:
    def test_empty(self):
        bridge = HolonomyBridge()
        report = bridge.check()
        assert report.node_count == 0
        assert report.edge_count == 0
        assert report.betti_1 == 0
        assert report.cycles_verified == 0
        assert report.emergence_detected is False

    def test_with_cycles(self):
        bridge = HolonomyBridge()
        for n in ["a", "b", "c", "d"]:
            bridge.add_fleet_node(n, state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        bridge.link("c", "d")
        bridge.link("d", "a")
        report = bridge.check()
        assert report.node_count == 4
        assert report.edge_count == 5
        assert report.betti_1 == 2
        assert report.cycles_verified == 2
        assert report.cycles_consistent == 2

    def test_bridge_report_errors_default(self):
        report = BridgeReport(
            node_count=0,
            edge_count=0,
            betti_1=0,
            cycles_verified=0,
            cycles_consistent=0,
            emergence_detected=False,
        )
        assert report.errors == []


# ---------------------------------------------------------------------------
# from_fleet_edges factory
# ---------------------------------------------------------------------------


class TestFromFleetEdges:
    def test_basic(self):
        edges = [("a", "b"), ("b", "c")]
        bridge = HolonomyBridge.from_fleet_edges(edges)
        report = bridge.check()
        assert report.node_count == 3
        assert report.edge_count == 2

    def test_with_states(self):
        edges = [("a", "b"), ("b", "c")]
        states = {"a": 1.0, "b": 2.0, "c": 3.0}
        bridge = HolonomyBridge.from_fleet_edges(edges, node_states=states)
        assert bridge._consensus._node_state["b"] == 2.0

    def test_duplicate_nodes(self):
        edges = [("a", "b"), ("a", "b")]
        bridge = HolonomyBridge.from_fleet_edges(edges)
        report = bridge.check()
        assert report.edge_count == 1
