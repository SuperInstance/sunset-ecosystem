"""Tests for HolonomyConsensus — cycle verification and H¹ cohomology.

Covers graph construction, cycle verification, Betti-1 computation,
emergence detection, and independent cycle basis.
"""

import math

import pytest

from swarm.holonomy_consensus import (
    CycleReport,
    CohomologySnapshot,
    EmergenceEvent,
    HolonomyConsensus,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_empty(self):
        hc = HolonomyConsensus()
        assert hc.node_count == 0
        assert hc.edge_count == 0

    def test_add_node(self):
        hc = HolonomyConsensus()
        hc.add_node("a", state=1.0)
        assert hc.node_count == 1

    def test_add_edge(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        assert hc.node_count == 2
        assert hc.edge_count == 1

    def test_add_duplicate_edge(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("a", "b")
        assert hc.edge_count == 1

    def test_remove_edge(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.remove_edge("a", "b")
        assert hc.edge_count == 0

    def test_remove_node(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("a", "c")
        hc.remove_node("a")
        assert hc.node_count == 2
        assert hc.edge_count == 0

    def test_add_node_idempotent(self):
        hc = HolonomyConsensus()
        hc.add_node("a")
        hc.add_node("a", state=5.0)
        assert hc._node_state["a"] == 0.0  # first add wins


# ---------------------------------------------------------------------------
# Cycle verification
# ---------------------------------------------------------------------------

class TestVerifyCycle:
    def test_short_cycle(self):
        hc = HolonomyConsensus()
        report = hc.verify_cycle(["a", "b"])
        assert report.consistent is True
        assert report.holonomy_error == 0.0

    def test_single_node(self):
        hc = HolonomyConsensus()
        report = hc.verify_cycle(["a"])
        assert report.consistent is True
        assert report.holonomy_error == 0.0

    def test_consistent_cycle(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        report = hc.verify_cycle(["a", "b", "c"])
        assert report.consistent is True
        assert report.holonomy_error == 0.0

    def test_inconsistent_cycle(self):
        hc = HolonomyConsensus()
        hc.add_node("a", state=0.0)
        hc.add_node("b", state=1.0)
        hc.add_node("c", state=0.0)
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        report = hc.verify_cycle(["a", "b", "c"])
        # avg drift = (1.0 + 1.0 + 0.0) / 3 = 0.666... > 1e-6
        assert report.consistent is False
        assert report.holonomy_error == pytest.approx(2.0 / 3, abs=1e-6)

    def test_missing_edge(self):
        hc = HolonomyConsensus()
        hc.add_node("a")
        hc.add_node("b")
        hc.add_node("c")
        report = hc.verify_cycle(["a", "b", "c"])
        assert report.consistent is False
        assert report.holonomy_error == float("inf")


# ---------------------------------------------------------------------------
# H¹ cohomology
# ---------------------------------------------------------------------------

class TestH1Cohomology:
    def test_empty_graph(self):
        hc = HolonomyConsensus()
        snap = hc.h1_cohomology()
        assert snap.nodes == 0
        assert snap.edges == 0
        assert snap.betti_1 == 0

    def test_tree(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        snap = hc.h1_cohomology()
        assert snap.nodes == 3
        assert snap.edges == 2
        assert snap.betti_1 == 0  # tree, no cycles

    def test_single_cycle(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        snap = hc.h1_cohomology()
        assert snap.betti_1 == 1
        assert len(snap.independent_cycles) == 1

    def test_two_cycles(self):
        hc = HolonomyConsensus()
        # cycle 1: a-b-c-a
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        # cycle 2: a-c-d-a
        hc.add_edge("c", "d")
        hc.add_edge("d", "a")
        snap = hc.h1_cohomology()
        assert snap.nodes == 4
        assert snap.edges == 5
        assert snap.betti_1 == 2

    def test_disconnected(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")  # one component with 1 cycle
        hc.add_edge("x", "y")  # second component (tree)
        snap = hc.h1_cohomology()
        assert snap.nodes == 5
        assert snap.edges == 4
        # β₁ = E - V + C = 4 - 5 + 2 = 1
        assert snap.betti_1 == 1


# ---------------------------------------------------------------------------
# Emergence detection
# ---------------------------------------------------------------------------

class TestDetectEmergence:
    def test_no_history(self):
        hc = HolonomyConsensus()
        assert hc.detect_emergence() is None

    def test_single_snapshot(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.h1_cohomology()
        assert hc.detect_emergence() is None

    def test_emergence_detected(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.h1_cohomology()
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        snap = hc.h1_cohomology()
        event = hc.detect_emergence()
        assert event is not None
        assert event.previous_betti_1 == 0
        assert event.current_betti_1 == 1
        assert len(event.new_cycles) >= 1

    def test_no_emergence_when_stable(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.h1_cohomology()
        hc.h1_cohomology()  # duplicate snapshot
        assert hc.detect_emergence() is None

    def test_no_emergence_when_betti_decreases(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        hc.h1_cohomology()
        hc.remove_edge("c", "a")
        hc.h1_cohomology()
        assert hc.detect_emergence() is None


# ---------------------------------------------------------------------------
# verify_all_cycles
# ---------------------------------------------------------------------------

class TestVerifyAllCycles:
    def test_empty(self):
        hc = HolonomyConsensus()
        reports = hc.verify_all_cycles()
        assert reports == []

    def test_triangle(self):
        hc = HolonomyConsensus()
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        reports = hc.verify_all_cycles()
        assert len(reports) == 1
        assert reports[0].consistent is True

    def test_figure_eight(self):
        hc = HolonomyConsensus()
        # cycle 1: a-b-c-a
        hc.add_edge("a", "b")
        hc.add_edge("b", "c")
        hc.add_edge("c", "a")
        # cycle 2: a-c-d-a
        hc.add_edge("c", "d")
        hc.add_edge("d", "a")
        reports = hc.verify_all_cycles()
        assert len(reports) == 2


# ---------------------------------------------------------------------------
# CycleReport
# ---------------------------------------------------------------------------

class TestCycleReport:
    def test_immutable(self):
        report = CycleReport(
            cycle=("a", "b", "a"),
            consistent=True,
            holonomy_error=0.0,
            threshold=1e-6,
        )
        with pytest.raises(AttributeError):
            report.consistent = False
