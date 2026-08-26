"""Tests for Penrose lattice agent distribution — aperiodic diversity guarantee.

Covers PHI/GOLDEN_ANGLE constants, PenrosePosition, assign_positions,
compute_overlap, and minimum_overlap.
"""

import math

import pytest

from swarm.penrose import (
    GOLDEN_ANGLE,
    PHI,
    PenrosePosition,
    assign_positions,
    compute_overlap,
    minimum_overlap,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_phi_value(self):
        assert PHI == pytest.approx((1 + math.sqrt(5)) / 2)
        assert PHI == pytest.approx(1.618, abs=0.001)

    def test_golden_angle(self):
        assert GOLDEN_ANGLE == pytest.approx(2 * math.pi / (PHI * PHI))
        # Should be roughly 137.5 degrees in radians
        assert GOLDEN_ANGLE == pytest.approx(math.radians(137.5), abs=0.01)


# ---------------------------------------------------------------------------
# PenrosePosition
# ---------------------------------------------------------------------------


class TestPenrosePosition:
    def test_init(self):
        p = PenrosePosition(agent_id="a1", x=1.0, y=2.0, ring=0, angle=0.5)
        assert p.agent_id == "a1"
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.ring == 0
        assert p.angle == 0.5

    def test_repr(self):
        p = PenrosePosition(agent_id="a1", x=1.234, y=5.678, ring=2, angle=1.0)
        r = repr(p)
        assert "a1" in r
        assert "ring=2" in r

    def test_distance_to_same(self):
        p = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        assert p.distance_to(p) == 0.0

    def test_distance_to_other(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=3.0, y=4.0, ring=0, angle=0.0)
        assert p1.distance_to(p2) == 5.0


# ---------------------------------------------------------------------------
# assign_positions
# ---------------------------------------------------------------------------


class TestAssignPositions:
    def test_empty(self):
        assert assign_positions([]) == []

    def test_single(self):
        pos = assign_positions(["a1"])
        assert len(pos) == 1
        assert pos[0].agent_id == "a1"
        assert pos[0].x == pytest.approx(1.0)
        assert pos[0].y == pytest.approx(0.0)

    def test_multiple_unique(self):
        agents = [f"a{i}" for i in range(10)]
        pos = assign_positions(agents)
        assert len(pos) == 10
        # All unique positions
        coords = [(p.x, p.y) for p in pos]
        assert len(set(coords)) == 10

    def test_ring_increases(self):
        pos = assign_positions(["a" + str(i) for i in range(100)])
        rings = {p.ring for p in pos}
        assert max(rings) > 1
        # First agent gets ring = int(sqrt(1)) = 1
        assert pos[0].ring == 1

    def test_angle_wraps(self):
        pos = assign_positions(["a1", "a2", "a3"])
        for p in pos:
            assert 0 <= p.angle < 2 * math.pi


# ---------------------------------------------------------------------------
# compute_overlap
# ---------------------------------------------------------------------------


class TestComputeOverlap:
    def test_same_position(self):
        p = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        assert compute_overlap(p, p, radius=1.0) == 1.0

    def test_far_apart(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=10.0, y=0.0, ring=0, angle=0.0)
        assert compute_overlap(p1, p2, radius=1.0) == 0.0

    def test_halfway(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=1.9, y=0.0, ring=0, angle=0.0)
        # distance = 1.9, max_dist = 4, overlap = 1 - 1.9/4 = 0.525
        assert compute_overlap(p1, p2, radius=2.0) == pytest.approx(0.525)

    def test_beyond_threshold(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=2.1, y=0.0, ring=0, angle=0.0)
        # distance = 2.1, max_dist = 2, overlap = 0
        assert compute_overlap(p1, p2, radius=1.0) == 0.0


# ---------------------------------------------------------------------------
# minimum_overlap
# ---------------------------------------------------------------------------


class TestMinimumOverlap:
    def test_empty(self):
        assert minimum_overlap([]) == 0.0

    def test_single(self):
        p = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        assert minimum_overlap([p]) == 0.0

    def test_two_far(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=100.0, y=0.0, ring=0, angle=0.0)
        assert minimum_overlap([p1, p2], radius=1.0) == 0.0

    def test_two_close(self):
        p1 = PenrosePosition(agent_id="a1", x=0.0, y=0.0, ring=0, angle=0.0)
        p2 = PenrosePosition(agent_id="a2", x=1.0, y=0.0, ring=0, angle=0.0)
        ov = compute_overlap(p1, p2, radius=1.0)
        assert minimum_overlap([p1, p2], radius=1.0) == pytest.approx(ov)
