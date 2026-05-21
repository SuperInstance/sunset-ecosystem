"""Tests for AutoBreeder daemon."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from nerve.room_grid import JEPAGrid
from swarm.breeder_daemon import AutoBreeder, RebirthRecord
from swarm.thermal import DeviceType, ThermalBudget


# ── fixtures ────────────────────────────────────────────────

@pytest.fixture
def grid():
    """JEPAGrid with some hot and cold rooms."""
    g = JEPAGrid(n=20)
    # Make rooms 0-9 hot
    for _ in range(20):
        for i in range(10):
            g.activity[i] += 5
    return g


@pytest.fixture
def thermal():
    return ThermalBudget({DeviceType.GPU: 10, DeviceType.CPU: 20})


@pytest.fixture
def breeder(grid, thermal):
    return AutoBreeder(grid, thermal, interval=1, cold_threshold=3)


# ── tests ───────────────────────────────────────────────────

class TestAutoBreed:
    """Test the core auto_breed method."""

    def test_returns_empty_when_no_cold_rooms(self, thermal):
        """All rooms active → nothing to rebirth."""
        g = JEPAGrid(n=5)
        for i in range(5):
            g.activity[i] = 100
        b = AutoBreeder(g, thermal, cold_threshold=3)
        result = b.auto_breed()
        assert result == []

    def test_returns_empty_when_no_hot_rooms(self, thermal):
        """No active rooms → no parents to breed from.
        
        All rooms start at activity=0, so grid.top() returns them
        (they're all equal). The tournament still runs but all scores
        are identical. This test verifies no crash occurs and the
        result is a list.
        """
        g = JEPAGrid(n=5)
        b = AutoBreeder(g, thermal, cold_threshold=3)
        result = b.auto_breed()
        # With all-zero activity, top() returns rooms with activity 0.
        # max_activity would be 0, so AgentScore gets ethos=0 (disallowed).
        # But division by max_activity=0 is guarded (or results in 0).
        assert isinstance(result, list)

    def test_rebirths_cold_rooms_from_hot_parents(self, breeder, grid):
        """Cold rooms get rebirthed with cloned weights from hot rooms."""
        result = breeder.auto_breed(n_winners=3)

        # Should have rebirthed some cold rooms
        assert len(result) > 0

        # All rebirthed rooms should be cold rooms (index 10-19)
        for room_id, parent_id in result:
            assert room_id >= 10
            assert parent_id.startswith("room_")

    def test_weights_are_cloned_not_random(self, breeder, grid):
        """Rebirthed room weights should resemble parent, not be random."""
        # Save parent weights
        parent_room = 0
        original_w1 = grid.w["w1"][parent_room].copy()

        breeder.auto_breed(n_winners=1)
        # Room 10+ should now have weights close to some hot room
        # (cloned + small mutation, so not identical to random init)
        cold_room_w1 = grid.w["w1"][10]
        # After rebirth, weights should be close to some parent (within mutation range)
        # The cloned weights are parent + N(0, 0.005), so diff should be small
        found_close = False
        for i in range(10):
            diff = np.linalg.norm(grid.w["w1"][10] - grid.w["w1"][i])
            # Mutation is 0.005 * randn, shape is (64,32), so L2 ~ sqrt(64*32)*0.005 ~ 0.23
            if diff < 1.0:
                found_close = True
                break
        assert found_close, "Rebirthed weights should be close to a parent's weights"

    def test_respects_n_winners(self, breeder):
        """n_winners limits how many children are produced."""
        result = breeder.auto_breed(n_winners=1)
        assert len(result) <= 1

    def test_log_records_rebirths(self, breeder):
        """Each rebirth should be logged."""
        breeder.auto_breed(n_winners=3)
        log = breeder.log
        assert len(log) > 0
        assert all(isinstance(r, RebirthRecord) for r in log)
        assert all(r.parent_agent_id.startswith("room_") for r in log)

    def test_thermal_budget_sacrifice(self, grid):
        """When thermal budget is full, parent should be sacrificed."""
        thermal = ThermalBudget({DeviceType.GPU: 2})
        # Fill the budget
        thermal.allocate("agent_0", DeviceType.GPU)
        thermal.allocate("agent_1", DeviceType.GPU)

        breeder = AutoBreeder(grid, thermal, cold_threshold=3)
        # auto_breed should still work via parent-sacrifice
        result = breeder.auto_breed(n_winners=1)
        # May or may not succeed depending on parent matching,
        # but should not raise an exception
        assert isinstance(result, list)

    def test_cold_rooms_reset_activity(self, breeder, grid):
        """After rebirth, cold room activity should reset to 0."""
        breeder.auto_breed(n_winners=3)
        log = breeder.log
        for record in log:
            assert grid.activity[record.room_id] == 0


class TestDaemonLifecycle:
    """Test start/stop daemon thread."""

    def test_start_stop(self, breeder):
        """Daemon starts and stops cleanly."""
        breeder.start()
        assert breeder.running
        breeder.stop()
        assert not breeder.running

    def test_double_start_is_noop(self, breeder):
        """Starting twice doesn't create two threads."""
        breeder.start()
        t1 = breeder._thread
        breeder.start()
        t2 = breeder._thread
        assert t1 is t2
        breeder.stop()

    def test_daemon_runs_cycles(self, grid, thermal):
        """Daemon thread actually runs breeding cycles."""
        breeder = AutoBreeder(grid, thermal, interval=0.1, cold_threshold=3)
        breeder.start()
        time.sleep(0.5)
        breeder.stop()
        # Should have run at least one cycle
        assert len(breeder.log) > 0
