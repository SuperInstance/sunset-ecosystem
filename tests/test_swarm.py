"""Tests for swarm modules: penrose, broadcast, swarm_runner."""

import pytest

from swarm.penrose import PenrosePosition, assign_positions, compute_overlap, minimum_overlap
from swarm.broadcast import BroadcastMessage, BroadcastingChannel
from swarm.swarm_runner import SwarmRunner, SwarmStatus
from nerve.fiber import NerveFiber


class TestPenrose:
    def test_assign_positions(self):
        agents = [f"agent-{i}" for i in range(12)]
        positions = assign_positions(agents)
        assert len(positions) == 12
        assert all(isinstance(p, PenrosePosition) for p in positions)

    def test_unique_positions(self):
        agents = [f"a{i}" for i in range(20)]
        positions = assign_positions(agents)
        coords = [(p.x, p.y) for p in positions]
        assert len(set(coords)) == 20  # All unique

    def test_compute_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        p2 = PenrosePosition("b", 0.5, 0.0, 0, 0.0)
        overlap = compute_overlap(p1, p2, radius=1.0)
        assert 0.0 < overlap < 1.0

    def test_no_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        p2 = PenrosePosition("b", 10.0, 10.0, 0, 0.0)
        assert compute_overlap(p1, p2, radius=1.0) == 0.0

    def test_full_overlap(self):
        p1 = PenrosePosition("a", 0.0, 0.0, 0, 0.0)
        assert compute_overlap(p1, p1, radius=1.0) == 1.0

    def test_minimum_overlap(self):
        agents = [f"a{i}" for i in range(12)]
        positions = assign_positions(agents)
        min_ov = minimum_overlap(positions, radius=2.0)
        assert 0.0 <= min_ov <= 1.0


class TestBroadcast:
    def test_subscribe_and_broadcast(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        msg = BroadcastMessage(content="hello", source_agent="src", target_room="room-1")
        recipients = ch.broadcast(msg)
        assert "agent-1" in recipients

    def test_no_match(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        msg = BroadcastMessage(content="hello", source_agent="src", target_room="room-2")
        recipients = ch.broadcast(msg)
        assert "agent-1" not in recipients

    def test_receive(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent-1", "room-1")
        ch.broadcast(BroadcastMessage(content="test", target_room="room-1"))
        msgs = ch.receive("agent-1")
        assert len(msgs) == 1

    def test_feedback(self):
        ch = BroadcastingChannel()
        ch.subscribe("a", "r")
        ch.broadcast(BroadcastMessage(content="x", source_agent="src", target_room="r"))
        w1 = ch.get_channel_weight("src", "a")
        ch.feedback("src", "a", useful=True)
        w2 = ch.get_channel_weight("src", "a")
        assert w2 > w1


class TestSwarmRunner:
    def test_create(self):
        runner = SwarmRunner()
        assert runner.status().total_agents == 0

    def test_add_fiber_and_tick(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1", epsilon=0.2))
        result = runner.tick("test signal")
        assert "tiles" in result
        assert "f1" in result["tiles"]

    def test_distribute(self):
        runner = SwarmRunner()
        agents = [f"agent-{i}" for i in range(12)]
        positions = runner.distribute(agents)
        assert len(positions) == 12

    def test_status(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1"))
        status = runner.status()
        assert isinstance(status, SwarmStatus)

    def test_spare_capacity(self):
        runner = SwarmRunner()
        runner.add_fiber(NerveFiber("f1"))
        cap = runner.spare_capacity()
        assert 0.0 <= cap <= 1.0

    def test_backtest_cycle(self):
        runner = SwarmRunner()
        # No spare capacity initially (adaptation=0)
        assert runner.run_backtest_cycle() is False
