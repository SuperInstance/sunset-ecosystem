"""Tests for swarm_runner.py — SwarmRunner orchestration."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from swarm.swarm_runner import SwarmRunner, SwarmStatus
from nerve.fiber import NerveFiber
from nerve.routing import RoutingLayer
from swarm.broadcast import BroadcastingChannel


class TestSwarmRunner:
    def test_init_defaults(self):
        runner = SwarmRunner()
        assert runner._fibers == {}
        assert runner._ticks == 0
        assert runner._tasks_processed == 0

    def test_init_with_fibers(self):
        fiber = MagicMock(spec=NerveFiber)
        fiber.fiber_id = "f1"
        runner = SwarmRunner(fibers={"f1": fiber})
        assert "f1" in runner._fibers

    def test_add_fiber(self):
        runner = SwarmRunner()
        fiber = MagicMock(spec=NerveFiber)
        fiber.fiber_id = "f2"
        runner.add_fiber(fiber)
        assert "f2" in runner._fibers

    def test_distribute(self):
        runner = SwarmRunner()
        positions = runner.distribute(["a", "b", "c"])
        assert len(positions) == 3

    def test_tick(self):
        runner = SwarmRunner()
        result = runner.tick("hello")
        assert "tiles" in result
        assert "fired_routes" in result
        assert "latency_ms" in result
        assert runner._ticks == 1
        assert runner._tasks_processed == 1

    def test_spare_capacity(self):
        runner = SwarmRunner()
        cap = runner.spare_capacity()
        assert 0.0 <= cap <= 1.0

    def test_run_backtest_cycle_low_capacity(self):
        runner = SwarmRunner()
        with patch.object(runner, 'spare_capacity', return_value=0.1):
            assert runner.run_backtest_cycle() is False

    def test_run_backtest_cycle_ok(self):
        runner = SwarmRunner()
        with patch.object(runner, 'spare_capacity', return_value=0.5):
            assert runner.run_backtest_cycle() is True
            assert runner._backtests_run == 1

    def test_status(self):
        runner = SwarmRunner()
        status = runner.status()
        assert isinstance(status, SwarmStatus)
        assert status.total_agents == 0

    def test_run_forever_yields(self):
        """run_forever yields at least one result with max_ticks=1."""
        grid = MagicMock()
        grid.l = 4
        grid.n = 2
        grid.cold.return_value = []
        grid.tick.return_value = {"activity": [0.1, 0.2]}

        results = list(SwarmRunner.run_forever(grid, max_ticks=1, breed_interval=1000))
        assert len(results) == 1
        assert results[0]["tick"] == 1

    def test_run_forever_breeds_cold(self):
        """run_forever breeds cold rooms when interval reached."""
        grid = MagicMock()
        grid.l = 4
        grid.n = 2
        grid.cold.return_value = [0]
        grid.tick.return_value = {"activity": [0.1, 0.2]}

        results = list(SwarmRunner.run_forever(grid, max_ticks=100, breed_interval=10))
        # Check that by some tick, cold breeding happened
        breeding_ticks = [r for r in results if "cold" in r]
        assert len(breeding_ticks) >= 1
        # Verify breed was called
        grid.breed.assert_called()

    def test_repr(self):
        runner = SwarmRunner()
        r = repr(runner)
        assert "SwarmRunner" in r

    def test_status_repr(self):
        status = SwarmStatus(total_agents=5, adaptation_score=0.5)
        r = repr(status)
        assert "agents=5" in r
        assert "adapt=50.0%" in r
