"""Tests for orchestrator.py — Fleet workload orchestrator.

Run: python3 -m pytest tests/test_orchestrator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.orchestrator import FleetOrchestrator


class TestFleetOrchestrator:
    def test_create(self):
        orch = FleetOrchestrator()
        assert orch.stats()["nodes"] == 0

    def test_add_node(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4, "mem": 16})
        node = orch.get_node("node-1")
        assert node.total["cpu"] == 4

    def test_remove_node(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        assert orch.remove_node("node-1") is True
        assert orch.remove_node("missing") is False

    def test_submit_and_schedule(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        orch.submit_task("job-1", {"cpu": 2})
        assignments = orch.schedule()
        assert "job-1" in assignments
        assert assignments["job-1"] == "node-1"

    def test_priority_scheduling(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 2})
        orch.submit_task("low", {"cpu": 1}, priority=1)
        orch.submit_task("high", {"cpu": 1}, priority=10)
        assignments = orch.schedule()
        assert "high" in assignments

    def test_insufficient_resources(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 1})
        orch.submit_task("big", {"cpu": 2})
        assignments = orch.schedule()
        assert "big" not in assignments
        assert "big" in orch.unassigned_tasks()

    def test_remove_task(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        orch.submit_task("job-1", {"cpu": 2})
        orch.schedule()
        assert orch.remove_task("job-1") is True
        assert orch.get_assignment("job-1") is None

    def test_node_utilization(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        orch.submit_task("job-1", {"cpu": 2})
        orch.schedule()
        util = orch.node_utilization("node-1")
        assert util["cpu"] == 0.5

    def test_node_tasks(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        orch.submit_task("job-1", {"cpu": 2})
        orch.schedule()
        assert orch.node_tasks("node-1") == ["job-1"]

    def test_constrain_node_id(self):
        orch = FleetOrchestrator()
        orch.add_node("node-1", {"cpu": 4})
        orch.add_node("node-2", {"cpu": 4})
        orch.submit_task("job-1", {"cpu": 2}, constraints={"node_id": "node-2"})
        assignments = orch.schedule()
        assert assignments["job-1"] == "node-2"

    def test_repr(self):
        orch = FleetOrchestrator()
        assert "FleetOrchestrator" in repr(orch)
