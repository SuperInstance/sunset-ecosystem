"""Tests for FleetOrchestrator — central conductor for the fleet.

Reference: fleet/fleet_orchestrator.py
"""

from __future__ import annotations

import pytest

from fleet.fleet_orchestrator import (
    ExecutionResult,
    FleetBeat,
    FleetOrchestrator,
    TaskSpec,
)
from fleet.ternary_types import TernaryValue


class TestFleetOrchestrator:
    def test_init(self) -> None:
        conductor = FleetOrchestrator(node_id="test")
        assert conductor.node_id == "test"
        assert conductor.cycle_number == 0
        assert not conductor._initialized

    def test_initialize_fleet(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        assert conductor._initialized
        assert conductor._harbor is not None
        assert len(conductor._harbor.modules) == 20

    def test_beat(self) -> None:
        conductor = FleetOrchestrator()
        beat = conductor.beat()
        assert isinstance(beat, FleetBeat)
        assert beat.cycle_number == 1
        assert beat.node_id == "default"
        assert beat.health_status in [-1, 0, +1]

    def test_multiple_beats(self) -> None:
        conductor = FleetOrchestrator()
        for _ in range(3):
            conductor.beat()
        assert len(conductor.beats) == 3
        assert conductor.cycle_number == 3

    def test_check_fleet_health(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        health = conductor.check_fleet_health()
        assert "total_modules" in health
        assert health["total_modules"] == 20
        assert "ternary_score" in health
        assert "ternary_emoji" in health

    def test_check_fleet_health_not_initialized(self) -> None:
        conductor = FleetOrchestrator()
        health = conductor.check_fleet_health()
        assert "error" in health

    def test_generate_tasks(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        tasks = conductor.generate_tasks()
        assert isinstance(tasks, list)
        # Should have at least some tasks (mine, breed)
        assert len(tasks) >= 1

    def test_generate_tasks_sorted(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        tasks = conductor.generate_tasks()
        if len(tasks) > 1:
            priorities = [t.priority for t in tasks]
            assert priorities == sorted(priorities, reverse=True)

    def test_execute_task(self) -> None:
        conductor = FleetOrchestrator()
        task = TaskSpec(task_id="test-1", task_type="health_check", priority=1)
        result = conductor.execute_task(task)
        assert isinstance(result, ExecutionResult)
        assert result.task_id == "test-1"
        assert result.success is True
        assert result.duration_ms >= 0

    def test_execute_task_with_target(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        task = TaskSpec(
            task_id="test-health",
            task_type="health_check",
            target_module="VectorSwarm",
            priority=1,
        )
        result = conductor.execute_task(task)
        assert result.success is True
        assert "health" in result.output

    def test_execute_optimize_task(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        task = TaskSpec(task_id="opt-1", task_type="optimize", priority=1)
        result = conductor.execute_task(task)
        assert result.success is True
        if conductor._breed_optimizer:
            assert "archive_coverage" in result.output

    def test_execute_breed_task(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        task = TaskSpec(task_id="breed-1", task_type="breed", priority=1)
        result = conductor.execute_task(task)
        assert result.success is True
        if conductor._breed_optimizer:
            assert "parents_selected" in result.output

    def test_execute_sync_task(self) -> None:
        conductor = FleetOrchestrator()
        task = TaskSpec(
            task_id="sync-1",
            task_type="sync",
            priority=1,
            payload={"gap": {"source": "A", "target": "B"}},
        )
        result = conductor.execute_task(task)
        assert result.success is True
        assert result.output["action"] == "sync_initiated"

    def test_execute_mine_task(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        task = TaskSpec(task_id="mine-1", task_type="mine", priority=1)
        result = conductor.execute_task(task)
        assert result.success is True
        if conductor._pattern_mine:
            assert "rules_found" in result.output
            assert "templates_found" in result.output

    def test_execute_predict_task(self) -> None:
        conductor = FleetOrchestrator()
        task = TaskSpec(task_id="pred-1", task_type="predict", priority=1)
        result = conductor.execute_task(task)
        assert result.success is True

    def test_execute_unknown_task(self) -> None:
        conductor = FleetOrchestrator()
        task = TaskSpec(task_id="unknown-1", task_type="unknown_type", priority=1)
        result = conductor.execute_task(task)
        assert result.success is True
        assert result.output["status"] == "unknown_task_type"

    def test_detect_anomalies(self) -> None:
        conductor = FleetOrchestrator()
        anomalies = conductor.detect_anomalies()
        assert isinstance(anomalies, list)

    def test_detect_anomalies_with_critical(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        # Set a module to critical
        conductor._harbor.update_module_health("VectorSwarm", "critical")
        anomalies = conductor.detect_anomalies()
        assert len(anomalies) > 0
        assert any(a["type"] == "critical_module" for a in anomalies)

    def test_detect_anomalies_beat_streak(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()  # Need harbor for anomaly detection
        for i in range(3):
            conductor.beats.append(
                FleetBeat(
                    timestamp=i,
                    node_id="test",
                    cycle_number=i + 1,
                    health_status=0,
                    tasks_executed=0,
                    tasks_failed=2,
                )
            )
        anomalies = conductor.detect_anomalies()
        assert any(a["type"] == "beat_failure_streak" for a in anomalies)

    def test_detect_anomalies_health_degradation(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()  # Need harbor for anomaly detection
        for i in range(5):
            conductor.beats.append(
                FleetBeat(
                    timestamp=i,
                    node_id="test",
                    cycle_number=i + 1,
                    health_status=TernaryValue.NEG,
                    tasks_executed=1,
                    tasks_failed=0,
                )
            )
        anomalies = conductor.detect_anomalies()
        assert any(a["type"] == "health_degradation" for a in anomalies)

    def test_get_stats_empty(self) -> None:
        conductor = FleetOrchestrator()
        stats = conductor.get_stats()
        assert stats["beats"] == 0
        assert stats["cycles"] == 0

    def test_get_stats_with_beats(self) -> None:
        conductor = FleetOrchestrator()
        conductor.beat()
        conductor.beat()
        stats = conductor.get_stats()
        assert stats["beats"] == 2
        assert stats["cycles"] == 2
        assert stats["mean_duration_ms"] > 0

    def test_get_stats_success_rate(self) -> None:
        conductor = FleetOrchestrator()
        conductor.beat()
        stats = conductor.get_stats()
        assert 0.0 <= stats["success_rate"] <= 1.0

    def test_generate_report(self) -> None:
        conductor = FleetOrchestrator()
        conductor.beat()
        report = conductor.generate_report()
        assert report["node_id"] == "default"
        assert report["cycle_number"] == 1
        assert report["initialized"] is True
        assert "health" in report
        assert "stats" in report
        assert "anomalies" in report
        assert "subsystems" in report

    def test_report_subsystems(self) -> None:
        conductor = FleetOrchestrator()
        conductor.beat()
        report = conductor.generate_report()
        subsystems = report["subsystems"]
        assert isinstance(subsystems["harbor"], bool)
        assert isinstance(subsystems["tminus"], bool)
        assert isinstance(subsystems["pattern_mine"], bool)
        assert isinstance(subsystems["breed_optimizer"], bool)
        assert isinstance(subsystems["cognitive_cache"], bool)
        assert isinstance(subsystems["vector_swarm"], bool)

    def test_task_with_deadline(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        # If tminus is available, critical tasks get deadlines
        tasks = conductor.generate_tasks()
        for task in tasks:
            if task.priority == 4 and conductor._tminus:
                assert task.deadline_secs is not None

    def test_beat_records_duration(self) -> None:
        conductor = FleetOrchestrator()
        beat = conductor.beat()
        assert beat.duration_ms >= 0

    def test_beat_health_status(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        # All modules are healthy, so health should be POS
        beat = conductor.beat()
        assert beat.health_status == TernaryValue.POS

    def test_beat_modules_checked(self) -> None:
        conductor = FleetOrchestrator()
        conductor.initialize_fleet()
        beat = conductor.beat()
        assert beat.modules_checked == 20

    def test_task_spec_defaults(self) -> None:
        task = TaskSpec(task_id="test", task_type="health_check")
        assert task.priority == 1
        assert task.status == "pending"
        assert task.payload == {}

    def test_execution_result_success(self) -> None:
        result = ExecutionResult(task_id="test", success=True)
        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 0.0

    def test_execution_result_failure(self) -> None:
        result = ExecutionResult(task_id="test", success=False, error="boom")
        assert result.success is False
        assert result.error == "boom"
