"""Tests for subagent_conductor.py — Fleet subagent lifecycle manager.

Run: python3 -m pytest tests/test_subagent_conductor.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.subagent_conductor import (
    GatewayHealthMonitor,
    SubagentConductor,
    TaskPriority,
    TaskResult,
    TaskSpec,
    TaskStatus,
)


class TestGatewayHealthMonitor:
    def test_initially_healthy(self):
        monitor = GatewayHealthMonitor()
        assert monitor.is_healthy
        assert monitor.recommend_direct is False

    def test_record_latency(self):
        monitor = GatewayHealthMonitor()
        monitor.record(100, True)
        assert monitor.ewma_latency_ms == pytest.approx(100.0)

    def test_high_latency_triggers_direct(self):
        monitor = GatewayHealthMonitor()
        for _ in range(5):
            monitor.record(10000, True)  # 10s latency
        assert monitor.recommend_direct is True

    def test_circuit_opens_on_failures(self):
        monitor = GatewayHealthMonitor()
        monitor.record(100, False)
        monitor.record(200, False)
        monitor.record(300, False)
        assert not monitor.is_healthy
        assert monitor._circuit_open

    def test_circuit_closes_after_cooldown(self):
        import time
        monitor = GatewayHealthMonitor()
        for _ in range(3):
            monitor.record(100, False)
        assert monitor._circuit_open
        # Fast-forward cooldown
        monitor._circuit_until = time.time() - 1
        assert monitor.is_healthy  # auto-close

    def test_success_rate(self):
        monitor = GatewayHealthMonitor()
        assert monitor.success_rate == 1.0
        monitor.record(100, True)
        monitor.record(100, False)
        assert monitor.success_rate == pytest.approx(0.5)

    def test_report(self):
        monitor = GatewayHealthMonitor()
        monitor.record(150, True)
        r = monitor.report()
        assert "ewma_latency_ms" in r
        assert "success_rate" in r
        assert "circuit_open" in r
        assert r["healthy"] is True


class TestTaskSpec:
    def test_cache_key_deterministic(self):
        t1 = TaskSpec(
            task_id="a",
            task_type="code",
            description="test",
            priority=TaskPriority.HIGH,
            payload={"file": "foo.py"},
        )
        t2 = TaskSpec(
            task_id="b",
            task_type="code",
            description="test",
            priority=TaskPriority.HIGH,
            payload={"file": "foo.py"},
        )
        # Same payload should give same key (ignoring id)
        assert t1.cache_key() == t2.cache_key()


class TestSubagentConductor:
    def test_create(self):
        c = SubagentConductor()
        assert c.max_concurrent == 3
        r = c.report()
        assert r["tasks_submitted"] == 0
        assert r["in_flight"] == 0

    def test_submit_queues_task(self):
        c = SubagentConductor()
        task = TaskSpec(
            task_id="t1",
            task_type="code",
            description="test",
            priority=TaskPriority.NORMAL,
            payload={},
        )
        c.submit(task)
        assert c.report()["tasks_submitted"] == 1
        assert c.queue.pending_count == 1

    def test_tick_dispatches_when_healthy(self):
        c = SubagentConductor()
        # Register a fallback so tick doesn't fail
        c.register_fallback("code", lambda t: TaskResult(
            task_id=t.task_id,
            status=TaskStatus.COMPLETED,
            output="fallback",
        ))

        # Make gateway appear overloaded so it falls back
        for _ in range(5):
            c.record_gateway_attempt(6000, True)

        task = TaskSpec(
            task_id="t1",
            task_type="code",
            description="test",
            priority=TaskPriority.NORMAL,
            payload={},
        )
        c.submit(task)
        results = c.tick()
        assert len(results) == 1
        assert results[0].execution_mode == "direct"
        assert c.report()["tasks_fallback"] == 1

    def test_on_subagent_complete(self):
        c = SubagentConductor()
        task = TaskSpec(
            task_id="t1",
            task_type="code",
            description="test",
            priority=TaskPriority.NORMAL,
            payload={},
        )
        c.submit(task)
        # Manually put in flight
        c._in_flight["t1"] = (task, 0.0)

        result = c.on_subagent_complete("t1", "output data")
        assert result.status == TaskStatus.COMPLETED
        assert result.output == "output data"
        assert "t1" not in c._in_flight

    def test_fallback_handler_error(self):
        c = SubagentConductor()
        c.register_fallback("code", lambda t: (_ for _ in ()).throw(ValueError("boom")))
        for _ in range(5):
            c.record_gateway_attempt(6000, True)

        task = TaskSpec(
            task_id="t1",
            task_type="code",
            description="test",
            priority=TaskPriority.NORMAL,
            payload={},
        )
        c.submit(task)
        results = c.tick()
        assert len(results) == 1
        assert results[0].status == TaskStatus.FAILED
        assert "boom" in results[0].error

    def test_priority_queue_order(self):
        c = SubagentConductor()
        tasks = [
            TaskSpec(task_id="low", task_type="x", description="", priority=TaskPriority.LOW, payload={}),
            TaskSpec(task_id="high", task_type="x", description="", priority=TaskPriority.HIGH, payload={}),
            TaskSpec(task_id="crit", task_type="x", description="", priority=TaskPriority.CRITICAL, payload={}),
            TaskSpec(task_id="norm", task_type="x", description="", priority=TaskPriority.NORMAL, payload={}),
        ]
        for t in tasks:
            c.submit(t)
        # Dequeue should return highest priority first
        first = c.queue.dequeue()
        assert first is not None
        assert first.priority == TaskPriority.CRITICAL
        second = c.queue.dequeue()
        assert second.priority == TaskPriority.HIGH

    def test_metrics_accumulate(self):
        c = SubagentConductor()
        c.register_fallback("code", lambda t: TaskResult(
            task_id=t.task_id,
            status=TaskStatus.COMPLETED,
        ))
        for _ in range(5):
            c.record_gateway_attempt(6000, True)

        for i in range(3):
            c.submit(TaskSpec(
                task_id=f"t{i}",
                task_type="code",
                description="",
                priority=TaskPriority.NORMAL,
                payload={},
            ))
        c.tick()
        r = c.report()
        assert r["tasks_submitted"] == 3
        assert r["tasks_fallback"] == 3

    def test_no_fallback_handler_skips(self):
        c = SubagentConductor()
        for _ in range(5):
            c.record_gateway_attempt(6000, True)

        c.submit(TaskSpec(
            task_id="t1",
            task_type="unknown_type",
            description="",
            priority=TaskPriority.NORMAL,
            payload={},
        ))
        results = c.tick()
        # No fallback registered — task stays in queue but nothing dispatched
        assert len(results) == 0
