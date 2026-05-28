"""Tests for task_scheduler.py — Cron-like task scheduling.

Run: python3 -m pytest tests/test_task_scheduler.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.task_scheduler import TaskScheduler


class TestTaskScheduler:
    def test_create(self):
        ts = TaskScheduler()
        assert ts.task_names() == []

    def test_schedule(self):
        ts = TaskScheduler()
        executed = []
        ts.schedule("test", lambda: executed.append(1), interval=1.0)
        assert "test" in ts.task_names()

    def test_schedule_no_params(self):
        ts = TaskScheduler()
        with pytest.raises(ValueError):
            ts.schedule("test", lambda: None)

    def test_unschedule(self):
        ts = TaskScheduler()
        ts.schedule("test", lambda: None, interval=1.0)
        assert ts.unschedule("test") is True
        assert ts.unschedule("test") is False

    def test_run_once_interval(self):
        ts = TaskScheduler()
        executed = []
        ts.schedule("test", lambda: executed.append(1), interval=0.1)
        # Not due yet
        ts.run_once()
        assert len(executed) == 0
        time.sleep(0.15)
        # Now due
        ts.run_once()
        assert len(executed) == 1

    def test_run_once_not_due(self):
        ts = TaskScheduler()
        executed = []
        ts.schedule("test", lambda: executed.append(1), interval=10.0)
        ts.run_once()
        assert len(executed) == 0

    def test_overlap_prevention(self):
        ts = TaskScheduler()
        executed = []
        def slow_task():
            executed.append(1)
            time.sleep(0.2)
        ts.schedule("test", slow_task, interval=0.1)
        # Simulate task already running
        ts._tasks["test"].running = True
        ts.run_once()  # Should skip because running=True
        assert len(executed) == 0
        # Now allow it to run
        ts._tasks["test"].running = False
        time.sleep(0.15)
        ts.run_once()
        assert len(executed) == 1

    def test_task_error(self):
        ts = TaskScheduler()
        ts.schedule("bad", lambda: (_ for _ in ()).throw(ValueError("boom")), interval=0.1)
        time.sleep(0.15)
        ts.run_once()
        execs = ts.executions("bad")
        assert len(execs) == 1
        assert execs[0].success is False
        assert "boom" in execs[0].error

    def test_success_rate(self):
        ts = TaskScheduler()
        count = [0]
        def maybe_fail():
            count[0] += 1
            if count[0] == 1:
                raise ValueError("fail")
        ts.schedule("test", maybe_fail, interval=0.1)
        time.sleep(0.15)
        ts.run_once()
        time.sleep(0.15)
        ts.run_once()
        assert ts.success_rate("test") == 0.5

    def test_stats(self):
        ts = TaskScheduler()
        ts.schedule("a", lambda: None, interval=1.0)
        ts.schedule("b", lambda: None, interval=2.0)
        stats = ts.stats()
        assert stats["tasks"] == 2
        assert stats["running"] is False

    def test_repr(self):
        ts = TaskScheduler()
        assert "TaskScheduler" in repr(ts)