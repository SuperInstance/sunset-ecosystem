"""Tests for job_scheduler.py — Job scheduling with intervals and one-off jobs.

Run: python3 -m pytest tests/test_job_scheduler.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.job_scheduler import JobScheduler


class TestJobScheduler:
    def test_create(self):
        sched = JobScheduler(clock=lambda: 0)
        assert sched.stats()["recurring"] == 0
        assert sched.stats()["one_off"] == 0

    def test_schedule_recurring(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        assert sched.schedule("backup", interval_sec=60, fn=lambda: calls.append(1)) is True
        assert "backup" in sched.job_names()

    def test_schedule_duplicate(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("backup", interval_sec=60, fn=lambda: None)
        assert sched.schedule("backup", interval_sec=60, fn=lambda: None) is False

    def test_schedule_once(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        sched.schedule_once("alert", delay_sec=10, fn=lambda: calls.append(1))
        assert "alert" in sched.job_names()
        assert sched.stats()["one_off"] == 1

    def test_tick_recurring(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        sched.schedule("task", interval_sec=10, fn=lambda: calls.append(1))
        results = sched.tick()
        assert len(results) == 1
        assert results[0]["name"] == "task"
        assert results[0]["type"] == "recurring"
        assert len(calls) == 1

    def test_tick_not_due(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("task", interval_sec=10, fn=lambda: None)
        results = sched.tick()
        assert len(results) == 1  # Due immediately at time 0

    def test_tick_interval(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        sched.schedule("task", interval_sec=10, fn=lambda: calls.append(1))
        sched.tick()  # Executes at t=0
        sched._clock = lambda: 5
        results = sched.tick()
        assert len(results) == 0  # Not due yet
        sched._clock = lambda: 10
        results = sched.tick()
        assert len(results) == 1  # Due again
        assert len(calls) == 2

    def test_tick_one_off(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        sched.schedule_once("alert", delay_sec=10, fn=lambda: calls.append(1))
        results = sched.tick()
        assert len(results) == 0  # Not due yet
        sched._clock = lambda: 15
        results = sched.tick()
        assert len(results) == 1
        assert len(calls) == 1

    def test_max_runs(self):
        sched = JobScheduler(clock=lambda: 0)
        calls = []
        sched.schedule("task", interval_sec=10, fn=lambda: calls.append(1), max_runs=2)
        sched.tick()  # Run 1
        sched._clock = lambda: 10
        sched.tick()  # Run 2
        sched._clock = lambda: 20
        results = sched.tick()  # Should not run (max reached)
        assert len(results) == 0
        assert len(calls) == 2

    def test_unschedule(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("task", interval_sec=10, fn=lambda: None)
        assert sched.unschedule("task") is True
        assert sched.unschedule("missing") is False

    def test_unschedule_one_off(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule_once("task", delay_sec=10, fn=lambda: None)
        assert sched.unschedule("task") is True
        assert sched.stats()["one_off"] == 0

    def test_due_jobs(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("a", interval_sec=10, fn=lambda: None)
        sched.schedule_once("b", delay_sec=5, fn=lambda: None)
        sched._clock = lambda: 10
        assert sorted(sched.due_jobs()) == ["a", "b"]

    def test_due_jobs_not_due(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule_once("a", delay_sec=10, fn=lambda: None)
        assert sched.due_jobs() == []

    def test_next_run(self):
        sched = JobScheduler(clock=lambda: 100)
        sched.schedule("task", interval_sec=10, fn=lambda: None)
        assert sched.next_run("task") == 100
        assert sched.next_run("missing") is None

    def test_tick_error(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("task", interval_sec=10, fn=lambda: 1 / 0)
        results = sched.tick()
        assert results[0]["error"] == "division by zero"
        assert sched.stats()["skipped"] == 1

    def test_stats(self):
        sched = JobScheduler(clock=lambda: 0)
        sched.schedule("a", interval_sec=10, fn=lambda: None)
        sched.schedule_once("b", delay_sec=5, fn=lambda: None)
        stats = sched.stats()
        assert stats["recurring"] == 1
        assert stats["one_off"] == 1

    def test_repr(self):
        sched = JobScheduler()
        assert "JobScheduler" in repr(sched)
