"""Tests for priority_scheduler.py — Priority task scheduler.

Run: python3 -m pytest tests/test_priority_scheduler.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.priority_scheduler import PriorityScheduler


class TestPriorityScheduler:
    def test_create(self):
        sched = PriorityScheduler()
        assert sched.queue_size() == 0

    def test_submit_and_next(self):
        sched = PriorityScheduler()
        tid = sched.submit("task-1", lambda: 42, priority=1)
        assert tid.startswith("task-")
        task = sched.next()
        assert task is not None
        assert task.name == "task-1"

    def test_priority_ordering(self):
        sched = PriorityScheduler()
        sched.submit("low", lambda: 1, priority=1)
        sched.submit("high", lambda: 2, priority=10)
        sched.submit("med", lambda: 3, priority=5)
        assert sched.next().name == "high"
        assert sched.next().name == "med"
        assert sched.next().name == "low"

    def test_fifo_same_priority(self):
        sched = PriorityScheduler()
        sched.submit("first", lambda: 1, priority=5)
        sched.submit("second", lambda: 2, priority=5)
        assert sched.next().name == "first"
        assert sched.next().name == "second"

    def test_peek(self):
        sched = PriorityScheduler()
        sched.submit("x", lambda: 1, priority=5)
        assert sched.peek().name == "x"
        assert sched.queue_size() == 1

    def test_empty_next(self):
        sched = PriorityScheduler()
        assert sched.next() is None

    def test_should_preempt(self):
        sched = PriorityScheduler()
        sched.submit("urgent", lambda: 1, priority=10)
        assert sched.should_preempt(current_priority=5) is True
        assert sched.should_preempt(current_priority=10) is False

    def test_preemptible_flag(self):
        sched = PriorityScheduler()
        sched.submit("bg", lambda: 1, priority=1, preemptible=True)
        task = sched.next()
        assert task.preemptible is True

    def test_mark_running_and_completed(self):
        sched = PriorityScheduler()
        sched.submit("x", lambda: 1)
        task = sched.next()
        sched.mark_running(task)
        assert sched.stats()["running"] == 1
        sched.mark_completed(task)
        assert sched.stats()["completed"] == 1
        assert sched.stats()["running"] == 0

    def test_list_queue(self):
        sched = PriorityScheduler()
        sched.submit("a", lambda: 1, priority=3)
        sched.submit("b", lambda: 2, priority=1)
        q = sched.list_queue()
        assert len(q) == 2
        assert q[0]["name"] == "a"

    def test_stats(self):
        sched = PriorityScheduler()
        sched.submit("x", lambda: 1)
        assert sched.stats()["queued"] == 1

    def test_task_execution(self):
        sched = PriorityScheduler()
        results = []
        sched.submit("add", lambda: results.append(1), priority=1)
        task = sched.next()
        task.fn()
        assert results == [1]

    def test_repr(self):
        sched = PriorityScheduler()
        assert "PriorityScheduler" in repr(sched)
