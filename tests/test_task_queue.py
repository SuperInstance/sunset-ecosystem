"""Tests for task_queue.py — Priority task queue with delayed tasks.

Run: python3 -m pytest tests/test_task_queue.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.task_queue import TaskQueue


class TestTaskQueue:
    def test_create(self):
        queue = TaskQueue()
        assert queue.is_empty() is True
        assert queue.stats()["size"] == 0

    def test_enqueue(self):
        queue = TaskQueue()
        assert queue.enqueue("task-1", priority=0) is True
        assert queue.size() == 1

    def test_enqueue_duplicate(self):
        queue = TaskQueue()
        queue.enqueue("task-1", priority=0)
        assert queue.enqueue("task-1", priority=1) is False

    def test_dequeue_priority(self):
        queue = TaskQueue(clock=lambda: 0)
        queue.enqueue("task-1", priority=1)
        queue.enqueue("task-2", priority=0)
        task = queue.dequeue()
        assert task["task_id"] == "task-2"  # Lower priority number = higher priority

    def test_dequeue_empty(self):
        queue = TaskQueue()
        assert queue.dequeue() is None

    def test_dequeue_delayed(self):
        queue = TaskQueue(clock=lambda: 0)
        queue.enqueue("task-1", delay_sec=10)
        assert queue.dequeue() is None
        queue._clock = lambda: 15
        task = queue.dequeue()
        assert task["task_id"] == "task-1"

    def test_cancel(self):
        queue = TaskQueue()
        queue.enqueue("task-1")
        assert queue.cancel("task-1") is True
        assert queue.dequeue() is None
        assert queue.cancel("missing") is False

    def test_peek(self):
        queue = TaskQueue()
        queue.enqueue("task-1", priority=0)
        task = queue.peek()
        assert task["task_id"] == "task-1"
        assert queue.size() == 1  # Not removed

    def test_completed(self):
        queue = TaskQueue()
        queue.enqueue("task-1")
        queue.dequeue()
        assert queue.completed() == ["task-1"]

    def test_pending(self):
        queue = TaskQueue()
        queue.enqueue("task-1")
        queue.enqueue("task-2")
        assert sorted(queue.pending()) == ["task-1", "task-2"]

    def test_stats(self):
        queue = TaskQueue()
        queue.enqueue("task-1")
        queue.enqueue("task-2")
        queue.dequeue()
        stats = queue.stats()
        assert stats["size"] == 1
        assert stats["completed"] == 1

    def test_repr(self):
        queue = TaskQueue()
        assert "TaskQueue" in repr(queue)
