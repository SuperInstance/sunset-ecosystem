"""Tests for priority_queue.py — Breeding task priority queue.

Run: python3 -m pytest tests/test_priority_queue.py -v --tb=short
"""
from __future__ import annotations

import pytest

from swarm.priority_queue import BreedingPriorityQueue, QueuedTask


class TestQueuedTask:
    def test_ordering(self):
        t1 = QueuedTask(priority=1, sequence=1, task_id="a")
        t2 = QueuedTask(priority=2, sequence=2, task_id="b")
        assert t1 < t2  # lower priority = higher priority

    def test_fifo_same_priority(self):
        t1 = QueuedTask(priority=1, sequence=1, task_id="a")
        t2 = QueuedTask(priority=1, sequence=2, task_id="b")
        assert t1 < t2  # earlier sequence = higher priority


class TestBreedingPriorityQueue:
    def test_create_empty(self):
        pq = BreedingPriorityQueue()
        assert len(pq) == 0
        assert pq.is_empty()
        assert not pq.is_full()

    def test_enqueue_and_dequeue(self):
        pq = BreedingPriorityQueue()
        task = pq.enqueue({"agent": "a"}, priority=3)
        assert task is not None
        assert len(pq) == 1
        out = pq.dequeue()
        assert out.task_id == task.task_id
        assert out.priority == 3

    def test_priority_order(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"a": 1}, priority=5)
        pq.enqueue({"b": 1}, priority=1)
        pq.enqueue({"c": 1}, priority=3)
        out = pq.dequeue()
        assert out.priority == 1  # highest priority first

    def test_fifo_within_priority(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"a": 1}, priority=1)
        pq.enqueue({"b": 1}, priority=1)
        out1 = pq.dequeue()
        out2 = pq.dequeue()
        assert out1.sequence < out2.sequence

    def test_capacity_limit(self):
        pq = BreedingPriorityQueue(capacity=2)
        pq.enqueue({"a": 1})
        pq.enqueue({"b": 1})
        dropped = pq.enqueue({"c": 1})
        assert dropped is None
        assert pq.stats()["dropped"] == 1

    def test_dequeue_empty(self):
        pq = BreedingPriorityQueue()
        assert pq.dequeue() is None

    def test_peek(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"a": 1}, priority=2)
        pq.enqueue({"b": 1}, priority=1)
        top = pq.peek()
        assert top.priority == 1
        assert len(pq) == 2  # peek doesn't remove

    def test_dequeue_many(self):
        pq = BreedingPriorityQueue()
        for i in range(5):
            pq.enqueue({"i": i}, priority=i)
        tasks = pq.dequeue_many(3)
        assert len(tasks) == 3
        assert tasks[0].priority == 0
        assert tasks[2].priority == 2

    def test_enqueue_many(self):
        pq = BreedingPriorityQueue()
        items = [({"i": i}, i) for i in range(5)]
        tasks = pq.enqueue_many(items)
        assert len(tasks) == 5
        assert len(pq) == 5

    def test_priorities(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({}, priority=1)
        pq.enqueue({}, priority=2)
        pq.enqueue({}, priority=1)
        counts = pq.priorities()
        assert counts[1] == 2
        assert counts[2] == 1

    def test_drop_where(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"agent": "a"}, priority=1)
        pq.enqueue({"agent": "b"}, priority=2)
        pq.enqueue({"agent": "a"}, priority=3)
        removed = pq.drop_where(lambda t: t.payload.get("agent") == "a")
        assert removed == 2
        assert len(pq) == 1

    def test_find(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"agent": "a"}, priority=1)
        pq.enqueue({"agent": "b"}, priority=2)
        found = pq.find(lambda t: t.payload.get("agent") == "a")
        assert len(found) == 1
        assert found[0].payload["agent"] == "a"

    def test_stats(self):
        pq = BreedingPriorityQueue(capacity=10)
        pq.enqueue({"a": 1}, priority=1)
        pq.enqueue({"b": 1}, priority=2)
        s = pq.stats()
        assert s["size"] == 2
        assert s["capacity"] == 10
        assert s["dropped"] == 0
        assert s["dequeued"] == 0
        assert 1 in s["priorities"]
        assert 2 in s["priorities"]

    def test_wait_times(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"a": 1})
        waits = pq.wait_times()
        assert len(waits) == 1
        assert waits[0] >= 0.0

    def test_mean_max_wait(self):
        pq = BreedingPriorityQueue()
        assert pq.mean_wait() == 0.0
        assert pq.max_wait() == 0.0

    def test_repr(self):
        pq = BreedingPriorityQueue(capacity=100)
        assert "BreedingPriorityQueue" in repr(pq)

    def test_dequeue_updates_stats(self):
        pq = BreedingPriorityQueue()
        pq.enqueue({"a": 1})
        pq.dequeue()
        assert pq.stats()["dequeued"] == 1
