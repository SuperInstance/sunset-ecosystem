"""Tests for work_queue.py — Priority work queue.

Run: python3 -m pytest tests/test_work_queue.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.work_queue import WorkQueue


class TestWorkQueue:
    def test_create(self):
        q = WorkQueue()
        assert q.size() == 0
        assert q.pending_count() == 0
        assert q.dead_letter_size() == 0

    def test_enqueue_and_claim(self):
        q = WorkQueue()
        q.enqueue("breed", payload={"room_id": 1})
        assert q.size() == 1

        item = q.claim("worker-1")
        assert item is not None
        assert item.kind == "breed"
        assert item.claimed_by == "worker-1"
        assert q.size() == 0
        assert q.pending_count() == 1

    def test_claim_empty(self):
        q = WorkQueue()
        item = q.claim("worker-1", timeout=0.1)
        assert item is None

    def test_ack(self):
        q = WorkQueue()
        uid = q.enqueue("breed", payload={"room_id": 1})
        item = q.claim("worker-1")
        assert q.ack(uid) is True
        assert q.pending_count() == 0
        assert q.ack(uid) is False  # Already acked

    def test_nack_requeue(self):
        q = WorkQueue(visibility_timeout=5.0, max_retries=3)
        uid = q.enqueue("breed", payload={"room_id": 1})
        item = q.claim("worker-1")
        assert q.nack(uid) is True
        assert item.claimed_by is None
        assert q.size() == 1

    def test_nack_max_retries(self):
        q = WorkQueue(visibility_timeout=5.0, max_retries=3)
        uid = q.enqueue("breed", payload={"room_id": 1})
        q.claim("worker-1")
        q.nack(uid)  # retry 1
        q.claim("worker-1")
        q.nack(uid)  # retry 2
        q.claim("worker-1")
        assert q.nack(uid) is False  # retry 3 >= max_retries, moved to dead letter
        assert q.dead_letter_size() == 1
        # Item is gone from queue
        assert q.claim("worker-1", timeout=0.1) is None

    def test_priority_order(self):
        q = WorkQueue()
        q.enqueue("low", payload={}, priority=5)
        q.enqueue("high", payload={}, priority=1)
        q.enqueue("medium", payload={}, priority=3)

        item = q.claim("worker-1")
        assert item.kind == "high"
        item = q.claim("worker-1")
        assert item.kind == "medium"
        item = q.claim("worker-1")
        assert item.kind == "low"

    def test_release(self):
        q = WorkQueue(visibility_timeout=5.0)
        uid = q.enqueue("breed", payload={"room_id": 1})
        q.claim("worker-1")
        assert q.release(uid) is True
        assert q.size() == 1
        assert q.pending_count() == 0

    def test_visibility_timeout_reclaim(self):
        fake_time = [0.0]
        def clock():
            return fake_time[0]

        q = WorkQueue(visibility_timeout=2.0, clock=clock)
        uid = q.enqueue("breed", payload={"room_id": 1})
        q.claim("worker-1")
        fake_time[0] = 3.0
        item = q.claim("worker-2")
        assert item is not None
        assert item.claimed_by == "worker-2"

    def test_dead_letter_peek_and_purge(self):
        q = WorkQueue(max_retries=2)
        uid = q.enqueue("breed", payload={"room_id": 1})
        q.claim("worker-1")
        q.nack(uid)  # retry 1
        q.claim("worker-1")
        q.nack(uid)  # retry 2 >= max_retries, to dead letter
        assert q.dead_letter_size() == 1
        assert len(q.peek_dead_letter(10)) == 1
        assert q.purge_dead_letter() == 1
        assert q.dead_letter_size() == 0

    def test_custom_id(self):
        q = WorkQueue()
        uid = q.enqueue("breed", payload={}, item_id="my-task-42")
        item = q.claim("worker-1")
        assert item.id == "my-task-42"

    def test_stats(self):
        q = WorkQueue()
        q.enqueue("breed", payload={})
        q.claim("worker-1")
        stats = q.stats()
        assert stats["visible"] == 0
        assert stats["claimed"] == 1
        assert stats["dead_letter"] == 0

    def test_repr(self):
        q = WorkQueue()
        assert "WorkQueue" in repr(q)
