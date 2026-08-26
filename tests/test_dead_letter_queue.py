"""Tests for dead_letter_queue.py — Failed message capture and replay.

Run: python3 -m pytest tests/test_dead_letter_queue.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.dead_letter_queue import DeadLetterQueue


class TestDeadLetterQueue:
    def test_create(self):
        dlq = DeadLetterQueue()
        assert dlq.size() == 0

    def test_enqueue_and_get(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("msg-1", error="timeout", payload={"x": 1})
        entry = dlq.get("msg-1")
        assert entry is not None
        assert entry.error == "timeout"

    def test_dequeue(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("msg-1", error="timeout", payload={})
        entry = dlq.dequeue("msg-1")
        assert entry is not None
        assert dlq.get("msg-1") is None

    def test_max_size_eviction(self):
        dlq = DeadLetterQueue(max_size=2)
        dlq.enqueue("a", error="e", payload={})
        dlq.enqueue("b", error="e", payload={})
        dlq.enqueue("c", error="e", payload={})
        assert dlq.size() == 2
        assert dlq.get("a") is None

    def test_replay_success(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("msg-1", error="e", payload={"x": 1})
        success = dlq.replay("msg-1", lambda p: True)
        assert success is True
        assert dlq.get("msg-1") is None

    def test_replay_failure(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("msg-1", error="e", payload={})
        success = dlq.replay("msg-1", lambda p: False)
        assert success is False
        assert dlq.get("msg-1") is not None

    def test_replay_all(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("a", error="e", payload={})
        dlq.enqueue("b", error="e", payload={})
        results = dlq.replay_all(lambda p: True)
        assert all(results.values())
        assert dlq.size() == 0

    def test_ttl_eviction(self):
        dlq = DeadLetterQueue(ttl_sec=0.05)
        dlq.enqueue("msg-1", error="e", payload={}, timestamp=time.time())
        time.sleep(0.06)
        dlq.enqueue("msg-2", error="e", payload={}, timestamp=time.time())
        assert dlq.get("msg-1") is None
        assert dlq.get("msg-2") is not None

    def test_list_failed(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("a", error="timeout", payload={})
        dlq.enqueue("b", error="crash", payload={})
        failed = dlq.list_failed()
        assert len(failed) == 2

    def test_errors_by_type(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("a", error="timeout", payload={})
        dlq.enqueue("b", error="timeout", payload={})
        dlq.enqueue("c", error="crash", payload={})
        counts = dlq.errors_by_type()
        assert counts["timeout"] == 2
        assert counts["crash"] == 1

    def test_purge(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("a", error="e", payload={})
        dlq.purge()
        assert dlq.size() == 0

    def test_repr(self):
        dlq = DeadLetterQueue()
        assert "DeadLetterQueue" in repr(dlq)
