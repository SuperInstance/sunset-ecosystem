"""Tests for ring_buffer.py — Circular ring buffer.

Run: python3 -m pytest tests/test_ring_buffer.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.ring_buffer import RingBuffer


class TestRingBuffer:
    def test_create(self):
        buf = RingBuffer(capacity=10)
        assert len(buf) == 0
        assert buf.capacity == 10

    def test_append_and_len(self):
        buf = RingBuffer(capacity=3)
        buf.append("a")
        buf.append("b")
        assert len(buf) == 2

    def test_capacity_eviction(self):
        buf = RingBuffer(capacity=3)
        buf.append("a")
        buf.append("b")
        buf.append("c")
        buf.append("d")
        assert len(buf) == 3
        assert buf[0] == "b"

    def test_peek(self):
        buf = RingBuffer(capacity=3)
        assert buf.peek() is None
        buf.append("a")
        assert buf.peek() == "a"

    def test_pop(self):
        buf = RingBuffer(capacity=3)
        buf.append("a")
        buf.append("b")
        assert buf.pop() == "a"
        assert len(buf) == 1

    def test_pop_empty(self):
        buf = RingBuffer(capacity=3)
        assert buf.pop() is None

    def test_extend(self):
        buf = RingBuffer(capacity=5)
        buf.extend(["a", "b", "c"])
        assert len(buf) == 3

    def test_clear(self):
        buf = RingBuffer(capacity=5)
        buf.append("a")
        buf.clear()
        assert len(buf) == 0

    def test_getitem(self):
        buf = RingBuffer(capacity=5)
        buf.append("a")
        buf.append("b")
        assert buf[0] == "a"
        assert buf[1] == "b"

    def test_iteration(self):
        buf = RingBuffer(capacity=5)
        buf.extend(["a", "b", "c"])
        assert list(buf) == ["a", "b", "c"]

    def test_contains(self):
        buf = RingBuffer(capacity=5)
        buf.append("a")
        assert "a" in buf
        assert "b" not in buf

    def test_is_full(self):
        buf = RingBuffer(capacity=2)
        assert not buf.is_full()
        buf.append("a")
        buf.append("b")
        assert buf.is_full()

    def test_to_list(self):
        buf = RingBuffer(capacity=3)
        buf.extend(["a", "b"])
        assert buf.to_list() == ["a", "b"]

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            RingBuffer(capacity=0)

    def test_repr(self):
        buf = RingBuffer(capacity=10)
        buf.append("a")
        assert "RingBuffer" in repr(buf)
