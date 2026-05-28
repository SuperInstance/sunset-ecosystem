"""Tests for ring_buffer.c via ctypes bindings.

Compile first: gcc -shared -fPIC -O3 -o nerve/ring_buffer.so nerve/ring_buffer.c
Run: python3 -m pytest tests/test_ring_buffer.py -v --tb=short
"""
from __future__ import annotations

import pytest

from nerve.ring_buffer_wrapper import RingBuffer


class TestRingBufferBasics:
    def test_create_and_capacity(self):
        rb = RingBuffer(capacity=64, elem_size=8)
        assert rb.capacity == 64  # power of 2
        assert rb.size == 0
        assert rb.free == 64
        assert rb.is_empty
        assert not rb.is_full

    def test_push_pop_u64(self):
        rb = RingBuffer(capacity=16, elem_size=8)
        assert rb.push_u64(42)
        assert rb.size == 1
        assert rb.pop_u64() == 42
        assert rb.is_empty

    def test_push_pop_f64(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        assert rb.push_f64(3.14159)
        val = rb.pop_f64()
        assert val == pytest.approx(3.14159)

    def test_push_pop_i64(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        assert rb.push_i64(-999)
        assert rb.pop_i64() == -999

    def test_fifo_order(self):
        rb = RingBuffer(capacity=32, elem_size=8)
        for i in range(10):
            assert rb.push_u64(i)
        for i in range(10):
            assert rb.pop_u64() == i

    def test_full_rejection(self):
        rb = RingBuffer(capacity=4, elem_size=8)
        for i in range(4):
            assert rb.push_u64(i)
        assert rb.is_full
        assert not rb.push_u64(99)  # should reject
        assert rb.size == 4

    def test_empty_pop(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        assert rb.pop_u64() is None
        assert rb.pop_bytes() is None

    def test_peek(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        rb.push_u64(123)
        assert rb.peek_bytes() is not None
        assert rb.size == 1  # not consumed
        assert rb.pop_u64() == 123

    def test_wraparound(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        # Fill
        for i in range(8):
            rb.push_u64(i)
        # Drain half
        for i in range(4):
            assert rb.pop_u64() == i
        # Fill again — wraps around
        for i in range(8, 12):
            rb.push_u64(i)
        # FIFO order: remaining old items, then new items
        for expected in [4, 5, 6, 7, 8, 9, 10, 11]:
            assert rb.pop_u64() == expected

    def test_bulk_push_pop(self):
        rb = RingBuffer(capacity=64, elem_size=8)
        vals = list(range(20))
        pushed = rb.push_bulk_u64(vals)
        assert pushed == 20
        popped = rb.pop_bulk_u64(20)
        assert popped == vals

    def test_bulk_partial(self):
        rb = RingBuffer(capacity=4, elem_size=8)
        pushed = rb.push_bulk_u64([1, 2, 3, 4, 5])
        assert pushed == 4  # only 4 fit
        popped = rb.pop_bulk_u64(10)
        assert popped == [1, 2, 3, 4]

    def test_capacity_is_power_of_two(self):
        rb = RingBuffer(capacity=100, elem_size=8)
        cap = rb.capacity
        assert cap >= 100
        assert (cap & (cap - 1)) == 0  # power of 2

    def test_repr(self):
        rb = RingBuffer(capacity=8, elem_size=8)
        assert "RingBuffer" in repr(rb)
        assert "cap=8" in repr(rb)


class TestRingBufferStress:
    def test_fill_and_empty(self):
        rb = RingBuffer(capacity=1024, elem_size=8)
        for i in range(1024):
            assert rb.push_u64(i)
        assert rb.is_full
        for i in range(1024):
            assert rb.pop_u64() == i
        assert rb.is_empty

    def test_alternating_push_pop(self):
        rb = RingBuffer(capacity=16, elem_size=8)
        for _ in range(100):
            for i in range(8):
                rb.push_u64(i)
            for i in range(8):
                assert rb.pop_u64() == i
