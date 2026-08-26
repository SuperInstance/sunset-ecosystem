"""Tests for throttle.py — Rate throttling with burst and sliding window.

Run: python3 -m pytest tests/test_throttle.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from fleet.throttle import Throttle


class TestThrottle:
    def test_create(self):
        throttle = Throttle(rate=10, burst=5)
        assert throttle.stats()["rate"] == 10
        assert throttle.stats()["burst"] == 5

    def test_acquire_within_burst(self):
        throttle = Throttle(rate=10, burst=3)
        assert throttle.acquire() is True
        assert throttle.acquire() is True
        assert throttle.acquire() is True

    def test_acquire_rate_limited(self):
        throttle = Throttle(rate=1, burst=1)
        assert throttle.acquire() is True
        assert throttle.acquire() is False

    def test_token_refill(self):
        throttle = Throttle(rate=20, burst=1)
        assert throttle.acquire() is True
        assert throttle.acquire() is False
        time.sleep(0.06)
        assert throttle.acquire() is True

    def test_stats(self):
        throttle = Throttle(rate=10, burst=2)
        throttle.acquire()
        throttle.acquire()
        throttle.acquire()
        stats = throttle.stats()
        assert stats["allowed"] == 2
        assert stats["denied"] == 1

    def test_wait_time(self):
        throttle = Throttle(rate=1, burst=1)
        throttle.acquire()
        wait = throttle.wait_time()
        assert wait > 0.9

    def test_repr(self):
        throttle = Throttle(rate=10, burst=5)
        assert "Throttle" in repr(throttle)
