"""Tests for rate_limiter.py — Token bucket and sliding window rate limiting.

Run: python3 -m pytest tests/test_rate_limiter.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_create(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        assert rl.stats()["rate"] == 10
        assert rl.stats()["burst"] == 20

    def test_allow(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        assert rl.allow("user-1", tokens=5) is True
        assert rl.allow("user-1", tokens=25) is False

    def test_consume(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20, clock=lambda: 0)
        assert rl.consume("user-1", tokens=5) is True
        assert rl.remaining("user-1") == 15.0
        assert rl.consume("user-1", tokens=20) is False

    def test_refill(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20, clock=lambda: 0)
        rl.consume("user-1", tokens=10)
        assert rl.remaining("user-1") == 10.0
        rl._clock = lambda: 1
        assert rl.remaining("user-1") == 20.0  # Refilled

    def test_per_key_tracking(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20, clock=lambda: 0)
        rl.consume("user-1", tokens=15)
        rl.consume("user-2", tokens=5)
        assert rl.remaining("user-1") == 5.0
        assert rl.remaining("user-2") == 15.0

    def test_reset(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        rl.consume("user-1", tokens=15)
        rl.reset("user-1")
        assert rl.remaining("user-1") == 20

    def test_reset_all(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        rl.consume("user-1", tokens=15)
        rl.consume("user-2", tokens=5)
        rl.reset_all()
        assert rl.keys() == []

    def test_keys(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        rl.consume("user-1", tokens=1)
        rl.consume("user-2", tokens=1)
        assert sorted(rl.keys()) == ["user-1", "user-2"]

    def test_stats(self):
        rl = RateLimiter(rate=10, per_sec=1, burst=20)
        rl.consume("user-1", tokens=5)
        stats = rl.stats()
        assert stats["keys"] == 1
        assert stats["total_consumed"] == 5

    def test_repr(self):
        rl = RateLimiter()
        assert "RateLimiter" in repr(rl)
