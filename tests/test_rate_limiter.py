"""Tests for rate_limiter.py — Token bucket rate limiter.

Run: python3 -m pytest tests/test_rate_limiter.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    def test_allow_basic(self):
        b = TokenBucket(rate=10.0, burst=5.0)
        assert b.allow() is True
        assert b.tokens == 4.0

    def test_allow_consumes_burst(self):
        b = TokenBucket(rate=10.0, burst=5.0)
        for _ in range(5):
            assert b.allow() is True
        assert b.allow() is False  # burst exhausted

    def test_refill_over_time(self):
        b = TokenBucket(rate=10.0, burst=1.0)
        assert b.allow() is True
        assert b.allow() is False
        time.sleep(0.15)  # wait for ~1.5 tokens at 10/sec
        assert b.allow() is True

    def test_wait_time(self):
        b = TokenBucket(rate=10.0, burst=1.0)
        b.allow()  # consume the 1 token
        assert b.wait_time > 0.0
        assert b.wait_time < 0.15

    def test_state(self):
        b = TokenBucket(rate=10.0, burst=5.0)
        b.allow()
        s = b.state()
        assert s["tokens"] == 4.0
        assert s["allowed"] == 1
        assert s["denied"] == 0

    def test_denied_count(self):
        b = TokenBucket(rate=1.0, burst=1.0)
        b.allow()
        assert b.allow() is False
        assert b.state()["denied"] == 1

    def test_initial_tokens(self):
        b = TokenBucket(rate=10.0, burst=5.0, tokens=3.0)
        assert b.allow() is True
        assert b.tokens == pytest.approx(2.0, abs=0.01)


class TestRateLimiter:
    def test_create(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        assert rl.allow() is True

    def test_per_key_isolation(self):
        rl = RateLimiter(rate=1.0, burst=1.0)
        assert rl.allow("key-a") is True
        assert rl.allow("key-b") is True
        assert rl.allow("key-a") is False
        assert rl.allow("key-b") is False

    def test_global_limit(self):
        rl = RateLimiter(rate=100.0, burst=100.0, global_rate=1.0, global_burst=1.0)
        assert rl.allow() is True
        assert rl.allow() is False  # global exhausted

    def test_wait_time(self):
        rl = RateLimiter(rate=10.0, burst=1.0)
        rl.allow("key")
        assert rl.wait_time("key") > 0.0

    def test_state(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        rl.allow("a")
        rl.allow("b")
        s = rl.state()
        assert s["keys"] == 2
        assert s["total_allowed"] == 2

    def test_state_key(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        rl.allow("x")
        s = rl.state("x")
        assert s["allowed"] == 1

    def test_state_missing_key(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        s = rl.state("missing")
        assert "error" in s

    def test_reset_key(self):
        rl = RateLimiter(rate=1.0, burst=1.0)
        rl.allow("key")
        rl.reset("key")
        assert rl.allow("key") is True

    def test_reset_all(self):
        rl = RateLimiter(rate=1.0, burst=1.0)
        rl.allow("a")
        rl.allow("b")
        rl.reset()
        assert rl.allow("a") is True
        assert rl.allow("b") is True

    def test_global_reset(self):
        rl = RateLimiter(rate=100.0, burst=100.0, global_rate=1.0, global_burst=1.0)
        rl.allow()
        rl.allow()  # denied by global
        rl.reset()
        assert rl.allow() is True

    def test_repr(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        assert "RateLimiter" in repr(rl)

    def test_cost(self):
        rl = RateLimiter(rate=10.0, burst=5.0)
        assert rl.allow(cost=4.0) is True  # 1 token left
        assert rl.allow(cost=1.0) is True  # 0 tokens left
        assert rl.allow(cost=5.0) is False
