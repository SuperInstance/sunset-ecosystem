"""Tests for circuit_breaker.py — Circuit breaker pattern.

Run: python3 -m pytest tests/test_circuit_breaker.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.circuit_breaker import CircuitBreaker, CircuitBreakerOpen


class TestCircuitBreaker:
    def test_create(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=10)
        assert cb.state() == "closed"
        assert cb.is_closed() is True

    def test_successful_call(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.stats()["total_calls"] == 1

    def test_failure_counting(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ZeroDivisionError):
                cb.call(lambda: 1 / 0)
        assert cb.stats()["failures"] == 2
        assert cb.state() == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ZeroDivisionError):
                cb.call(lambda: 1 / 0)
        assert cb.state() == "open"
        assert cb.is_open() is True

    def test_open_raises(self):
        cb = CircuitBreaker()
        cb.open()
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "ok")

    def test_recovery_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10, clock=lambda: 0)
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        assert cb.state() == "open"
        cb._clock = lambda: 20
        # Transition to half-open, call succeeds
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state() == "half_open"

    def test_half_open_closes_on_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10, half_open_max_calls=2, clock=lambda: 0)
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        cb._clock = lambda: 20
        cb.call(lambda: "ok")
        cb.call(lambda: "ok")
        assert cb.state() == "closed"

    def test_half_open_reopens_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10, clock=lambda: 0)
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        cb._clock = lambda: 20
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        assert cb.state() == "open"

    def test_manual_close(self):
        cb = CircuitBreaker()
        cb.open()
        assert cb.is_open() is True
        cb.close()
        assert cb.is_closed() is True
        result = cb.call(lambda: "ok")
        assert result == "ok"

    def test_half_open_limit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=10, half_open_max_calls=1, clock=lambda: 0)
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        cb._clock = lambda: 20
        cb.call(lambda: "ok")  # Succeeds and closes (1 success >= half_open_max=1)
        assert cb.state() == "closed"
        # Circuit is now closed, next call should succeed too
        result = cb.call(lambda: "ok")
        assert result == "ok"

    def test_stats(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.call(lambda: "ok")
        with pytest.raises(ZeroDivisionError):
            cb.call(lambda: 1 / 0)
        stats = cb.stats()
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1

    def test_repr(self):
        cb = CircuitBreaker()
        assert "CircuitBreaker" in repr(cb)
