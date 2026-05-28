"""Tests for circuit_breaker.py — Circuit breaker pattern.

Run: python3 -m pytest tests/test_circuit_breaker.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_create(self):
        cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True

    def test_successful_call(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb._stats.successes == 1

    def test_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb._stats.consecutive_failures == 2
        assert cb.state == CircuitState.CLOSED

    def test_trip(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN
        assert cb._stats.trips == 1

    def test_open_fast_fail(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        # Circuit is open, call returns None (fallback not provided)
        result = cb.call(lambda: 42)
        assert result is None
        assert cb.state == CircuitState.OPEN

    def test_open_with_fallback(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        result = cb.call(lambda: 42, fallback=lambda: "fallback")
        assert result == "fallback"

    def test_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        # After timeout, next call transitions to half-open
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        time.sleep(0.15)
        # Half-open, fail again -> back to open
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN

    def test_half_open_limit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, half_open_max_calls=1)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        time.sleep(0.15)
        # First half-open call succeeds -> circuit closes
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED
        # Now closed, normal calls work
        result2 = cb.call(lambda: 99)
        assert result2 == 99

    def test_custom_exception_types(self):
        cb = CircuitBreaker(exception_types=(RuntimeError,))
        # ValueError should NOT trigger circuit
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.CLOSED

    def test_stats(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.call(lambda: 1)
        cb.call(lambda: 2)
        stats = cb.stats()
        assert stats.state == CircuitState.CLOSED
        assert stats.successes == 2
        assert stats.failures == 0

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._stats.consecutive_failures == 0

    def test_repr(self):
        cb = CircuitBreaker(name="cb1")
        assert "CircuitBreaker" in repr(cb)
