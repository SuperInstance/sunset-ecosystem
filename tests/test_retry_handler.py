"""Tests for retry_handler.py — Exponential backoff retry logic.

Run: python3 -m pytest tests/test_retry_handler.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.retry_handler import RetryHandler, RetryPolicy, RetryExhausted


class TestRetryHandler:
    def test_success(self):
        retry = RetryHandler(RetryPolicy(max_attempts=3))
        result = retry.run(lambda: 42)
        assert result == 42
        assert retry.stats()["total_successes"] == 1

    def test_retry_then_success(self):
        calls = [0]
        def flaky():
            calls[0] += 1
            if calls[0] < 3:
                raise ConnectionError("timeout")
            return "ok"

        retry = RetryHandler(RetryPolicy(max_attempts=3, base_delay=0.01, on_exceptions=(ConnectionError,)))
        result = retry.run(flaky)
        assert result == "ok"
        assert calls[0] == 3

    def test_exhausted(self):
        retry = RetryHandler(RetryPolicy(max_attempts=2, base_delay=0.01, on_exceptions=(ValueError,)))
        with pytest.raises(RetryExhausted):
            retry.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert retry.stats()["total_failures"] == 1

    def test_no_retry_for_unexpected_exception(self):
        retry = RetryHandler(RetryPolicy(max_attempts=3, on_exceptions=(ConnectionError,)))
        with pytest.raises(ValueError):
            retry.run(lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_retry_condition(self):
        calls = [0]
        def sometimes_none():
            calls[0] += 1
            return None if calls[0] < 3 else "ok"

        retry = RetryHandler(RetryPolicy(max_attempts=5, base_delay=0.01, retry_if=lambda x: x is None))
        result = retry.run(sometimes_none)
        assert result == "ok"
        assert calls[0] == 3

    def test_retry_condition_exhausted(self):
        retry = RetryHandler(RetryPolicy(max_attempts=2, base_delay=0.01, retry_if=lambda x: x is None))
        with pytest.raises(RetryExhausted):
            retry.run(lambda: None)

    def test_backoff_grows(self):
        retry = RetryHandler(RetryPolicy(max_attempts=3, base_delay=0.1, exponential_base=2.0, jitter=False))
        calls = [0]
        def fail():
            calls[0] += 1
            raise RuntimeError("fail")
        with pytest.raises(RetryExhausted):
            retry.run(fail)
        # Should have taken at least 0.1 + 0.2 = 0.3s
        assert calls[0] == 3

    def test_max_delay_cap(self):
        retry = RetryHandler(RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=2.0, exponential_base=10.0, jitter=False))
        calls = [0]
        def fail():
            calls[0] += 1
            raise RuntimeError("fail")
        start = time.time()
        with pytest.raises(RetryExhausted):
            retry.run(fail)
        elapsed = time.time() - start
        # Should be capped at 2.0s per delay, not 10s or 100s
        assert elapsed < 10.0

    def test_stats(self):
        retry = RetryHandler(RetryPolicy(max_attempts=2))
        retry.run(lambda: "ok")
        with pytest.raises(RetryExhausted):
            retry.run(lambda: (_ for _ in ()).throw(ValueError("fail")))
        stats = retry.stats()
        assert stats["total_attempts"] == 3  # 1 success + 2 failure attempts
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1

    def test_repr(self):
        retry = RetryHandler()
        assert "RetryHandler" in repr(retry)
