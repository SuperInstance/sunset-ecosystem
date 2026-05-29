"""Tests for retry_policy.py — Retry policy with backoff strategies.

Run: python3 -m pytest tests/test_retry_policy.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.retry_policy import RetryPolicy


class TestRetryPolicy:
    def test_create(self):
        policy = RetryPolicy(max_retries=3, backoff="exponential", base_delay_sec=1.0)
        assert policy.stats()["max_retries"] == 3

    def test_next_delay_fixed(self):
        policy = RetryPolicy(backoff="fixed", base_delay_sec=2.0)
        assert policy.next_delay(0) == 2.0
        assert policy.next_delay(5) == 2.0

    def test_next_delay_exponential(self):
        policy = RetryPolicy(backoff="exponential", base_delay_sec=1.0)
        assert policy.next_delay(0) == 1.0
        assert policy.next_delay(1) == 2.0
        assert policy.next_delay(2) == 4.0
        assert policy.next_delay(3) == 8.0

    def test_next_delay_max_cap(self):
        policy = RetryPolicy(backoff="exponential", base_delay_sec=1.0, max_delay_sec=5.0)
        assert policy.next_delay(10) == 5.0

    def test_next_delay_jitter(self):
        policy = RetryPolicy(backoff="jitter", base_delay_sec=1.0, jitter_factor=0.5)
        delay = policy.next_delay(1)
        assert 1.0 <= delay <= 3.0  # base=2, jitter up to 50%

    def test_should_retry(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(0) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False
        assert policy.should_retry(4) is False

    def test_should_retry_error_filter(self):
        policy = RetryPolicy(max_retries=3, retryable_errors=["ValueError"])
        assert policy.should_retry(0, ValueError("bad")) is True
        assert policy.should_retry(0, TypeError("bad")) is False

    def test_should_retry_no_error_filter(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(0, TypeError("bad")) is True

    def test_execute_success(self):
        policy = RetryPolicy(max_retries=3)
        calls = []
        result = policy.execute(lambda: calls.append(1) or "success")
        assert result == "success"
        assert len(calls) == 1

    def test_execute_retry_then_success(self):
        policy = RetryPolicy(max_retries=3, backoff="fixed", base_delay_sec=0.01)
        attempts = []
        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("fail")
            return "success"
        result = policy.execute(flaky)
        assert result == "success"
        assert len(attempts) == 2

    def test_execute_exhausted(self):
        policy = RetryPolicy(max_retries=2, backoff="fixed", base_delay_sec=0.01)
        def always_fail():
            raise ValueError("fail")
        with pytest.raises(ValueError, match="fail"):
            policy.execute(always_fail)

    def test_stats(self):
        policy = RetryPolicy(max_retries=5, backoff="exponential", base_delay_sec=1.0, max_delay_sec=30.0, jitter_factor=0.2)
        policy.next_delay(0)
        policy.next_delay(1)
        stats = policy.stats()
        assert stats["max_retries"] == 5
        assert stats["backoff"] == "exponential"
        assert stats["base_delay"] == 1.0
        assert stats["max_delay"] == 30.0
        assert stats["jitter_factor"] == 0.2
        assert stats["total_delays"] == 3.0

    def test_repr(self):
        policy = RetryPolicy()
        assert "RetryPolicy" in repr(policy)
