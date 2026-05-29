"""Tests for result_aggregator.py — Aggregates results from multiple workers.

Run: python3 -m pytest tests/test_result_aggregator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.result_aggregator import ResultAggregator


class TestResultAggregator:
    def test_create(self):
        agg = ResultAggregator(strategy="all")
        assert agg.is_complete() is False

    def test_expect(self):
        agg = ResultAggregator(strategy="all")
        agg.expect(3)
        assert agg.stats()["expected"] == 3

    def test_submit(self):
        agg = ResultAggregator(strategy="all")
        agg.submit("w-1", result=42)
        assert agg.get_result("w-1") == 42

    def test_submit_error(self):
        agg = ResultAggregator(strategy="all")
        agg.submit("w-1", error="timeout")
        assert agg.get_error("w-1") == "timeout"
        assert agg.get_result("w-1") is None

    def test_is_complete_all(self):
        agg = ResultAggregator(strategy="all")
        agg.expect(2)
        agg.submit("w-1", result=1)
        assert agg.is_complete() is False
        agg.submit("w-2", result=2)
        assert agg.is_complete() is True

    def test_is_complete_all_with_errors(self):
        agg = ResultAggregator(strategy="all")
        agg.expect(2)
        agg.submit("w-1", result=1)
        agg.submit("w-2", error="fail")
        assert agg.is_complete() is True

    def test_is_complete_any(self):
        agg = ResultAggregator(strategy="any")
        agg.submit("w-1", result=1)
        assert agg.is_complete() is True

    def test_is_complete_any_error(self):
        agg = ResultAggregator(strategy="any")
        agg.submit("w-1", error="fail")
        assert agg.is_complete() is True

    def test_is_complete_n_first(self):
        agg = ResultAggregator(strategy="n_first", n=2)
        agg.submit("w-1", result=1)
        assert agg.is_complete() is False
        agg.submit("w-2", result=2)
        assert agg.is_complete() is True

    def test_result_dict(self):
        agg = ResultAggregator(strategy="all")
        agg.expect(2)
        agg.submit("w-1", result=1)
        agg.submit("w-2", error="fail")
        result = agg.result()
        assert result["results"] == {"w-1": 1}
        assert result["errors"] == {"w-2": "fail"}
        assert result["complete"] is True
        assert result["success_count"] == 1
        assert result["error_count"] == 1

    def test_timeout(self):
        agg = ResultAggregator(strategy="all", timeout_sec=10, clock=lambda: 0)
        agg.expect(2)
        agg.submit("w-1", result=1)
        assert agg.is_timed_out() is False
        agg._clock = lambda: 15
        assert agg.is_timed_out() is True
        assert agg.result()["timed_out"] is True

    def test_sources(self):
        agg = ResultAggregator(strategy="all")
        agg.submit("a", result=1)
        agg.submit("b", error="fail")
        assert sorted(agg.sources()) == ["a", "b"]

    def test_successful_failed(self):
        agg = ResultAggregator(strategy="all")
        agg.submit("a", result=1)
        agg.submit("b", error="fail")
        assert agg.successful() == ["a"]
        assert agg.failed() == ["b"]

    def test_stats(self):
        agg = ResultAggregator(strategy="n_first", n=3, timeout_sec=60)
        agg.expect(5)
        agg.submit("a", result=1)
        agg.submit("b", error="fail")
        stats = agg.stats()
        assert stats["strategy"] == "n_first"
        assert stats["n"] == 3
        assert stats["timeout"] == 60
        assert stats["expected"] == 5
        assert stats["success_count"] == 1
        assert stats["error_count"] == 1
        assert stats["complete"] is False

    def test_repr(self):
        agg = ResultAggregator()
        assert "ResultAggregator" in repr(agg)
