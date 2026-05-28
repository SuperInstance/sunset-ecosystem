"""Tests for quota_manager.py — Resource quotas per tenant.

Run: python3 -m pytest tests/test_quota_manager.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.quota_manager import QuotaManager, QuotaExceeded


class TestQuotaManager:
    def test_create(self):
        qm = QuotaManager()
        assert qm.tenants() == []

    def test_set_quota(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100, window=3600)
        quota = qm.get_quota("t1", "api_calls")
        assert quota is not None
        assert quota.limit == 100
        assert quota.window == 3600

    def test_record_usage(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        assert qm.record_usage("t1", "api_calls", 50) is True
        assert qm.record_usage("t1", "api_calls", 50) is True
        assert qm.record_usage("t1", "api_calls", 1) is False  # Over limit

    def test_check(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        assert qm.check("t1", "api_calls", 50) is True
        qm.record_usage("t1", "api_calls", 90)
        assert qm.check("t1", "api_calls", 15) is False

    def test_require(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        qm.require("t1", "api_calls", 50)
        with pytest.raises(QuotaExceeded):
            qm.require("t1", "api_calls", 60)

    def test_window_reset(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100, window=0.1)
        qm.record_usage("t1", "api_calls", 100)
        assert qm.check("t1", "api_calls", 1) is False
        time.sleep(0.15)
        assert qm.check("t1", "api_calls", 1) is True  # Window reset

    def test_burst(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100, burst=20)
        qm.record_usage("t1", "api_calls", 110)
        assert qm.check("t1", "api_calls", 5) is True  # Within burst
        assert qm.check("t1", "api_calls", 15) is False  # Over burst

    def test_usage(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        qm.record_usage("t1", "api_calls", 30)
        assert qm.usage("t1", "api_calls") == 30

    def test_remaining(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        qm.record_usage("t1", "api_calls", 30)
        assert qm.remaining("t1", "api_calls") == 70

    def test_no_quota(self):
        qm = QuotaManager()
        assert qm.check("t1", "anything", 1000) is True
        assert qm.remaining("t1", "anything") == float("inf")

    def test_reset(self):
        qm = QuotaManager()
        qm.set_quota("t1", "api_calls", limit=100)
        qm.record_usage("t1", "api_calls", 90)
        assert qm.reset("t1", "api_calls") is True
        assert qm.usage("t1", "api_calls") == 0.0
        assert qm.reset("missing", "api_calls") is False

    def test_tenants_and_resources(self):
        qm = QuotaManager()
        qm.set_quota("t1", "cpu", limit=4)
        qm.set_quota("t1", "memory", limit=16)
        qm.set_quota("t2", "cpu", limit=2)
        assert sorted(qm.tenants()) == ["t1", "t2"]
        assert sorted(qm.resources("t1")) == ["cpu", "memory"]

    def test_stats(self):
        qm = QuotaManager()
        qm.set_quota("t1", "cpu", limit=4)
        qm.set_quota("t2", "cpu", limit=2)
        qm.set_quota("t2", "mem", limit=8)
        stats = qm.stats()
        assert stats["tenants"] == 2
        assert stats["total_quotas"] == 3

    def test_repr(self):
        qm = QuotaManager()
        assert "QuotaManager" in repr(qm)