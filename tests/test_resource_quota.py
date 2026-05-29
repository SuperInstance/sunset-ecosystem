"""Tests for resource_quota.py — Resource quota management and enforcement.

Run: python3 -m pytest tests/test_resource_quota.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.resource_quota import ResourceQuota


class TestResourceQuota:
    def test_create(self):
        quota = ResourceQuota(tenant="team-a")
        assert quota.tenant == "team-a"

    def test_set_limit(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        assert quota.get_limit("cpu") == 100

    def test_remove_limit(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        assert quota.remove_limit("cpu") is True
        assert quota.get_limit("cpu") is None
        assert quota.remove_limit("missing") is False

    def test_request(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        assert quota.request("cpu", 50) is True
        assert quota.usage("cpu") == 50

    def test_request_exceeds(self):
        quota = ResourceQuota(burst_ratio=1.0)
        quota.set_limit("cpu", 100)
        assert quota.request("cpu", 60) is True
        assert quota.request("cpu", 50) is False  # 110 > 100

    def test_request_burst(self):
        quota = ResourceQuota(burst_ratio=1.2)
        quota.set_limit("cpu", 100)
        assert quota.request("cpu", 110) is True  # Within burst (110 <= 120)
        assert quota.request("cpu", 11) is False  # 121 > 120 (exceeds burst)

    def test_release(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        quota.request("cpu", 50)
        quota.release("cpu", 20)
        assert quota.usage("cpu") == 30

    def test_release_no_negative(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        quota.request("cpu", 50)
        quota.release("cpu", 100)
        assert quota.usage("cpu") == 0

    def test_available(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        quota.request("cpu", 30)
        assert quota.available("cpu") == 70

    def test_available_no_limit(self):
        quota = ResourceQuota()
        quota.request("cpu", 30)
        assert quota.available("cpu") == float("inf")

    def test_resources(self):
        quota = ResourceQuota()
        quota.set_limit("cpu", 100)
        quota.set_limit("memory", 1024)
        assert sorted(quota.resources()) == ["cpu", "memory"]

    def test_is_exceeded(self):
        quota = ResourceQuota(burst_ratio=2.0)
        quota.set_limit("cpu", 100)
        assert quota.is_exceeded("cpu") is False
        quota.request("cpu", 60)
        quota.request("cpu", 50)  # 110 total, allowed by burst (200)
        assert quota.is_exceeded("cpu") is True  # 110 > 100 base limit

    def test_stats(self):
        quota = ResourceQuota(tenant="team-a")
        quota.set_limit("cpu", 100)
        quota.request("cpu", 50)
        stats = quota.stats()
        assert stats["tenant"] == "team-a"
        assert stats["resources"] == 1
        assert stats["used"]["cpu"] == 50

    def test_repr(self):
        quota = ResourceQuota()
        assert "ResourceQuota" in repr(quota)
