"""Tests for resource_allocator.py — Resource quota management.

Run: python3 -m pytest tests/test_resource_allocator.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.resource_allocator import ResourceAllocator, Allocation


class TestResourceAllocator:
    def test_create(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        assert ra.total_cpu == 8.0
        assert ra.total_memory == 32000.0

    def test_allocate(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        alloc = ra.allocate("agent-a", cpu=2.0, memory=4000.0)
        assert alloc.granted is True
        assert alloc.cpu == 2.0
        assert alloc.memory == 4000.0

    def test_allocate_cpu_exhausted(self):
        ra = ResourceAllocator(total_cpu=2.0, total_memory=32000.0)
        ra.allocate("agent-a", cpu=2.0, memory=1000.0)
        alloc = ra.allocate("agent-b", cpu=1.0, memory=1000.0)
        assert alloc.granted is False
        assert "CPU exhausted" in alloc.message

    def test_allocate_memory_exhausted(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=1000.0)
        ra.allocate("agent-a", cpu=1.0, memory=800.0)
        alloc = ra.allocate("agent-b", cpu=1.0, memory=300.0)
        assert alloc.granted is False
        assert "memory exhausted" in alloc.message

    def test_allocate_duplicate(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("agent-a", cpu=2.0, memory=4000.0)
        alloc = ra.allocate("agent-a", cpu=1.0, memory=1000.0)
        assert alloc.granted is False
        assert "already allocated" in alloc.message

    def test_release(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("agent-a", cpu=2.0, memory=4000.0)
        assert ra.release("agent-a") is True
        assert "agent-a" not in ra.agents()

    def test_release_nonexistent(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        assert ra.release("missing") is False

    def test_available(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("a", cpu=2.0, memory=4000.0)
        avail = ra.available()
        assert avail["cpu"] == pytest.approx(6.0)
        assert avail["memory"] == pytest.approx(28000.0)

    def test_utilization(self):
        ra = ResourceAllocator(total_cpu=4.0, total_memory=1000.0)
        ra.allocate("a", cpu=1.0, memory=250.0)
        util = ra.utilization()
        assert util["cpu"] == pytest.approx(0.25)
        assert util["memory"] == pytest.approx(0.25)

    def test_agent_usage(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("agent-x", cpu=2.0, memory=4000.0)
        usage = ra.agent_usage("agent-x")
        assert usage is not None
        assert usage.cpu == 2.0
        assert usage.memory == 4000.0

    def test_update_usage(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("agent-x", cpu=2.0, memory=4000.0)
        ra.update_usage("agent-x", cpu=3.0, memory=5000.0)
        usage = ra.agent_usage("agent-x")
        assert usage.cpu == 3.0
        assert usage.memory == 5000.0
        avail = ra.available()
        assert avail["cpu"] == pytest.approx(5.0)
        assert avail["memory"] == pytest.approx(27000.0)

    def test_evict_least_active(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("a", cpu=1.0, memory=1000.0)
        ra.allocate("b", cpu=1.0, memory=1000.0)
        # Update b's last_seen more recently
        ra.update_usage("b", cpu=1.0, memory=1000.0)
        evicted = ra.evict_least_active(count=1)
        assert evicted == ["a"]
        assert "a" not in ra.agents()

    def test_evict_by_memory(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("a", cpu=1.0, memory=5000.0)
        ra.allocate("b", cpu=1.0, memory=3000.0)
        ra.allocate("c", cpu=1.0, memory=1000.0)
        evicted = ra.evict_by_memory(target_mb=4000.0)
        # Should evict a (5000) to free enough
        assert "a" in evicted

    def test_report(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("a", cpu=2.0, memory=4000.0)
        r = ra.report()
        assert r["agents"] == 1
        assert r["used_cpu"] == 2.0
        assert r["used_memory"] == 4000.0

    def test_repr(self):
        ra = ResourceAllocator(total_cpu=8.0, total_memory=32000.0)
        ra.allocate("a", cpu=2.0, memory=4000.0)
        assert "ResourceAllocator" in repr(ra)
