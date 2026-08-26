"""Tests for async thermal budget manager."""

from __future__ import annotations

import asyncio

import pytest

from swarm.async_thermal import (
    AsyncDeviceBudget,
    AsyncThermalBudget,
    ThermalThrottled,
)
from swarm.thermal import DeviceType, DEFAULT_BUDGETS


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture
def async_budget():
    """Return an AsyncThermalBudget with default (slot-based) budgets."""
    return AsyncThermalBudget()


@pytest.fixture
def small_async_budget():
    """Return an AsyncThermalBudget with tiny budgets for edge-case testing."""
    return AsyncThermalBudget(
        {
            DeviceType.GPU: 2.0,
            DeviceType.CPU: 4.0,
        }
    )


# ── tests ───────────────────────────────────────────────────


class TestAsyncDeviceBudget:
    """Unit tests for AsyncDeviceBudget dataclass."""

    def test_available(self):
        db = AsyncDeviceBudget(DeviceType.GPU, max_cost=10.0, current_cost=3.0)
        assert db.available == 7.0

    def test_utilization(self):
        db = AsyncDeviceBudget(DeviceType.CPU, max_cost=10.0, current_cost=5.0)
        assert db.utilization == 0.5

    def test_utilization_zero_max(self):
        db = AsyncDeviceBudget(DeviceType.NPU, max_cost=0.0, current_cost=0.0)
        assert db.utilization == 0.0

    def test_repr(self):
        db = AsyncDeviceBudget(DeviceType.IGPU, max_cost=14.0, current_cost=7.0)
        r = repr(db)
        assert "igpu" in r
        assert "7.0/14.0" in r
        assert "50%" in r


@pytest.mark.asyncio
class TestAsyncThermalBudget:
    """Async tests for AsyncThermalBudget allocation logic."""

    async def test_check_budget_sufficient(self, async_budget):
        """check_budget returns True when cost fits."""
        ok = await async_budget.check_budget(DeviceType.GPU, 5.0)
        assert ok is True

    async def test_check_budget_insufficient(self, async_budget):
        """check_budget returns False when cost exceeds available."""
        # Fill GPU to 9/9
        await async_budget.allocate(DeviceType.GPU, 9.0)
        ok = await async_budget.check_budget(DeviceType.GPU, 1.0)
        assert ok is False

    async def test_allocate_success(self, small_async_budget):
        """allocate succeeds and updates current_cost."""
        ok = await small_async_budget.allocate(DeviceType.GPU, 1.5)
        assert ok is True
        db = small_async_budget.device_budget(DeviceType.GPU)
        assert db.current_cost == 1.5

    async def test_allocate_raises_when_exhausted(self, small_async_budget):
        """allocate raises ThermalThrottled when budget exhausted."""
        await small_async_budget.allocate(DeviceType.GPU, 2.0)
        with pytest.raises(ThermalThrottled):
            await small_async_budget.allocate(DeviceType.GPU, 0.1)

    async def test_release_frees_budget(self, small_async_budget):
        """release frees cost so re-allocation succeeds."""
        await small_async_budget.allocate(DeviceType.GPU, 2.0)
        released = await small_async_budget.release(DeviceType.GPU, 2.0)
        assert released is True
        # Should now be able to allocate again
        ok = await small_async_budget.allocate(DeviceType.GPU, 1.5)
        assert ok is True

    async def test_release_unknown_device(self, small_async_budget):
        """release returns False for an unconfigured device."""
        released = await small_async_budget.release(DeviceType.NPU, 1.0)
        assert released is False

    async def test_thermal_headroom(self, small_async_budget):
        """thermal_headroom reflects total utilization."""
        # Fill half of CPU (4.0 max)
        await small_async_budget.allocate(DeviceType.CPU, 2.0)
        headroom = await small_async_budget.thermal_headroom()
        assert headroom == 2.0 / 6.0  # 2 of 6 total

    async def test_can_breed(self, small_async_budget):
        """can_breed respects threshold."""
        # Empty → should breed
        assert await small_async_budget.can_breed(threshold=0.8) is True
        # Fill to 100%
        await small_async_budget.allocate(DeviceType.GPU, 2.0)
        await small_async_budget.allocate(DeviceType.CPU, 4.0)
        assert await small_async_budget.can_breed(threshold=0.8) is False

    async def test_reset(self, small_async_budget):
        """reset clears all costs."""
        await small_async_budget.allocate(DeviceType.GPU, 1.0)
        await small_async_budget.allocate(DeviceType.CPU, 2.0)
        await small_async_budget.reset()
        assert small_async_budget.total_current == 0.0

    async def test_backpressure_wait(self, small_async_budget):
        """allocate(wait=True) suspends until release frees budget."""
        # Fill GPU completely
        await small_async_budget.allocate(DeviceType.GPU, 2.0)

        async def delayed_release():
            await asyncio.sleep(0.05)
            await small_async_budget.release(DeviceType.GPU, 1.5)

        # Start the release in background
        asyncio.create_task(delayed_release())

        # This should wait, then succeed once release runs
        ok = await small_async_budget.allocate(DeviceType.GPU, 1.0, wait=True)
        assert ok is True
        db = small_async_budget.device_budget(DeviceType.GPU)
        assert db.current_cost == 1.5  # 2.0 - 1.5 + 1.0

    async def test_backpressure_timeout(self, small_async_budget):
        """allocate(wait=True, timeout=...) raises ThermalThrottled on timeout."""
        # Fill GPU completely
        await small_async_budget.allocate(DeviceType.GPU, 2.0)

        with pytest.raises(ThermalThrottled, match="Timeout"):
            await small_async_budget.allocate(
                DeviceType.GPU, 1.0, wait=True, timeout=0.02
            )

    async def test_default_budgets_match_sync(self):
        """Default async budgets mirror sync DEFAULT_BUDGETS (1 slot = 1.0 cost)."""
        async_budget = AsyncThermalBudget()
        for device_type, max_agents in DEFAULT_BUDGETS.items():
            db = async_budget.device_budget(device_type)
            assert db.max_cost == float(max_agents)

    async def test_repr(self, small_async_budget):
        r = repr(small_async_budget)
        assert "AsyncThermalBudget" in r
        assert "cost=" in r
