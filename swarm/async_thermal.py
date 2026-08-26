"""Async thermal budget manager — asyncio-compatible cost-based allocation.

Manages compute budgets across GPU, CPU, iGPU, and NPU devices using
async/await primitives. Cost-based (not slot-based) to support
heterogeneous workloads.

Default budgets mirror the sync thermal module (1 slot = 1.0 cost):
    GPU: 9.0, CPU: 36.0, iGPU: 14.0, NPU: 6.0 (total: 65.0)
"""

from __future__ import annotations

__all__ = [
    "AsyncDeviceBudget",
    "AsyncThermalBudget",
    "ThermalThrottled",
]

import asyncio
from dataclasses import dataclass
from typing import Optional

from swarm.thermal import DeviceType, DEFAULT_BUDGETS


class ThermalThrottled(Exception):
    """Raised when a thermal allocation request exceeds available budget."""


@dataclass
class AsyncDeviceBudget:
    """Budget for a single device type — async-safe companion to DeviceBudget.

    Attributes:
        device_type: Which device.
        max_cost: Maximum thermal cost the device can absorb.
        current_cost: Currently allocated thermal cost.
    """

    device_type: DeviceType
    max_cost: float
    current_cost: float = 0.0

    @property
    def available(self) -> float:
        """Thermal headroom still available."""
        return max(0.0, self.max_cost - self.current_cost)

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction [0, 1]."""
        if self.max_cost == 0.0:
            return 0.0
        return self.current_cost / self.max_cost

    def __repr__(self) -> str:
        return (
            f"AsyncDeviceBudget({self.device_type.value}, "
            f"cost={self.current_cost:.1f}/{self.max_cost:.1f}, "
            f"util={self.utilization:.0%})"
        )


class AsyncThermalBudget:
    """Manages thermal costs across all compute devices, async-safe.

    All mutations acquire an internal asyncio.Lock. Optional backpressure
    via asyncio.Condition — allocate(wait=True) will suspend until enough
    budget is freed by a matching release().

    Args:
        budgets: Optional per-device max costs. Defaults to slot-based
            DEFAULT_BUDGETS (converted 1 slot = 1.0 cost).
    """

    def __init__(
        self,
        budgets: dict[DeviceType, float] | None = None,
    ) -> None:
        config = (
            budgets
            if budgets is not None
            else {dt: float(max_agents) for dt, max_agents in DEFAULT_BUDGETS.items()}
        )
        self._devices: dict[DeviceType, AsyncDeviceBudget] = {
            dt: AsyncDeviceBudget(device_type=dt, max_cost=max_cost)
            for dt, max_cost in config.items()
        }
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    def __repr__(self) -> str:
        total = sum(d.current_cost for d in self._devices.values())
        max_total = sum(d.max_cost for d in self._devices.values())
        return f"AsyncThermalBudget(cost={total:.1f}/{max_total:.1f})"

    @property
    def total_max(self) -> float:
        """Total maximum cost across all devices."""
        return sum(d.max_cost for d in self._devices.values())

    @property
    def total_current(self) -> float:
        """Total currently allocated cost."""
        return sum(d.current_cost for d in self._devices.values())

    def device_budget(self, device: DeviceType) -> AsyncDeviceBudget:
        """Get the budget for a specific device."""
        return self._devices[device]

    async def check_budget(self, device: DeviceType, cost: float) -> bool:
        """Non-blocking thermal check.

        Args:
            device: The target device type.
            cost: The thermal cost to check.

        Returns:
            True if the cost fits in available budget.
        """
        async with self._lock:
            db = self._devices.get(device)
            if db is None:
                return False
            return db.current_cost + cost <= db.max_cost

    async def allocate(
        self,
        device: DeviceType,
        cost: float,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Async thermal allocation with optional backpressure.

        Args:
            device: Target device type.
            cost: Thermal cost to allocate.
            wait: If True, suspend until enough budget becomes available
                rather than failing immediately.
            timeout: Max seconds to wait when wait=True. None means
                wait indefinitely.

        Returns:
            True if allocation succeeded.

        Raises:
            ThermalThrottled: If budget exhausted and wait=False, or if
                timeout expires while waiting.
        """
        async with self._condition:
            db = self._devices.get(device)
            if db is None:
                raise ThermalThrottled(f"Device {device.value!r} not configured")

            if wait:
                # Backpressure: suspend until enough budget is available
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: db.current_cost + cost <= db.max_cost
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    raise ThermalThrottled(
                        f"Timeout waiting for {device.value} budget after {timeout}s"
                    )

            if db.current_cost + cost > db.max_cost:
                raise ThermalThrottled(
                    f"Device {device.value} budget exhausted: "
                    f"need {cost:.1f}, available {db.available:.1f}"
                )

            db.current_cost += cost
            return True

    async def release(self, device: DeviceType, cost: float) -> bool:
        """Free thermal cost previously allocated to a device.

        Notifies all waiters so back-pressured allocations can proceed.

        Args:
            device: The device to release cost from.
            cost: Amount of cost to release.

        Returns:
            True if release succeeded (device exists), False otherwise.
        """
        async with self._condition:
            db = self._devices.get(device)
            if db is None:
                return False
            db.current_cost = max(0.0, db.current_cost - cost)
            self._condition.notify_all()
            return True

    async def thermal_headroom(self) -> float:
        """Total utilization across all devices (0 = empty, 1 = full)."""
        async with self._lock:
            max_total = self.total_max
            if max_total == 0.0:
                return 0.0
            return self.total_current / max_total

    async def can_breed(self, threshold: float = 0.8) -> bool:
        """Whether there's enough headroom to breed."""
        return await self.thermal_headroom() < threshold

    async def reset(self) -> None:
        """Release all costs and reset all device budgets."""
        async with self._condition:
            for db in self._devices.values():
                db.current_cost = 0.0
            self._condition.notify_all()
