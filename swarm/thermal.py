"""Thermal budget manager — device-aware agent slot allocation.

Manages compute budgets across GPU, CPU, iGPU, and NPU devices.
Thread-safe: all mutations go through a threading.Lock.

Default budgets:
    GPU: 9 agents, CPU: 36 agents, iGPU: 14 agents, NPU: 6 agents (total: 65)
"""

from __future__ import annotations

__all__ = [
    "DeviceBudget",
    "DeviceType",
    "ThermalBudget",
]

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceType(Enum):
    """Supported compute device types."""

    GPU = "gpu"
    CPU = "cpu"
    IGPU = "igpu"
    NPU = "npu"


# Default max agents per device type
DEFAULT_BUDGETS: dict[DeviceType, int] = {
    DeviceType.GPU: 9,
    DeviceType.CPU: 36,
    DeviceType.IGPU: 14,
    DeviceType.NPU: 6,
}


@dataclass
class DeviceBudget:
    """Budget for a single device type.

    Attributes:
        device_type: Which device.
        max_agents: Maximum concurrent agents.
        current_agents: Currently allocated agents.
    """
    device_type: DeviceType
    max_agents: int
    current_agents: int = 0

    @property
    def available(self) -> int:
        """Slots still available."""
        return max(0, self.max_agents - self.current_agents)

    @property
    def utilization(self) -> float:
        """Current utilization as a fraction [0, 1]."""
        if self.max_agents == 0:
            return 0.0
        return self.current_agents / self.max_agents

    def __repr__(self) -> str:
        return (
            f"DeviceBudget({self.device_type.value}, "
            f"{self.current_agents}/{self.max_agents}, "
            f"util={self.utilization:.0%})"
        )


class ThermalBudget:
    """Manages agent slots across all compute devices.

    Thread-safe. All mutations acquire an internal lock.

    Args:
        budgets: Optional per-device budgets. Defaults to DEFAULT_BUDGETS.
    """

    def __init__(
        self,
        budgets: dict[DeviceType, int] | None = None,
    ) -> None:
        config = budgets if budgets is not None else DEFAULT_BUDGETS
        self._devices: dict[DeviceType, DeviceBudget] = {
            dt: DeviceBudget(device_type=dt, max_agents=max_agents)
            for dt, max_agents in config.items()
        }
        self._allocations: dict[str, DeviceType] = {}  # agent_id → device
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        total = sum(d.current_agents for d in self._devices.values())
        max_total = sum(d.max_agents for d in self._devices.values())
        return f"ThermalBudget({total}/{max_total} agents)"

    @property
    def total_max(self) -> int:
        """Total maximum agents across all devices."""
        return sum(d.max_agents for d in self._devices.values())

    @property
    def total_current(self) -> int:
        """Total currently allocated agents."""
        return sum(d.current_agents for d in self._devices.values())

    def device_budget(self, device: DeviceType) -> DeviceBudget:
        """Get the budget for a specific device."""
        return self._devices[device]

    def can_spawn(self, device: DeviceType) -> bool:
        """Check if there's room on the given device for another agent.

        Args:
            device: The target device type.

        Returns:
            True if at least one slot is available.
        """
        with self._lock:
            db = self._devices.get(device)
            if db is None:
                return False
            return db.current_agents < db.max_agents

    def allocate(self, agent_id: str, device: DeviceType) -> bool:
        """Assign an agent to a device.

        Args:
            agent_id: Unique agent identifier.
            device: Target device type.

        Returns:
            True if allocation succeeded, False if no room.

        Raises:
            ValueError: If agent_id is already allocated somewhere.
        """
        with self._lock:
            if agent_id in self._allocations:
                raise ValueError(
                    f"Agent {agent_id!r} already allocated to "
                    f"{self._allocations[agent_id].value}"
                )
            db = self._devices.get(device)
            if db is None or db.current_agents >= db.max_agents:
                return False
            db.current_agents += 1
            self._allocations[agent_id] = device
            return True

    def release(self, agent_id: str) -> bool:
        """Free a device slot previously allocated to an agent.

        Args:
            agent_id: The agent to release.

        Returns:
            True if the agent was found and released, False otherwise.
        """
        with self._lock:
            device = self._allocations.pop(agent_id, None)
            if device is None:
                return False
            self._devices[device].current_agents -= 1
            return True

    def get_device(self, agent_id: str) -> Optional[DeviceType]:
        """Get the device an agent is allocated to.

        Returns None if the agent is not allocated.
        """
        with self._lock:
            return self._allocations.get(agent_id)

    def parent_sacrifice_before_spawn(
        self,
        parent_id: str,
        child_device: DeviceType,
    ) -> bool:
        """If no room on child_device, retire parent to make room.

        Tries to allocate on child_device first. If that fails and
        the parent is currently allocated, releases the parent and
        tries again.

        Args:
            parent_id: The parent agent that may be sacrificed.
            child_device: Where the child wants to spawn.

        Returns:
            True if the child slot is now available (either directly
            or after parent sacrifice), False if impossible.
        """
        with self._lock:
            db = self._devices.get(child_device)
            if db is None:
                return False

            # Direct allocation possible
            if db.current_agents < db.max_agents:
                return True

            # Try sacrificing parent
            parent_device = self._allocations.get(parent_id)
            if parent_device is None:
                return False

            # Release parent
            self._devices[parent_device].current_agents -= 1
            del self._allocations[parent_id]

            # Now try child device again
            if db.current_agents < db.max_agents:
                return True

            # Still no room — put parent back (undo sacrifice)
            self._devices[parent_device].current_agents += 1
            self._allocations[parent_id] = parent_device
            return False

    def thermal_headroom(self) -> float:
        """Total utilization across all devices (0 = empty, 1 = full)."""
        with self._lock:
            max_total = self.total_max
            if max_total == 0:
                return 0.0
            return self.total_current / max_total

    def can_breed(self, threshold: float = 0.8) -> bool:
        """Whether there's enough headroom to breed."""
        return self.thermal_headroom() < threshold

    def reset(self) -> None:
        """Release all agents and reset all device budgets."""
        with self._lock:
            for db in self._devices.values():
                db.current_agents = 0
            self._allocations.clear()
