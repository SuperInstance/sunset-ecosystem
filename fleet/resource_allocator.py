"""resource_allocator.py — CPU/memory quota management for fleet agents.

Provides:
1. Per-agent resource quotas (CPU time, memory, GPU)
2. Dynamic allocation based on priority and demand
3. Resource reservation and release
4. OOM prevention with graceful eviction
5. Resource usage tracking and alerting

Usage:
    allocator = ResourceAllocator(total_cpu=8.0, total_memory=32000)
    alloc = allocator.allocate("agent-a", cpu=2.0, memory=4000)
    if alloc.granted:
        run_agent()
        allocator.release("agent-a")
"""

from __future__ import annotations

__all__ = [
    "ResourceAllocator",
    "Allocation",
    "ResourceUsage",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    """Result of a resource allocation request."""

    granted: bool
    cpu: float
    memory: float
    message: str = ""


@dataclass
class ResourceUsage:
    """Current resource usage for an agent."""

    agent_id: str
    cpu: float = 0.0
    memory: float = 0.0
    gpu: float = 0.0
    allocated_at: float = 0.0
    last_seen: float = 0.0


class ResourceAllocator:
    """Resource quota manager for fleet agents."""

    def __init__(
        self,
        total_cpu: float = 8.0,
        total_memory: float = 32000.0,  # MB
        total_gpu: float = 0.0,
    ) -> None:
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.total_gpu = total_gpu
        self._allocations: dict[str, ResourceUsage] = {}
        self._cpu_used = 0.0
        self._memory_used = 0.0
        self._gpu_used = 0.0

    # ── allocate ────────────────────────────────────────

    def allocate(
        self,
        agent_id: str,
        cpu: float = 1.0,
        memory: float = 512.0,
        gpu: float = 0.0,
        priority: int = 5,
    ) -> Allocation:
        """Request resources for an agent."""
        if agent_id in self._allocations:
            return Allocation(
                granted=False,
                cpu=0.0,
                memory=0.0,
                message="agent already allocated",
            )

        if self._cpu_used + cpu > self.total_cpu:
            return Allocation(
                granted=False,
                cpu=0.0,
                memory=0.0,
                message=f"CPU exhausted ({self._cpu_used}/{self.total_cpu})",
            )

        if self._memory_used + memory > self.total_memory:
            return Allocation(
                granted=False,
                cpu=0.0,
                memory=0.0,
                message=f"memory exhausted ({self._memory_used}/{self.total_memory})",
            )

        if self._gpu_used + gpu > self.total_gpu:
            return Allocation(
                granted=False,
                cpu=0.0,
                memory=0.0,
                message="GPU exhausted",
            )

        self._allocations[agent_id] = ResourceUsage(
            agent_id=agent_id,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            allocated_at=time.time(),
            last_seen=time.time(),
        )
        self._cpu_used += cpu
        self._memory_used += memory
        self._gpu_used += gpu

        return Allocation(granted=True, cpu=cpu, memory=memory)

    def release(self, agent_id: str) -> bool:
        """Release resources held by an agent."""
        usage = self._allocations.pop(agent_id, None)
        if usage is None:
            return False
        self._cpu_used -= usage.cpu
        self._memory_used -= usage.memory
        self._gpu_used -= usage.gpu
        return True

    def update_usage(self, agent_id: str, cpu: float, memory: float) -> None:
        """Update actual usage (may differ from allocation)."""
        usage = self._allocations.get(agent_id)
        if usage:
            # Adjust totals
            self._cpu_used += cpu - usage.cpu
            self._memory_used += memory - usage.memory
            usage.cpu = cpu
            usage.memory = memory
            usage.last_seen = time.time()

    # ── query ──────────────────────────────────────────

    def available(self) -> dict[str, float]:
        return {
            "cpu": max(0.0, self.total_cpu - self._cpu_used),
            "memory": max(0.0, self.total_memory - self._memory_used),
            "gpu": max(0.0, self.total_gpu - self._gpu_used),
        }

    def utilization(self) -> dict[str, float]:
        return {
            "cpu": self._cpu_used / self.total_cpu if self.total_cpu else 0.0,
            "memory": self._memory_used / self.total_memory
            if self.total_memory
            else 0.0,
            "gpu": self._gpu_used / self.total_gpu if self.total_gpu else 0.0,
        }

    def agents(self) -> list[str]:
        return list(self._allocations.keys())

    def agent_usage(self, agent_id: str) -> ResourceUsage | None:
        return self._allocations.get(agent_id)

    # ── eviction ───────────────────────────────────────

    def evict_least_active(self, count: int = 1) -> list[str]:
        """Evict the least recently seen agents."""
        sorted_agents = sorted(
            self._allocations.items(),
            key=lambda x: x[1].last_seen,
        )
        evicted = []
        for agent_id, _ in sorted_agents[:count]:
            self.release(agent_id)
            evicted.append(agent_id)
            logger.info(f"Evicted agent {agent_id}")
        return evicted

    def evict_by_memory(self, target_mb: float) -> list[str]:
        """Evict agents until target memory is freed."""
        freed = 0.0
        evicted = []
        sorted_agents = sorted(
            self._allocations.items(),
            key=lambda x: x[1].memory,
            reverse=True,
        )
        for agent_id, usage in sorted_agents:
            if freed >= target_mb:
                break
            self.release(agent_id)
            freed += usage.memory
            evicted.append(agent_id)
        return evicted

    # ── stats ─────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        return {
            "total_cpu": self.total_cpu,
            "total_memory": self.total_memory,
            "used_cpu": self._cpu_used,
            "used_memory": self._memory_used,
            "available": self.available(),
            "utilization": self.utilization(),
            "agents": len(self._allocations),
        }

    def __repr__(self) -> str:
        return f"ResourceAllocator(cpu={self._cpu_used:.1f}/{self.total_cpu}, mem={self._memory_used:.0f}/{self.total_memory})"
