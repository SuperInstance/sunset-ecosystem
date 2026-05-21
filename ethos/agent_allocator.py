"""Allocate agent types to compute units based on hardware profile and stress test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ethos.hardware_survey import HardwareProfile
from ethos.stress_test import StressReport

__all__ = ["AgentAllocation", "AllocationPlan", "build_allocation_plan"]


@dataclass
class AgentAllocation:
    """Allocation for a single agent type."""

    agent_type: str
    count: int
    device: str  # "cuda:0", "cpu", "igpu", etc.
    compute_budget_pct: float  # fraction of device compute
    memory_budget_mb: float
    thermal_headroom_c: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"AgentAllocation({self.agent_type!r} x{self.count} on {self.device}, "
            f"compute={self.compute_budget_pct:.0%}, mem={self.memory_budget_mb:.0f}MB)"
        )


@dataclass
class AllocationPlan:
    """Complete allocation plan mapping agent types to hardware."""

    allocations: List[AgentAllocation] = field(default_factory=list)
    total_agents: int = 0
    thermal_budget_pct: float = 0.0  # estimated thermal usage
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AllocationPlan({self.total_agents} agents, "
            f"thermal={self.thermal_budget_pct:.0%}, "
            f"{len(self.allocations)} types)"
        )

    def for_device(self, device: str) -> List[AgentAllocation]:
        """Get allocations for a specific device."""
        return [a for a in self.allocations if a.device == device]

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines = [f"Allocation Plan: {self.total_agents} total agents"]
        for a in self.allocations:
            lines.append(f"  {a}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


# Agent type definitions with typical resource requirements
_AGENT_PROFILES: Dict[str, Dict[str, float]] = {
    "inference": {
        "compute_intensity": 0.9,  # GPU-heavy
        "memory_gb": 4.0,
        "preferred_device": "cuda",
        "thermal_impact": 0.8,
    },
    "embedding": {
        "compute_intensity": 0.5,
        "memory_gb": 1.0,
        "preferred_device": "cuda",
        "thermal_impact": 0.4,
    },
    "routing": {
        "compute_intensity": 0.1,
        "memory_gb": 0.5,
        "preferred_device": "cpu",
        "thermal_impact": 0.1,
    },
    "tool_execution": {
        "compute_intensity": 0.2,
        "memory_gb": 0.5,
        "preferred_device": "cpu",
        "thermal_impact": 0.1,
    },
    "memory": {
        "compute_intensity": 0.1,
        "memory_gb": 0.5,
        "preferred_device": "cpu",
        "thermal_impact": 0.05,
    },
    "vision": {
        "compute_intensity": 0.7,
        "memory_gb": 2.0,
        "preferred_device": "cuda",
        "thermal_impact": 0.6,
    },
    "code_execution": {
        "compute_intensity": 0.3,
        "memory_gb": 1.0,
        "preferred_device": "cpu",
        "thermal_impact": 0.2,
    },
}


def _get_thermal_headroom(profile: HardwareProfile) -> float:
    """Estimate thermal headroom in °C (how much hotter we can go)."""
    if profile.cuda_gpus:
        gpu = profile.cuda_gpus[0]
        if gpu.temperature_c is not None and gpu.power_limit_w is not None:
            # Assume throttle at ~85°C
            return max(0.0, 85.0 - gpu.temperature_c)
    if profile.thermal_zones:
        temps = [z.temperature_c for z in profile.thermal_zones if z.temperature_c is not None]
        if temps:
            return max(0.0, 85.0 - max(temps))
    return 30.0  # conservative default


def build_allocation_plan(
    profile: HardwareProfile,
    stress: StressReport,
    agent_types: Optional[List[str]] = None,
) -> AllocationPlan:
    """Build an allocation plan mapping agent types to hardware.

    Considers compute capacity, memory, thermal budget, and agent type
    characteristics to produce an :class:`AllocationPlan`.

    Args:
        profile: Hardware profile from survey.
        stress: Stress test report.
        agent_types: Which agent types to plan for. None = all known types.

    Returns:
        AllocationPlan with per-type allocations.
    """
    types = agent_types or list(_AGENT_PROFILES.keys())
    allocations: List[AgentAllocation] = []
    notes: List[str] = []
    total_agents = 0

    thermal_headroom = _get_thermal_headroom(profile)
    available_ram_mb = profile.memory.available_ram_mb

    # Calculate per-GPU resources
    gpu_count = len(profile.cuda_gpus)
    gpu_memory: Dict[int, float] = {}  # index -> free MB
    gpu_compute: Dict[int, float] = {}  # index -> GFLOPS

    for gpu in profile.cuda_gpus:
        gpu_memory[gpu.index] = gpu.free_memory_mb

    for bench in stress.benchmarks:
        if bench.device_type == "cuda":
            for m in reversed(bench.matrix_benchmarks):
                gpu_compute[bench.device_index] = m.gflops
                break

    # CPU resources
    cpu_cores = profile.cpu.cores_logical
    cpu_gflops = stress.cpu_gflops() or 100.0  # default estimate

    # Thermal budget: fraction of headroom we're willing to use (max 80%)
    thermal_budget_frac = min(0.8, thermal_headroom / 50.0) if thermal_headroom > 0 else 0.3

    # Allocate each agent type
    gpu_agents_total = 0
    gpu_memory_used = 0.0
    cpu_compute_used = 0.0

    for agent_type in types:
        ag = _AGENT_PROFILES.get(agent_type)
        if ag is None:
            notes.append(f"Unknown agent type {agent_type!r}, skipping")
            continue

        pref = ag["preferred_device"]
        mem_mb = ag["memory_gb"] * 1024
        thermal_impact = ag["thermal_impact"]
        compute_pct = ag["compute_intensity"]

        if pref == "cuda" and gpu_count > 0:
            # Use first available GPU
            gpu_idx = 0
            device = f"cuda:{gpu_idx}"
            free = gpu_memory.get(gpu_idx, 0)
            gflops = gpu_compute.get(gpu_idx, 0)

            # How many fit in memory?
            count_by_mem = max(1, int(free / mem_mb)) if mem_mb > 0 else 1
            # How many fit in thermal budget?
            count_by_thermal = max(
                1, int(thermal_budget_frac / (thermal_impact / gpu_count))
            )
            # How many fit in compute (assume 60% utilization target)
            per_agent_gflops = gflops * compute_pct * 0.6
            count_by_compute = (
                max(1, int(gflops * 0.6 / per_agent_gflops)) if per_agent_gflops > 0 else 1
            )

            count = min(count_by_mem, count_by_thermal, count_by_compute, 4)
            compute_budget = compute_pct / count

            alloc = AgentAllocation(
                agent_type=agent_type,
                count=count,
                device=device,
                compute_budget_pct=compute_budget,
                memory_budget_mb=mem_mb * count,
                thermal_headroom_c=thermal_headroom,
            )
            allocations.append(alloc)
            gpu_agents_total += count
            gpu_memory_used += mem_mb * count

        else:
            # CPU allocation
            per_core_compute = cpu_gflops / cpu_cores if cpu_cores > 0 else 10.0
            count = max(1, min(cpu_cores // 4, int(cpu_gflops * 0.4 / (compute_pct * per_core_compute + 1))))
            count = min(count, cpu_cores)

            alloc = AgentAllocation(
                agent_type=agent_type,
                count=count,
                device="cpu",
                compute_budget_pct=compute_pct / max(count, 1),
                memory_budget_mb=mem_mb * count,
            )
            allocations.append(alloc)
            cpu_compute_used += compute_pct * count

    total_agents = sum(a.count for a in allocations)

    if gpu_agents_total > 0 and gpu_memory:
        mem_pct = gpu_memory_used / gpu_memory.get(0, 1) * 100
        notes.append(f"GPU memory utilization: {mem_pct:.0f}%")

    notes.append(f"Thermal headroom: {thermal_headroom:.1f}°C (budget: {thermal_budget_frac:.0%})")

    return AllocationPlan(
        allocations=allocations,
        total_agents=total_agents,
        thermal_budget_pct=thermal_budget_frac,
        notes=notes,
    )
