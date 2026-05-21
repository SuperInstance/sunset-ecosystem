"""Score how connected a room is to ETHOS (the metal surveyor).

A high score means the agent's work uses hardware efficiently, latency is
within the metal's capability, and resource usage fits the thermal budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ethos.hardware_survey import HardwareProfile
from ethos.stress_test import StressReport

__all__ = ["EthosConnectionScore", "score_ethos_connection"]


@dataclass
class EthosConnectionScore:
    """Score (0.0-1.0) of how well-connected an agent/room is to ETHOS."""

    total: float
    hardware_efficiency: float  # Is compute being used well?
    latency_fit: float  # Is latency within the metal's capability?
    thermal_fit: float  # Does usage fit the thermal budget?
    memory_fit: float  # Does usage fit available memory?
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"EthosConnectionScore({self.total:.2f}: "
            f"eff={self.hardware_efficiency:.2f} "
            f"lat={self.latency_fit:.2f} "
            f"therm={self.thermal_fit:.2f} "
            f"mem={self.memory_fit:.2f})"
        )


def score_ethos_connection(
    profile: HardwareProfile,
    stress: StressReport,
    agent_compute_utilization: float = 0.5,
    agent_latency_ms: float = 100.0,
    agent_memory_mb: float = 2048.0,
    agent_thermal_impact: float = 0.3,
) -> EthosConnectionScore:
    """Score how well an agent's work aligns with the hardware.

    Takes the hardware profile and stress report along with an agent's
    actual or expected resource usage, and returns a 0.0-1.0 score
    indicating alignment with the metal.

    Args:
        profile: Hardware profile.
        stress: Stress test report.
        agent_compute_utilization: Fraction of compute the agent uses (0-1).
        agent_latency_ms: Agent's typical latency in ms.
        agent_memory_mb: Agent's memory usage in MB.
        agent_thermal_impact: Agent's thermal impact (0-1).

    Returns:
        EthosConnectionScore with overall and per-dimension scores.
    """
    notes: list[str] = []

    # --- Hardware efficiency ---
    # Is the agent using a good fraction of available compute without wasting it?
    efficiency = 1.0
    if agent_compute_utilization < 0.1:
        efficiency = 0.3  # under-utilizing
        notes.append("Compute under-utilized")
    elif agent_compute_utilization > 0.95:
        efficiency = 0.7  # near-saturation, risky
        notes.append("Compute near saturation")
    else:
        efficiency = 0.5 + 0.5 * (1.0 - abs(agent_compute_utilization - 0.6) / 0.6)

    # --- Latency fit ---
    # Compare agent latency to best benchmark latency
    best_device_latency = 1000.0  # default 1s
    for bench in stress.benchmarks:
        for m in bench.matrix_benchmarks:
            if m.size == 1024:
                best_device_latency = min(best_device_latency, m.avg_ms)

    if agent_latency_ms <= best_device_latency * 2:
        latency_fit = 1.0
    elif agent_latency_ms <= best_device_latency * 10:
        latency_fit = 0.5
    else:
        latency_fit = max(0.0, 1.0 - (agent_latency_ms / (best_device_latency * 100)))
    notes.append(f"Agent latency {agent_latency_ms:.0f}ms vs metal {best_device_latency:.0f}ms")

    # --- Thermal fit ---
    thermal_headroom = 30.0  # default
    if profile.cuda_gpus:
        gpu = profile.cuda_gpus[0]
        if gpu.temperature_c is not None:
            thermal_headroom = max(0.0, 85.0 - gpu.temperature_c)
    elif profile.thermal_zones:
        temps = [z.temperature_c for z in profile.thermal_zones if z.temperature_c is not None]
        if temps:
            thermal_headroom = max(0.0, 85.0 - max(temps))

    if thermal_headroom > 30:
        thermal_fit = 1.0
    elif thermal_headroom > 15:
        thermal_fit = 0.6
    elif thermal_headroom > 5:
        thermal_fit = 0.3
        notes.append("Thermal headroom low")
    else:
        thermal_fit = 0.1
        notes.append("Thermal headroom critical!")

    # Adjust by agent's thermal impact
    thermal_fit *= (1.0 - agent_thermal_impact * 0.3)
    thermal_fit = max(0.0, min(1.0, thermal_fit))

    # --- Memory fit ---
    available_mb = profile.memory.available_ram_mb
    if available_mb > 0:
        mem_ratio = agent_memory_mb / available_mb
        if mem_ratio < 0.3:
            memory_fit = 1.0
        elif mem_ratio < 0.6:
            memory_fit = 0.7
        elif mem_ratio < 0.85:
            memory_fit = 0.4
            notes.append("Memory usage high")
        else:
            memory_fit = 0.1
            notes.append("Memory near capacity!")
    else:
        memory_fit = 0.5  # unknown

    # Weighted total
    total = (
        efficiency * 0.30
        + latency_fit * 0.25
        + thermal_fit * 0.25
        + memory_fit * 0.20
    )
    total = max(0.0, min(1.0, total))

    return EthosConnectionScore(
        total=total,
        hardware_efficiency=round(efficiency, 3),
        latency_fit=round(latency_fit, 3),
        thermal_fit=round(thermal_fit, 3),
        memory_fit=round(memory_fit, 3),
        notes="; ".join(notes) if notes else "All dimensions nominal",
    )
