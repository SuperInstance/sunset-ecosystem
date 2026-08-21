---
author: "Cocapn Fleet"
date: "2026-05-29"
category: "operational"
tags: [health, monitoring, trap, circuit-breaker, fleet, thermal]
---

# Operational Trap: Detecting Fleet Health Degradation

## Summary

How the `OperationalTrap` module detects thermal runaway, FLUX gating failures, dispatch cascades, and memory exhaustion — then triggers circuit breakers, cooldowns, and alerts.

## Problem

A fleet of 2,400 agents can fail in cascading ways:
1. **Thermal runaway** — CPU/GPU overheating from 100% utilization
2. **FLUX gating failures** — Constraint violations flood the system
3. **Dispatch cascades** — Gateway overload from too many subagents
4. **Memory exhaustion** — Context buffers fill and truncate
5. **Stalled breeding** — No new agents spawned for hours

Without detection, these issues silently degrade the fleet until total failure.

## Solution

### The Operational Trap Architecture

```
┌─────────────────────────────────────────────┐
│  OperationalTrap                             │
│  ┌─────────────┐  ┌─────────────┐           │
│  │  SENSE      │  │  DECIDE     │           │
│  │  - thermal  │  │  - threshold│           │
│  │  - flux     │  │  - trend    │           │
│  │  - dispatch │  │  - pattern  │           │
│  │  - memory   │  │             │           │
│  └─────────────┘  └─────────────┘           │
│         │                │                    │
│         ▼                ▼                    │
│  ┌─────────────────────────────────────┐     │
│  │  ACT                                │     │
│  │  - circuit_breaker (stop spawning)│     │
│  │  - cooldown (wait 20 min)           │     │
│  │  - alert (notify Casey)           │     │
│  │  - degrade (reduce batch size)    │     │
│  └─────────────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

### Threshold Configuration

```python
from fleet.operational_trap import OperationalTrap, TrapConfig

trap = OperationalTrap(
    config=TrapConfig(
        # Thermal thresholds
        thermal_cpu_max=85.0,  # degrees C
        thermal_gpu_max=80.0,  # degrees C
        thermal_rise_rate=5.0,  # degrees per minute
        # FLUX thresholds
        flux_violation_rate_max=0.1,  # 10% of breeds violating
        flux_gate_latency_ms=500,  # max FLUX check time
        # Dispatch thresholds
        dispatch_queue_max=100,  # pending subagents
        dispatch_timeout_rate=0.5,  # 50% of spawns timing out
        # Memory thresholds
        memory_context_max=0.85,  # 85% context utilization
        memory_truncation_rate=0.3,  # 30% of sessions truncated
        # Breeding thresholds
        breed_stall_minutes=60,  # no breeding for 1 hour
    )
)
```

### Wiring to FleetConductorV2

```python
from nexus.fleet_conductor_v2 import FleetConductorV2
from fleet.operational_trap import OperationalTrap

conductor = FleetConductorV2()
trap = OperationalTrap()

# Wire trap as a SenseDecideAct pipeline
conductor.register_pipeline("operational_trap", trap.sda_pipeline)

# On every beat, the trap senses and decides
conductor.beat()  # trap runs automatically
```

### Response Actions

When a trap triggers, the fleet responds in escalating levels:

| Level | Condition | Action |
|-------|-----------|--------|
| 1 (INFO) | Single threshold crossed | Log warning, increment counter |
| 2 (WARNING) | 2+ related thresholds | Reduce batch size, slow breeding |
| 3 (CRITICAL) | Any absolute max | Circuit breaker: stop all subagent spawns |
| 4 (EMERGENCY) | Thermal runaway | Shutdown non-essential agents, alert human |

```python
def on_trap_trigger(trap: OperationalTrap, level: int, context: dict) -> None:
    if level == 1:
        logger.warning(f"Trap: {context['metric']} = {context['value']}")
    elif level == 2:
        logger.warning(f"Degrading: batch_size → {context['recommended_batch_size']}")
        conductor.degrade_batch_size(context["recommended_batch_size"])
    elif level == 3:
        logger.critical("CIRCUIT BREAKER: Stopping all subagent spawns")
        conductor.circuit_breaker(duration_minutes=20)
    elif level == 4:
        logger.critical("EMERGENCY: Thermal runaway detected")
        alert_human(urgent=True, message=context["description"])
```

## Code Example

```python
#!/usr/bin/env python3
"""Operational trap detection and response for fleet health."""

import time
import psutil
from fleet.operational_trap import OperationalTrap, TrapConfig


class FleetHealthMonitor:
    def __init__(self, conductor):
        self.conductor = conductor
        self.trap = OperationalTrap(
            config=TrapConfig(
                thermal_cpu_max=85.0,
                flux_violation_rate_max=0.05,
                dispatch_timeout_rate=0.3,
                memory_context_max=0.80,
            )
        )
        self.alert_history = []

    def sense(self) -> dict:
        """Collect fleet health metrics."""
        return {
            "cpu_temp": self._get_cpu_temp(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "context_usage": self.conductor.context_usage(),
            "flux_violations": self.conductor.flux_violation_rate(),
            "pending_spawns": self.conductor.pending_spawn_count(),
            "timeout_rate": self.conductor.timeout_rate(minutes=10),
        }

    def decide(self, metrics: dict) -> dict:
        """Run trap logic on metrics."""
        return self.trap.evaluate(metrics)

    def act(self, decision: dict) -> None:
        """Execute trap response."""
        if decision["level"] == 0:
            return

        self.alert_history.append(decision)

        if decision["level"] >= 3:
            self.conductor.circuit_breaker(duration_minutes=20)
        elif decision["level"] == 2:
            self.conductor.degrade_batch_size(decision.get("recommended_batch_size", 8))

        # Log to fleet health dashboard
        self.conductor.publish_event(
            {
                "type": "OPERATIONAL_TRAP",
                "level": decision["level"],
                "metric": decision["metric"],
                "value": decision["value"],
                "timestamp": time.time(),
            }
        )

    def tick(self) -> None:
        """Full Sense→Decide→Act cycle."""
        metrics = self.sense()
        decision = self.decide(metrics)
        self.act(decision)

    def _get_cpu_temp(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                return max(t.current for t in temps["coretemp"])
        except Exception:
            pass
        return 0.0


def main():
    from nexus.fleet_conductor_v2 import FleetConductorV2

    conductor = FleetConductorV2()
    monitor = FleetHealthMonitor(conductor)

    # Run every 30 seconds
    while True:
        monitor.tick()
        time.sleep(30)


if __name__ == "__main__":
    main()
```

## References

- [Circuit Breaker Pattern] Nygard, M. (2018). Release It! (2nd ed.).
- [Thermal Management] Data center thermal design: https://www.energystar.gov/products/data_center_equipment
- [FleetConductorV2] Sunset Ecosystem orchestrator: `nexus/fleet_conductor_v2.py`
- [psutil] System monitoring: https://github.com/giampaolo/psutil
