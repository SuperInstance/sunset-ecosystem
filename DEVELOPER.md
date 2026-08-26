# Sunset Ecosystem — Developer Guide

**Version:** 0.9  
**Branch:** `turbovec-integration-ccc`  
**Audience:** Engineers extending the fleet  
**Prerequisite:** Read `README.md` first

---

## Table of Contents

1. [Architecture at a Glance](#1-architecture-at-a-glance)
2. [Adding a New Room](#2-adding-a-new-room)
3. [Adding a New Spell](#3-adding-a-new-spell)
4. [Extending the Breeding System](#4-extending-the-breeding-system)
5. [Hardware Swarm Integration](#5-hardware-swarm-integration)
6. [Component Integration Patterns](#6-component-integration-patterns)
7. [Testing Guidelines](#7-testing-guidelines)
8. [Debugging & Telemetry](#8-debugging--telemetry)

---

## 1. Architecture at a Glance

The system is organised around three ideas:

| Layer | Responsibility | Key Modules |
|-------|---------------|-------------|
| **Nerve** | Forward-only inference, RoomGrid topology | `nerve/room_grid.py`, `nerve/routing.py` |
| **Swarm** | Agent lifecycle, breeding, diversity | `swarm/breeder_daemon_v2.py`, `swarm/flux_vector_table.py`, `swarm/hdc_novelty.py` |
| **Sunset** | Trinity scoring, thermal management, sunset decisions | `sunset/trinity.py`, `sunset/thermal.py` |
| **Logos** | Decision journals, intent protocol, audit | `logos/decision_journal.py`, `logos/intent_protocol.py` |
| **Nexus** | Fleet coordination, event bus, mesh | `nexus/fleet_event_bus.py`, `nexus/distributed_consensus.py` |
| **Compiler** | Auto-compile, hot-swap, A/B gating | `compiler/hot_swap_integration.py` |
| **Perception** | Vision, audio, tile encoding | `perception/vision_encoder.py`, `perception/audio_encoder.py` |

The **trinity** (`ethos × pathos × logos`) is the central scoring model. Every agent must maintain non-zero scores in all three dimensions or it sunsets.

---

## 2. Adding a New Room

A "room" is a functional domain inside the RoomGrid. Rooms hold state, receive ticks, and broadcast metrics.

### 2.1 Minimal room implementation

```python
# my_domain/rooms/my_room.py
from dataclasses import dataclass
from typing import Any


@dataclass
class MyRoom:
    name: str = "my_room"
    activity: list[float] = None

    def __post_init__(self):
        if self.activity is None:
            self.activity = [0.0] * 64

    def tick(self, signal: np.ndarray) -> dict[str, Any]:
        """Process one tick, return metrics dict."""
        self.activity = self.activity * 0.9 + signal * 0.1
        return {
            "room": self.name,
            "mean_activity": float(np.mean(self.activity)),
            "peak": float(np.max(self.activity)),
        }
```

### 2.2 Registering the room

Add your room to `nerve/room_grid.py` in the `RoomGrid._init_rooms()` method:

```python
from my_domain.rooms.my_room import MyRoom

# Inside RoomGrid.__init__ or _init_rooms:
self.rooms["my_room"] = MyRoom()
```

### 2.3 Testing the room

Create `tests/test_my_room.py`:

```python
import numpy as np
from my_domain.rooms.my_room import MyRoom


def test_tick_returns_metrics():
    room = MyRoom()
    metrics = room.tick(np.random.randn(64))
    assert "mean_activity" in metrics
    assert 0.0 <= metrics["mean_activity"] <= 1.0
```

---

## 3. Adding a New Spell

"Spells" are automation primitives that agents can cast on rooms. They live in the PLATO spellbook pattern.

### 3.1 Spell interface

```python
# rooms/spells.py (or your own spell module)


class Spell:
    """Base class for all spells."""

    name: str = "base_spell"

    def cast(self, room: Any, **kwargs) -> Any:
        raise NotImplementedError


class SummonScout(Spell):
    """Spawn a subagent to explore a domain."""

    name = "summon_scout"

    def cast(self, room: Any, domain: str = "harbor", query: str = "") -> dict:
        # Implementation
        return {"spawned": True, "domain": domain}
```

### 3.2 Registering spells

Spells are typically registered in a `Spellbook` class or loaded dynamically via entry points. See `rooms/spells.py` for the existing pattern.

---

## 4. Extending the Breeding System

The breeding system has three main components:

### 4.1 BreederDaemonV2 (`swarm/breeder_daemon_v2.py`)

The daemon manages the full lifecycle: `EGG → INCUBATE → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)`.

To extend:
- **New lifecycle states** — Add to `LifecycleState` enum in `swarm/lifecycle_fsm.py`
- **New transition guards** — Add guard functions to `swarm/breeder_fsm_v2.py`
- **New sunset triggers** — Add to `TrajectoryMonitor` in `swarm/trajectory_monitor.py`

### 4.2 FluxVectorTable (`swarm/flux_vector_table.py`)

Diversity search for parent selection. Key methods:

```python
from swarm.flux_vector_table import FluxVectorTable

fvt = FluxVectorTable(dim=64)
fvt.add_agent(agent_id="abc", latent=vector)

# Find diverse parents
parents = fvt.find_diverse_parents(n=2, min_distance=0.3)
```

To extend:
- **New distance metrics** — Implement in `swarm/hdc_novelty.py` and wire into `FluxVectorTable.score_pair()`
- **New niche strategies** — Modify `update_niches()` and `niche_centroids`

### 4.3 HDC Novelty (`swarm/hdc_novelty.py`)

Binary hypervector novelty scoring. Uses:
- `BinaryVectorEncoder` — float32 → packed binary
- `HDCDiversityScorer` — XOR+POPCNT scoring
- `hdc_novelty_score()` — convenience function

To add a new encoding scheme:
1. Subclass `BinaryVectorEncoder`
2. Override `encode()` with your binarisation logic
3. Update `HDCDiversityScorer` to use the new encoder

---

## 5. Hardware Swarm Integration

The swarm scheduler (`sunset/hardware_swarm.py`) allocates agents to devices based on thermal headroom and workload type.

### 5.1 Adding a new device type

```python
# sunset/hardware_swarm.py


class MyDevice:
    """Custom accelerator."""

    device_type = "my_accelerator"

    def benchmark(self) -> dict:
        return {"tflops": 10.0, "watts": 50.0, "latency_us": 100}

    def allocate(self, agent: Agent) -> bool:
        # Return True if agent fits thermal budget
        return agent.thermal_estimate < self.headroom()
```

Register in `HardwareSwarm._discover_devices()`.

### 5.2 Thermal profiles

Thermal profiles are JSON files in `data/thermal_profiles/`. Each profile specifies:
- TDP (thermal design power)
- Clock curves
- Throttle thresholds

Add a new profile by creating `data/thermal_profiles/my_device.json` and reloading the swarm.

---

## 6. Component Integration Patterns

### 6.1 EventBus pattern

```python
from nexus.fleet_event_bus import FleetEventBus

bus = FleetEventBus()

# Subscribe
bus.on("grid_tick_metrics", lambda ev: print(ev.payload))

# Emit
bus.emit({"type": "grid_tick_metrics", "thermal_pressure": 0.7})
```

All cross-component communication should use the EventBus. Direct imports between components are discouraged except for type hints.

### 6.2 Metronome integration

```python
from nerve.metronome_integration import MetronomeIntegration

metro = MetronomeIntegration()
metro.register_device("gpu_0", callback=gpu_tick)
metro.start(period_ms=16)  # 60 FPS
```

Metronome handles drift correction and offline device detection.

### 6.3 Compiler hot-swap

```python
from compiler.hot_swap_integration import CompilerHotSwap

swap = CompilerHotSwap(grid, compiler=my_compiler)
swap.enable_auto_compile()

# grid.resize(200)  # auto-triggers recompile
```

The compiler monitors config changes, A/B tests compiled versions, and rolls back on failure.

---

## 7. Testing Guidelines

### 7.1 Unit tests

Every new module must have tests in `tests/`. Use the existing patterns:

```python
import pytest
from my_module import MyClass


def test_basic_functionality():
    obj = MyClass()
    result = obj.do_thing()
    assert result == expected


def test_error_handling():
    obj = MyClass()
    with pytest.raises(ValueError):
        obj.bad_input()
```

### 7.2 Integration tests

For cross-component features, use `tests/test_cross_repo_integration.py` as a template:

```python
def test_my_feature_end_to_end():
    grid = RoomGrid(10)
    bus = FleetEventBus()
    # ... wire components ...
    # ... assert behavior ...
```

### 7.3 Performance tests

For performance-critical code, add benchmarks in `benchmarks/`:

```python
def test_my_kernel_speed():
    import time

    t0 = time.perf_counter()
    for _ in range(1000):
        my_fast_function()
    dt = time.perf_counter() - t0
    assert dt < 0.01  # must be < 10ms for 1000 calls
```

---

## 8. Debugging & Telemetry

### 8.1 WAL (Write-Ahead Log)

Breeding decisions are logged to `data/wal/` for replay and audit:

```python
from swarm.breeder_daemon_v2 import WAL

wal = WAL(path="data/wal/breeding.wal")
wal.append({"event": "breed", "parents": ["a", "b"], "child": "c"})
```

Replay: `python3 scripts/replay_wal.py data/wal/breeding.wal`

### 8.2 Tide Pool visualization

The Tide Pool dashboard shows real-time fleet state:

```bash
python3 logos/tide_pool_viz.py
# → http://localhost:8080/tide-pool
```

Dark, bioluminescent UI. Shows thermal pressure, active agents, diversity metrics.

### 8.3 Decision journals

Every significant decision is logged with rationale:

```python
from logos.decision_journal import log_decision

log_decision(
    action="sunset_agent",
    agent_id="abc",
    reason="thermal_violation",
    context={"temp_c": 85, "threshold_c": 80},
)
```

Query: `python3 scripts/query_journal.py --agent abc --since "2026-05-01"`

---

## Quick Reference

| Task | File | Test |
|------|------|------|
| Add room | `nerve/room_grid.py` | `tests/test_room_grid.py` |
| Add spell | `rooms/spells.py` | `tests/test_spells.py` |
| Add breeder state | `swarm/lifecycle_fsm.py` | `tests/test_breeder_fsm_v2.py` |
| Add device | `sunset/hardware_swarm.py` | `tests/test_hardware_profiler.py` |
| Add distance metric | `swarm/hdc_novelty.py` | `tests/test_hdc_novelty.py` |
| EventBus emit | `nexus/fleet_event_bus.py` | `tests/test_fleet_event_bus.py` |

---

*This guide is a living document. If you find a gap, fix it and PR.*
