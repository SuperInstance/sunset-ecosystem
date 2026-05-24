# Operational Trap Architecture

**Module:** `fleet/operational_trap.py`  
**Status:** P0 — Fleet Health Monitoring Base Class  
**Pattern Source:** `cocapn-traps` Operational Monitor Trap (Cross-Pollination Catalog #4)

---

## Purpose

Provide a unified, extensible trap framework for monitoring fleet health conditions and surfacing alerts.  Every fleet subsystem that needs health checks — thermal, FLUX constraints, agent processes, memory pressure — inherits from one base class and plugs into a single `TrapRegistry` run by `FleetConductor`.

---

## Design Principles

1. **One base, many traps** — `OperationalTrap` is abstract; all real checks are subclasses.
2. **Rate limiting at the base** — No alert storms.  Every trap gets a per-condition in-memory rate limiter.
3. **Configurable channels** — Alerts can go to log, Python callback, or A2A message without changing the trap logic.
4. **Dashboard-ready** — `TrapDashboard` flattens registry state into JSON for SSE streams or health endpoints.
5. **Thread-safe** — All mutable state is protected by `threading.Lock`; safe to call from FleetConductor's beat loop.

---

## Class Overview

```
OperationalTrap (abstract)
├── check() → TrapResult | None
├── escalate(result) → severity-based routing
├── notify(result) → channel dispatch
├── rate_limit(key) → bool
└── run() → check + escalate + rate_limit

TrapRegistry
├── register(trap) / unregister(trap)
├── run_all() → list[TrapResult]
└── get_status() → dict

TrapDashboard(registry)
└── get_status() → flattened snapshot

Built-in traps:
├── ThermalTrap(budget, threshold)
├── FluxViolationTrap(checker, score_threshold)
└── AgentCrashTrap(get_agent_pids, expected_agents)
```

---

## Data Model

### `TrapSeverity`

| Level | Meaning | Default Action |
|-------|---------|----------------|
| `INFO` | Worth noting, no response needed | Log at INFO |
| `WARNING` | Degraded, investigate soon | Log + callback |
| `CRITICAL` | Immediate action required | Log + callback + A2A |

### `TrapResult`

```python
@dataclass(frozen=True)
class TrapResult:
    condition: str          # e.g. "thermal_overcommit"
    severity: TrapSeverity
    message: str
    metadata: dict[str, Any]
    timestamp: float
```

Immutability guarantees that once a result is produced it can be passed across threads without defensive copies.

---

## Built-in Traps

### `ThermalTrap`

- **Input:** `ThermalBudget` instance + utilization threshold (default 95%).
- **Logic:** For each `DeviceType`, check `current_agents > max_agents` or `utilization > threshold`.
- **Severity:** `CRITICAL` if overcommitted (`current > max`), otherwise `WARNING` if above threshold.
- **Metadata:** list of offending devices, max utilization, threshold value.

### `FluxViolationTrap`

- **Input:** `FluxGatingChecker` instance + `score_threshold`.
- **Logic:** Queries the checker's WAL for violation records in the last 60 seconds.  Counts by severity string.
- **Severity:** `CRITICAL` if any critical violations, otherwise `WARNING` if any warnings.
- **Metadata:** critical count, warning count, window size, threshold.

> **Note:** This trap does not call `check_batch()` directly because that requires candidate data.  Instead it inspects the persistent WAL, making it suitable for periodic polling by `FleetConductor`.

### `AgentCrashTrap`

- **Input:** `get_agent_pids() → dict[agent_id, pid | None]` callable + optional `expected_agents` list.
- **Logic:** Any agent with `None` or `0` PID is flagged.  Any agent in `expected_agents` missing from the mapping is also flagged.
- **Severity:** Always `CRITICAL` — a dead agent is never acceptable.
- **Metadata:** missing agent list, expected count.

---

## Integration Points

| Consumer | How it uses traps |
|----------|-------------------|
| **FleetConductor** | Runs `TrapRegistry.run_all()` on every beat; routes criticals to A2A halt messages |
| **BreederDaemonV2** | Attaches `FluxViolationTrap`; if CRITICAL, pauses breeding for the cycle |
| **MeshVectorGossip** | Attaches `ThermalTrap`; if WARNING, increases `chaos` to force diversity exploration |
| **Health endpoint** | `TrapDashboard.get_status()` served as JSON on `/health/traps` |

---

## Usage Example

```python
from fleet.operational_trap import TrapRegistry, ThermalTrap, TrapDashboard
from swarm.thermal import ThermalBudget, DeviceType

# Set up
budget = ThermalBudget({DeviceType.GPU: 9, DeviceType.CPU: 36})
registry = TrapRegistry()
registry.register(ThermalTrap(budget=budget, threshold=0.95))

# On every beat
fired = registry.run_all()
for result in fired:
    if result.severity == TrapSeverity.CRITICAL:
        emergency_stop(result.message)

# Dashboard snapshot
dash = TrapDashboard(registry)
print(dash.get_status())
```

---

## Rate Limiting

Each trap maintains an in-memory dict:

```python
_rate_limit_store: dict[str, float]  # condition_key → last_alert_timestamp
```

- Keyed by `condition` string (e.g. `"thermal_overcommit"`), not trap name.
- Default interval: 60 seconds per condition.
- Thread-safe via `threading.Lock`.
- Rate-limited results are silently dropped from `run()` output but still counted in `total_fired`.

---

## Alert Channels

| Channel | Mechanism |
|---------|-----------|
| `log` | Python `logging` (already used by `escalate()`) |
| `callback` | User-supplied callable via `set_callback()` |
| `a2a` | User-supplied callable via `set_a2a_callback()` (intended for A2A message send) |

Channels are orthogonal — adding `"a2a"` does not remove `"log"`.

---

## Testing

15+ tests in `tests/test_operational_trap.py` covering:

- Base class contract (`NotImplementedError`)
- Escalation routing per severity
- Rate limiting (duplicate suppression, interval expiry, key isolation)
- `ThermalTrap` overcommit and threshold detection
- `FluxViolationTrap` WAL breach detection
- `AgentCrashTrap` missing / zero PID detection
- `TrapRegistry` registration, execution, thread safety
- `TrapDashboard` status aggregation

Run:
```bash
python -m pytest tests/test_operational_trap.py -x --tb=short
```

---

## Future Work

- **Persistence:** Move `_rate_limit_store` to a small TTL cache (Redis / memcached) for multi-node fleets.
- **Metrics:** Export `total_checks` and `total_fired` as Prometheus counters.
- **Auto-remediation:** Add `remediate(result)` hook to `OperationalTrap` for self-healing actions (e.g. kill dead agent, release thermal slot).
- **Custom traps via plugin:** Load trap classes from entry points so fleet nodes can register node-specific checks without code changes.

---

*CCC, Fleet Pattern Scout | "The trap should be beautiful, not deceptive."*
