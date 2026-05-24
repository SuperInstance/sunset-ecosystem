# Gateway Pacing

Circuit-breaker for subagent dispatch. Prevents gateway cascade failures by tracking consecutive timeouts and enforcing structured backoff.

## Design

```
OPEN  ──timeout──▶  CLOSED  ──backoff elapsed──▶  HALF_OPEN  ──3 successes──▶  OPEN
       x2                                (linear → exponential)      (1 probe / 30s)
```

## Quick Start

```python
from fleet.gateway_pacing import GatewayPacing

gp = GatewayPacing()

# Before spawning a subagent
ok, reason = gp.can_dispatch()
if not ok:
    logger.warning("Dispatch blocked: %s", reason)
    return

# After outcome is known
try:
    run_subagent(...)
    gp.record_success()
except TimeoutError:
    gp.record_timeout()
```

## State Machine

| State | Dispatches | Trigger |
|-------|------------|---------|
| **OPEN** | Unlimited | Default / recovered from HALF_OPEN |
| **CLOSED** | Blocked | `consecutive_timeouts >= max_consecutive_timeouts` |
| **HALF_OPEN** | 1 probe per `half_open_probe_interval` | Backoff elapsed from CLOSED |

## Parameters

```python
GatewayPacing(
    max_consecutive_timeouts=2,       # Trip after 2 timeouts
    linear_backoff_max=300.0,          # Cap linear phase at 5min
    exponential_backoff_max=1200.0,    # Hard cap at 20min
    half_open_probe_interval=30.0,     # 1 probe every 30s
    successes_to_reopen=3,             # 3 successes to reopen
    history_limit=50,                  # Retain last 50 events
)
```

## Backoff Math

| Consecutive Timeouts | Backoff |
|------------------------|---------|
| 1 | 30s |
| 2 | 60s |
| … | … |
| 10 | 300s (5min, linear cap) |
| 11 | 300s + 60s = 360s |
| 12 | 300s + 120s = 420s |
| … | … |
| large | 1200s (20min, hard cap) |

## API

### `can_dispatch() -> tuple[bool, str]`

Returns `(allowed, reason)`. Thread-safe.

- **OPEN**: Always `(True, "OPEN — dispatch allowed")`
- **HALF_OPEN**: `(True, ...)` only if `half_open_probe_interval` has elapsed since the last probe. Otherwise `(False, "HALF_OPEN — probe throttled; wait Xs")`.
- **CLOSED**: Always `(False, "CLOSED — circuit open; backoff Xs remaining")`.

### `record_success()`

Register a successful dispatch. In HALF_OPEN, counts toward `successes_to_reopen`. Resets `consecutive_timeouts`.

### `record_failure()`

Register a non-timeout failure. Does **not** trip the breaker. Resets `consecutive_timeouts`.

### `record_timeout()`

Register a timeout. Increments `consecutive_timeouts`. Trips to CLOSED when threshold reached.

### `get_status() -> dict`

Snapshot of internal state. Useful for dashboards / logs.

```python
{
    "state": "OPEN",
    "success_count": 42,
    "failure_count": 3,
    "consecutive_timeouts": 0,
    "backoff_remaining": 0.0,
    "last_dispatch_time": 1716604800.0,
    "circuit_closed_at": None,
    "half_open_probes_sent": 0,
    "half_open_successes": 0,
    "successes_to_reopen": 3,
    "history": [...],
}
```

## Integration Notes

- Import before spawning subagents in the main session dispatcher.
- Future: `FleetConductor` can share a `GatewayPacing` instance across nodes for fleet-wide pacing.
- The history ring-buffer is capped at `history_limit` to prevent unbounded memory growth during long sessions.

## Fleet Rule

> **Wait 20min after 2 consecutive timeouts.**

The default parameters encode this rule. When the gateway chokes:
1. Do NOT retry immediately.
2. Do NOT stack more spawns.
3. Do direct work (merge-prep, session cleanup, test finishing).
4. Return to spawns only when the gateway has breathed.

## Thread Safety

All public methods acquire a single `threading.Lock()`. Safe to call from multiple threads concurrently.

## Test Coverage

12+ tests in `tests/test_gateway_pacing.py`:
- OPEN state allows dispatch
- 2 consecutive timeouts → CLOSED
- CLOSED rejects dispatch with reason
- Backoff durations correct (linear then exponential)
- HALF_OPEN allows 1 probe per 30s
- 3 probes succeed → reopens to OPEN
- Status dict is accurate
- Thread-safe (multiple concurrent calls)
- `record_failure` does not trip breaker
- Success resets `consecutive_timeouts`
- Default parameters
- History entry structure
