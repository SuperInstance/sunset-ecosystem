# Dispatch Router — Two-Minute Test

**Pattern source:** Essay #4, *"The Scout Does Not Carry the Mountain"*  
**Status:** P0 — code module  
**Branch:** `feature/two-minute-test`

---

## What It Does

`fleet/dispatch_router.py` implements the **Two-Minute Test** pattern:

> If a task takes < 2 minutes, do it directly.  
> If it takes longer, delegate to a subagent — but only if the gateway isn't choking.

This prevents:
- **Subagent overhead on trivial work** (typo fixes, one-liner tweaks)
- **Gateway congestion** from spawning scouts for tasks that are faster done inline
- **Cascade failures** by integrating with the circuit breaker

---

## Quick Start

```python
from fleet.dispatch_router import DispatchRouter
from fleet.gateway_pacing import GatewayPacing

router = DispatchRouter(gateway=GatewayPacing())

decision = router.route("Fix typo in README")
# → {"mode": "direct", "reason": "...", "estimated_seconds": 30, ...}

decision = router.route("Implement mesh gossip with 12 tests")
# → {"mode": "subagent", "reason": "...", "estimated_seconds": 345, ...}
```

---

## API Reference

### `DispatchRouter`

#### `estimate_duration(task_description: str, context: dict | None = None) -> int`

Returns estimated seconds based on keyword heuristics and file count.

**Heuristic categories (default weights):**

| Category | Weight | Triggers |
|----------|--------|----------|
| `file_creation` | +60 s | "create", "implement", "new file", "scaffold" |
| `test_writing` | +90 s | "test", "assert", "pytest", "verify" |
| `doc_writing` | +45 s | "doc", "readme", "guide", "comment" |
| `research` | +180 s | "research", "investigate", "explore", "survey" |
| `bug_fix` | +120 s | "fix", "bug", "repair", "resolve" |
| `bug_fix_complex` | +300 s | "race condition", "memory leak", "deadlock", "intermittent" |
| `refactor` | +90 s | "refactor", "rename", "restructure", "clean up" |
| `integration` | +150 s | "integrate", "bridge", "wire up", "cross-module" |
| `simple_edit` | +30 s | "typo", "one-liner", "tweak", "quick" (additive only if alone) |
| `config_change` | +45 s | "config", "yaml", "json", "env" |
| `merge_conflict` | +60 s | "merge conflict", "rebase" |
| `dependency` | +120 s | "dependency", "requirements", "package" |
| `architecture` | +240 s | "architect", "design", "blueprint", "FSM" |

**File multiplier:** Each additional file beyond the first multiplies the estimate by +40 %, capped at 5×.

#### `should_delegate(task_description: str, context: dict | None = None) -> bool`

Shorthand: `True` if `estimate_duration > 120`.

#### `route(task_description: str, context: dict | None = None) -> dict`

Returns a routing decision:

| `mode` | Meaning | When |
|--------|---------|------|
| `direct` | Do inline | Estimate ≤ 120 s |
| `subagent` | Spawn a scout | Estimate > 120 s + gateway OPEN/HALF_OPEN |
| `deferred` | Queue for later | Estimate > 120 s + gateway CLOSED |

Example return value:

```python
{
    "mode": "subagent",
    "reason": "Estimated 345s > 120s threshold; OPEN — dispatch allowed",
    "estimated_seconds": 345,
    "gateway_state": "OPEN",
}
```

#### `record_actual(task_id, estimated, actual, task_description=None) -> dict`

**Learning hook.** After a task finishes, feed the actual duration back to the router. It adjusts the weights of matched categories toward the observed ratio.

```python
router.record_actual(
    task_id="mesh-gossip-scout",
    estimated=345,
    actual=420,
    task_description="Implement mesh gossip with 12 tests",
)
# → {'file_creation': 68.25, 'test_writing': 102.75, ...}
```

#### `get_weights() -> dict[str, float]`

Snapshot of current weights (useful for telemetry).

#### `get_feedback_summary() -> dict`

Aggregate learning stats:

```python
{
    "count": 42,
    "mean_ratio": 1.15,  # we tend to under-estimate by 15 %
    "median_ratio": 0.98,
}
```

---

## Integration Points

### 1. Main Session (before spawning)

```python
from fleet.dispatch_router import DispatchRouter

router = DispatchRouter()

decision = router.route(user_request)
if decision["mode"] == "direct":
    do_work(user_request)
elif decision["mode"] == "subagent":
    spawn_subagent(user_request)
else:
    queue_for_later(user_request)
```

### 2. Gateway Pacing

The router consults `GatewayPacing.can_dispatch()` before returning `subagent`. If the circuit is `CLOSED`, it returns `deferred` instead of overloading the gateway.

### 3. Fleet Conductor (future)

Planned: `FleetConductor` will use `DispatchRouter` for fleet-wide task routing, combining:
- Local two-minute test
- Node thermal budgets
- Cross-node load balancing

---

## Decision Rationale

### Why 120 seconds?

Empirically, subagent spawn + bootstrap + result delivery takes ~30–60 s of wall-clock time. If the task itself is < 2 min, the overhead dominates. The threshold keeps the overhead ratio below ~30 %.

### Why keyword heuristics instead of AST analysis?

Fast, stateless, no file-system reads. The main session calls `route()` before deciding to spawn — it must be cheap.

### Why a learning rate instead of exact regression?

The domain is non-stationary. A new framework, a new repo, a new model — all shift the true durations. A small per-feedback adjustment (`learning_rate=0.15`) adapts without overfitting to outliers.

---

## Testing

Run:

```bash
cd sunset-ecosystem
pytest tests/test_dispatch_router.py -v
```

12+ tests cover:
- Simple → direct
- Complex → subagent
- Gateway CLOSED → deferred
- Feedback loop convergence
- Edge cases: empty, ambiguous, very long descriptions

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-25 | Initial implementation (`feature/two-minute-test`) |

---

*CCC, Fleet Pattern Scout | "The scout does not carry the mountain.