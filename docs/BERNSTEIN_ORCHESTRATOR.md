# Bernstein Orchestrator

Deterministic parallel agent scheduler with **git worktree isolation**, **HMAC-signed audit chains**, and **gate-check verification** — distilled from the Bernstein project's core patterns without the 108K-line CLI baggage.

---

## Architecture

```
BernsteinOrchestrator
├── GitWorktreeSpawner      → isolated worktree + branch per task
├── DeterministicScheduler   → ThreadPoolExecutor + retry + alternate strategy
├── HMACAuditChain           → tamper-evident decision log
└── JanitorVerifier          → files + tests + lint gates
```

---

## Quick Start

```python
from fleet.bernstein_orchestrator import (
    BernsteinOrchestrator,
    OrchestratorConfig,
    SchedulerTask,
)

config = OrchestratorConfig(max_workers=4, default_timeout=300.0)
orth = BernsteinOrchestrator(config)

tasks = [
    SchedulerTask(
        task_id="fix_typos",
        command=lambda: {"files_changed": ["README.md"]},
        expected_outputs=["README.md"],
        timeout=60.0,
    ),
]

result = orch.orchestrate("/path/to/repo", tasks)
print(result["merged"])   # tasks that passed all gates
print(result["cleaned"])  # worktrees removed
```

---

## Classes

### `GitWorktreeSpawner`

Creates an isolated git worktree per task.

- `spawn(task_id) -> (worktree_path, branch_name)`
- `cleanup(task_id, branch_name)`
- Thread-safe; duplicate `task_id` raises `ValueError`
- Branch naming: `agent-{task_id}-{timestamp_ms}`

### `DeterministicScheduler`

Pure-Python parallel scheduling. No LLM involvement.

- `schedule(tasks: List[SchedulerTask], worktree_map=None) -> Dict[str, ScheduleResult]`
- Each `SchedulerTask` carries a zero-argument callable
- Retry: exponential backoff (`base_backoff * multiplier^attempt`)
- Alternate strategy: if `task.alternate_strategy` is set, it runs on retry
- Result fields: `task_id`, `status`, `worktree_path`, `output`, `retry_count`, `duration`, `error`

### `HMACAuditChain`

Tamper-evident decision log.

- Key: `BERNSTEIN_AUDIT_KEY` env var, or auto-generated per session
- `log_decision(decision_type, task_id, details) -> AuditEntry`
- `verify_chain() -> (bool, int)` — returns `(all_valid, first_invalid_index)`
- Each entry links previous hash + HMAC-SHA256 signature
- Export to newline-delimited JSON

### `JanitorVerifier`

Three-gate verification before merging.

1. **Files gate**: expected output files exist
2. **Tests gate**: test command returns 0
3. **Lint gate**: lint command returns 0

- `verify(worktree_path, expected_outputs, test_cmd, lint_cmd) -> VerificationReport`
- If any gate fails, `passed=False` and `gate` names the culprit

### `BernsteinOrchestrator`

Composes all four classes + GatewayPacing integration.

- `orchestrate(repo_path, tasks, config) -> result_dict`
- `attach_to_fleet_conductor(fleet_conductor_v2)` — registers as backend
- Aborts via GatewayPacing circuit breaker (`can_dispatch()` → False)
- Aborts if `_active_count >= gateway_max_concurrent`
- Cleanup: `cleanup_on_success` / `cleanup_on_failure` flags

Result dict keys:
```python
{
    "spawned":   {task_id: {worktree, branch}},
    "scheduled": {task_id: {status, worktree_path, output, retry_count, duration, error}},
    "verified":  {task_id: {passed, gate, details}},
    "merged":    [task_id, ...],
    "cleaned":   [task_id, ...],
    "audit_entries": int,
    "aborted": bool,
    "abort_reason": str,  # if aborted
}
```

---

## Integration with FleetConductorV2

```python
from nexus.fleet_conductor_v2 import FleetConductorV2
from fleet.bernstein_orchestrator import BernsteinOrchestrator, OrchestratorConfig

conductor = FleetConductorV2()
orch = BernsteinOrchestrator(OrchestratorConfig(max_workers=8))
orth.attach_to_fleet_conductor(conductor)

# Now conductor.orchestrate() delegates to Bernstein
```

---

## Testing

Run: `python3 -m pytest tests/test_bernstein_orchestrator.py -v`

31 tests covering:
- Git worktree spawn / cleanup / race conditions
- Scheduler parallel execution, retry, alternate strategy
- HMAC chain integrity, tamper detection, export/import
- Janitor gates (files, tests, lint)
- Orchestrator abort on GatewayPacing / max concurrent

---

## Decision: Clean-Room Implementation

We studied `chernistry/bernstein` (43 CLI adapters, 108K lines) and extracted three patterns:
1. **Git worktree isolation** — prevents merge conflicts between agents
2. **Deterministic scheduling** — no LLM in the loop, pure Python parallelism
3. **HMAC audit chain** — tamper-evident decision log

Everything else (CLI adapters, model-specific code, 40+ Git providers) was left behind.
