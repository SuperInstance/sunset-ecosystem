# Fleet Bernstein Scheduler

**Purpose:** Integrate Bernstein's deterministic orchestration primitives into FleetConductorV2.

**What it does:**
1. **DeterministicReplay** — Hermetic LLM call recording/replay backed by fleet SignedWAL
2. **PhasedDispatch** — research→plan→implement→verify phase separation for subagents
3. **ScheduleSupervisor** — Cron-fired schedule dispatch with catch-up/skip policies
4. **WorkerIsolation** — Process-visible subagent wrapper with signal forwarding

**Reference:** `fleet/fleet_bernstein_scheduler.py`

---

## Architecture

```
FleetConductorV2.beat()
  └── bernstein_scheduler.tick()
        ├── ScheduleSupervisor: check cron → fire due tasks
        ├── PhasedDispatch: route through research/plan/impl/verify
        ├── DeterministicReplay: record or replay LLM calls
        └── WorkerIsolation: spawn with PID metadata + signal forwarding
```

All four primitives integrate with existing fleet subsystems:
- **DeterministicReplay → SignedWAL** (crypto-integrity audit chain)
- **PhasedDispatch → AgentRegistry + GatewayPacing** (A2A identity + circuit breaker)
- **ScheduleSupervisor → MetronomeBridge** (cron fires on metronome beat)
- **WorkerIsolation → OperationalTrap** (health monitoring via PID files)

---

## Components

### FleetDeterministicReplay

Hermetic LLM call recorder/replayer. In recording mode, every call appends a signed WAL entry. In replay mode, pre-loaded responses are returned in FIFO order.

**Key feature:** The lookup key is a SHA-256 hash folding every response-determining input — model, prompt, provider, temperature, max_tokens. Any drift in any parameter causes a miss, which is exactly what you want for strict replay.

```python
replay = FleetDeterministicReplay(
    wal=signed_wal,
    run_id="breed-run-42",
    replay=True,
    strict=True,
)

# Replay mode: returns recorded response or raises ReplayMissError
try:
    response = replay.get_replay(prompt, model)
except ReplayMissError:
    # Strict replay will NOT call the live model
    pass
```

**Coverage line:** `replay-coverage run_id=breed-run-42 cached=12 hits=8 misses=0 strict_violations=0 strict=True`

**Non-hermetic fall-through:** Set `FLEET_REPLAY_ALLOW_LIVE_MISS=1` to return None on miss instead of raising. Use only for debugging — never in CI.

### FleetPhasedDispatch

Drives a subagent task through discrete phases. Each phase runs in a fresh invocation (clean context window). Only the `PhaseArtifact` is forwarded between phases — never raw transcripts or tool outputs.

```python
def my_executor(task_spec, phase_spec, prior_artifact):
    # Spawn subagent, run phase, return distilled artifact
    return PhaseArtifact(
        summary="Implemented feature X",
        decisions=["Use async I/O", "Add retry logic"],
        constraints=["Must pass test_foo"],
    )

dispatch = FleetPhasedDispatch(executor=my_executor, phases=[Phase.RESEARCH, Phase.PLAN, Phase.IMPLEMENT])
result = dispatch.run(task_spec, pacing=gateway_pacing)
```

**Mechanical gate:** Each artifact is validated — must have non-empty summary, decisions list, constraints list. Fails trigger retry (up to `gate_max_retries`).

**WAL integration:** Every phase transition is recorded as a `phase_transition` WAL entry with parent hash chaining.

### FleetWorkerIsolation

Process-visible wrapper for spawned subagents, inspired by `bernstein-worker`.

**Features:**
1. Process title shows role + session (visible in `ps`)
2. PID metadata file written for fleet monitoring
3. Signals forwarded to child process
4. Tool abort policies: `contain` / `sibling` / `session`
5. Cleanup on exit

```python
worker = FleetWorkerIsolation()
info = worker.spawn(
    role="auditor",
    session_id="audit-001",
    command=["python", "-m", "pytest", "tests/"],
    wal=signed_wal,
)
# info: {worker_pid, child_pid, pid_file, started_at, status}
```

### FleetBernsteinScheduler

The main scheduler class that runs inside FleetConductorV2's `beat()` loop.

**Schedule registration:**
```python
scheduler.register_schedule(
    "breed-every-6h",
    "0 */6 * * *",
    {"task_id": "breed", "preset": "FleetHealth"},
    misfire_policy="catch_up",
    goal="Run breeding cycle every 6 hours",
)
```

**Tick (called from conductor.beat):**
```python
result = scheduler.tick(
    now=time.time(),
    pacing=conductor._get_pacing(),
    wal=conductor._get_wal(),
)
```

**Misfire policies:**
- `skip` (default): Only dispatch the most recent missed instant. Earlier missed windows are logged as counterfactual receipts.
- `catch_up`: Dispatch up to `catch_up_limit` missed windows sequentially. Beyond the limit, remaining windows are skipped and logged as counterfactuals.

---

## Cron Format

5-field POSIX cron: `minute hour day month weekday`

| Field | Range | Special |
|-------|-------|---------|
| minute | 0-59 | `*`, `*/15`, `1-10/2`, `1,3,5` |
| hour | 0-23 | same |
| day | 1-31 | same |
| month | 1-12 | same |
| weekday | 0-6 (Sun=0) | same |

**Day matching:** POSIX union rule — if both day and weekday are restricted, a date matches if either condition is true.

**Determinism:** All fire computation is UTC-only. Host timezone is NOT part of the deterministic contract.

---

## Integration with FleetConductorV2

The scheduler is registered as a subsystem:

```python
# In FleetConductorV2.__init__:
if config.enable_bernstein_scheduler:
    from fleet.fleet_bernstein_scheduler import FleetBernsteinScheduler, BernsteinScheduleConfig
    bconfig = BernsteinScheduleConfig(
        node_id=config.node_id,
        tick_interval_s=config.sda_interval_ms / 1000.0,
        enable_phased_dispatch=True,
        enable_deterministic_replay=config.enable_deterministic_replay,
        enable_worker_isolation=True,
    )
    self._subsystems["bernstein_scheduler"] = SubsystemWrapper(
        name="bernstein_scheduler",
        factory=lambda: FleetBernsteinScheduler(bconfig),
    )
```

In `beat()`:
```python
# After SDA tick, before mesh gossip:
if self._get_bernstein_scheduler():
    tick_result = self._get_bernstein_scheduler().tick(
        now=now,
        pacing=self._get_pacing(),
        wal=self._get_wal(),
    )
    heartbeat["bernstein_scheduler"] = tick_result
```

---

## Files

- `fleet/fleet_bernstein_scheduler.py` — Implementation
- `tests/test_fleet_bernstein_scheduler.py` — 30+ tests covering cron, replay, phases, workers, integration
- `docs/FLEET_BERNSTEIN_SCHEDULER.md` — This document

---

## Test Summary

| Category | Tests | Key Assertions |
|----------|-------|----------------|
| Cron parsing | 6 parametric | `*/15`, `1-10/2`, unions |
| Schedule fire | 4 | Due detection, skip vs catch-up |
| Counterfactual | 2 | Skipped windows recorded |
| Deterministic replay | 7 | Hit, miss strict, miss non-strict, seed, coverage |
| Phased dispatch | 7 | Default executor, pacing block, gate retry, WAL |
| Worker isolation | 5 | Spawn, PID file, cleanup, sunset WAL, command not found |
| Integration | 4 | ConductorV2 subsystem, beat trigger, health check |
| Thread safety | 1 | Concurrent register + tick |
| Edge cases | 2 | UTC only, empty schedules |

**Total: 38 tests**
