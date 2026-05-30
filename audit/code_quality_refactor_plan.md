# Code Quality Audit Report — Sunset Ecosystem

**Auditor:** Code Quality Scout (CCC subagent)  
**Date:** 2026-05-31  
**Scope:** 872 non-test Python files across the sunset-ecosystem  
**Focus:** Code quality, duplication, dead code, complexity, security  

---

## Executive Summary

The codebase shows signs of rapid growth with **216 duplicate code block groups**, **massive 400+ line functions**, **unsafe deserialization paths**, and **inconsistent error handling**. The most critical issues are:
1. **P0: Unsafe pickle deserialization** in two widely-used modules
2. **P0: 421-line `step()` function** with 45 branches handling 15+ responsibilities
3. **P0: Entire simulator modules duplicated** (`tournament_sim.py` vs `tournament_sweep.py`)

---

## 1. Duplicate Code Blocks (P0–P1)

### P0 — Near-Total Module Duplication

| Files | Duplication |
|-------|------------|
| `simulators/tournament_sim.py` ↔ `simulators/tournament_sweep.py` | Identical `Agent` class, `crossover()`, `mutate()`, `tournament_step()`, `dominates()`, `diversity_metric()` |

**Impact:** 87 lines of identical logic. Changes to breeding mechanics must be made in two places. `tournament_sweep.py` adds CSV export and `breeding_events` counting on top of the same core.

**Fix:** Extract `simulators/tournament_core.py` with the shared `Agent` class and tournament logic. Both modules import from it.

### P1 — Fleet Module Stats Boilerplate (47+ files)

Every `fleet/` module duplicates the same introspection pattern:
```python
def stats(self) -> Dict[str, Any]:
    return {"runs": self._stats["runs"], ...}

def get_stats(self) -> Dict[str, Any]:
    return self.stats()

def to_dict(self) -> Dict[str, Any]:
    return {"stats": self.get_stats()}

def export_json(self) -> str:
    return json.dumps({"node": self.fleet_node_id, ...})
```

**Affected files:** `fleet/api_gateway.py`, `fleet/audit_logger.py`, `fleet/service_mesh.py`, `fleet/workflow_engine.py`, `fleet/request_proxy.py`, `fleet/leader_election.py`, `fleet/data_validator.py`, `fleet/encryption_helper.py`, `fleet/feature_flag.py`, `fleet/traffic_splitter.py`, `fleet/adaptive_timeout.py`, `fleet/capacity_planner.py`, `fleet/header_filter.py`, `fleet/plugin_manager.py`, `fleet/metric_reporter.py`, `fleet/sandbox_runner.py`, `fleet/rollback_manager.py`, `fleet/secret_rotator.py`, `fleet/request_deduplicator.py`, `fleet/weighted_router.py`, `fleet/alert_manager.py`, `fleet/trace_collector.py`, `fleet/request_recorder.py`, `fleet/feature_toggles.py`, `fleet/ip_allowlist.py`, `fleet/payload_compressor.py`, `fleet/geo_distributor.py`, `fleet/distributed_counter.py`, `fleet/response_cache.py`, `fleet/dependency_resolver.py`, `fleet/retry_policy.py`, `fleet/task_queue.py`, `fleet/chaos_engine.py`, `fleet/telemetry_buffer.py`, `fleet/health_aggregator.py`, `fleet/schema_registry.py`, `fleet/state_machine.py`, `fleet/throttle.py`, `fleet/result_aggregator.py`, `fleet/shard_manager.py`, `fleet/request_signer.py`, `fleet/config_loader.py`, `fleet/log_shipper.py`, `fleet/resource_quota.py`, `fleet/dns_cache.py`, `fleet/exception_tracker.py`, `fleet/event_correlator.py`, `fleet/secret_manager.py`, `fleet/event_bus.py`, `fleet/backup_manager.py`, `fleet/search_engine.py`, `fleet/canary_deployer.py`, `fleet/tracing.py`, `fleet/endpoint_registry.py`, `fleet/ledger_manager.py`, `fleet/feature_flags.py`, `fleet/a2a_plugin_manager.py`, `fleet/distributed_cache.py`, `fleet/ab_tester.py`, `fleet/compression_engine.py`, `fleet/data_pipeline.py`, `fleet/notification_system.py`

**Fix:** Create a `FleetStatsMixin` dataclass or base class in `fleet/_base.py` that provides `stats()`, `get_stats()`, `to_dict()`, `export_json()`. All modules inherit from it.

### P1 — `__repr__` Patterns Duplicated (40+ files)

The `__repr__` format string pattern is copy-pasted across 40+ modules with only the class name changed:
```python
def __repr__(self) -> str:
    return (
        f"ClassName(root={self.root!r}, files={self.file_count}, ..."
    )
```

**Fix:** Use a generic `reprlib`-style helper or a `@dataclass` decorator where possible.

---

## 2. Dead Code (P1–P2)

### P1 — Unused exports in `__init__.py` modules

| File | Dead exports |
|------|-------------|
| `grammar/__init__.py` | `Production`, `Rule`, `ValidationError`, `validate_rule_name`, `validate_tagline`, `validate_condition`, `validate_exec_field`, `create_rule`, `create_rule_from_dict`, `score_rule`, `evolve`, `batch_create_rules` — none imported or used in the `__init__` itself |
| `triage/__init__.py` | `DriftDetector`, `DriftReport`, `detect_drift`, `DuplicateDetector`, `DuplicatePair`, `find_duplicates`, `GitHubIssues`, `IssueState`, `RepoHealthMetrics`, `HealthScore`, `run_health_check`, `RepoDuplicateDetector`, `RepoDuplicatePair`, `find_repo_duplicates`, `WeeklyTriage`, `TriageReport`, `run_triage` |
| `distill/__init__.py` | `PromptHistory`, `PromptRecord`, `HintSchedule`, `ExponentialBackoffSchedule`, `BacktestRunner`, `BacktestResult`, `DistillationSignal`, `DistillationGuidance`, `DeltaSnapshot`, `DeltaTracker` |

### P2 — Unused imports in individual files

| File | Unused imports |
|------|---------------|
| `pathos/moment_scorer.py` | `time` (line 9), `field` (line 10) |
| `a2a/handlers.py` | `time` (line 10) |
| `nerve/metronome_integration.py` | `field` (line 22) |
| `triage/drift_detect.py` | `os` (line 19), `Dict` (line 24) |
| `triage/metrics.py` | `os` (line 15), `re` (line 16), `Optional` (line 20) |
| `distill/backtest_runner.py` | `random` (line 7), `PromptRecord` (line 12) |
| `distill/prompt_history.py` | `Any` (line 10) |
| `distill/distillation_signal.py` | `Any` (line 8) |
| `distill/hint_schedule.py` | `dataclass` (line 8) |

---

## 3. Overly Complex Functions (P0–P1)

### P0 — `swarm/breeder_daemon_v2.py:step()` (421 lines, 45 branches)

**Responsibilities:** 15+ distinct operations in one function:
1. Dequeue breed request
2. Thermal budget check + hysteresis
3. Parent sacrifice logic
4. FLUX gating (Path A)
5. Find cold room
6. Release old agent + record SUNSET
7. Inheritance tax calculation
8. Genealogy lookup
9. FLUX batch check on top-k rooms
10. Place child in EGG state
11. Room allocation via `grid.birth()` / `grid.rebirth()`
12. Vector table sync
13. TrajectoryMonitor security circuit breaker
14. LineageSanityChecker validation
15. Diversity collapse monitoring
16. Thermal allocation
17. Signed WAL logging

**Nesting depth:** Up to 6 levels deep (`if → if → try → if → for → if`).

**Fix:** Decompose into a pipeline of private methods:
```python
def step(self) -> list[LifecycleTransition]:
    request = self._dequeue_request()
    if not request: return []
    if not self._check_thermal(request): return []
    if not self._check_flux_gate(request): return []
    room = self._find_or_evict_room(request)
    child = self._create_child(request, room)
    self._run_post_spawn_checks(child, room)
    return self._transitions
```

### P0 — `swarm/breeder_daemon_v2.py:_select_parents_vector()` (190 lines, 46 branches)

Uses deeply nested vector table logic, CRDT merging, lineage checking, and thermal constraints all in one parent selection routine.

### P1 — `logos/tide_pool_viz.py:_html_template()` (378 lines, 0 branches)

A 378-line string literal containing HTML/CSS/JS. Not complex in branches, but a maintainability nightmare. Should be a Jinja2 template file.

### P1 — `experiments/distillation_demo.py:run_experiment()` (221 lines, 19 branches)

### P1 — `flux_compat/opcode_map.py:map_opcode()` (213 lines, 18 branches)

### P1 — `nexus/fleet_conductor_v2.py:_build_subsystems()` (195 lines, 8 branches)

### P1 — `fleet/holodeck.py:_generate_threejs_html()` (191 lines, 0 branches)

Another massive HTML template as a Python string. Should be a `.html` template file.

### P1 — `swarm/breeder_daemon.py:auto_breed()` (167 lines, 17 branches)

### P1 — `fleet/bernstein_orchestrator.py:_run_flow()` (136 lines, 9 branches)

### P1 — `a2a/server.py:_make_handler()` (133 lines, 16 branches)

### P1 — `fleet/fleet_weather_report.py:from_conductor()` (132 lines, 21 branches)

### P1 — `nexus/fleet_conductor_v2.py:beat()` (124 lines, 19 branches)

### P1 — `nerve/topology.py:tick()` (121 lines, 20 branches)

### P1 — `swarm/flux_vm_runner.py:run()` (120 lines, 23 branches)

### P2 — `scripts/demo_breeding_cycle.py:run_demo()` (180 lines, 20 branches)

### P2 — `scripts/demo_full_stack.py:run_demo()` (127 lines, 9 branches)

---

## 4. Missing Type Hints on Public APIs (P1–P2)

### P1 — `simulators/tournament_sim.py` & `simulators/tournament_sweep.py`

The entire public API of these modules lacks type hints:
```python
def crossover(a, b):          # no types
def mutate(agent, rate):      # no types
def tournament_step(pop, thermal_cap, mutation_rate):  # no types
def simulate(pop_size=20, ...):  # no types
def diversity_metric(pop):    # no types
```

### P2 — `scripts/demo_*.py` modules

All `run_demo()` functions in `scripts/demo_breeding_cycle.py`, `scripts/demo_full_stack.py`, `scripts/demo_audio_tiles.py`, `scripts/demo_vision_tiles.py` lack return type annotations and parameter types.

### P2 — `perception/audio_encoder.py` & `perception/vision_encoder.py`

Mixed type hints: some public methods have full annotations, others have partial or missing annotations on helper functions.

---

## 5. Inconsistent Error Handling (P0–P1)

### P0 — Bare `except Exception: pass` in `swarm/breeder_daemon_v2.py`

```python
try:
    self._vector_table.remove(child_id)
except Exception:
    pass
```

**Line:** inside `step()` after lineage tamper detection. This silently swallows **all** errors including database corruption, file system errors, and programming bugs. **Never use bare `except Exception: pass` in production code.**

### P1 — Mixed patterns in the same module

`swarm/breeder_daemon_v2.py` uses three different error handling strategies within 100 lines:
- `logger.exception(...)` — logs and swallows
- `raise ValueError(...)` — propagates
- `try: ... except Exception: pass` — silently ignores

### P1 — `grammar/core.py` contradictory security posture

The module validates against SQL injection with `SQLI_BLACKLIST`, then immediately uses `eval(compile(...))` in `score_rule()`. This sends a mixed signal: "We care about security" + "We use eval anyway."

### P2 — `fleet/serialization_helper.py` vs `logos/compression_utils.py`

`SerializationHelper.unpack()` raises `ValueError` for unknown formats. `DictCompressor.decompress()` silently returns whatever `pickle.loads()` gives. No format validation, no checksum, no error recovery.

---

## 6. Hardcoded Values That Should Be Configurable (P1–P2)

| File | Value | What it controls | Should be |
|------|-------|-----------------|-----------|
| `swarm/breeder_daemon_v2.py` | `0.8` | Inheritance tax rate (parent slots retained) | `InheritanceTax.DEFAULT_RATE` or config |
| `swarm/breeder_daemon_v2.py` | `0.1` | FLUX batch chaos bump increment | `FluxGatingConfig.CHAOS_BUMP` |
| `swarm/breeder_daemon_v2.py` | `0.9999` | Champion fitness threshold | `ChampionConfig.FITNESS_THRESHOLD` |
| `swarm/breeder_daemon_v2.py` | `5` | LineageSanityChecker max_depth | `LineageConfig.MAX_DEPTH` |
| `swarm/breeder_daemon_v2.py` | `0xFFFF` | Default capability mask | `CapabilityConfig.DEFAULT_MASK` |
| `simulators/tournament_sweep.py` | `20, 100, 0.1, 30, 'dynamic', 42` | All default simulation parameters | `SimulationConfig` dataclass |
| `simulators/tournament_sim.py` | `20, 50, 0.1, 30, 42` | Same parameters, different defaults | `SimulationConfig` dataclass |
| `claw_fleet_bridge.py` | `127.0.0.1`, `8850` | Default bridge host/port | `BridgeConfig` or env vars |
| `claw_fleet_bridge.py` | `200` | Default HTTP status code | `HTTPStatus.OK` constant |
| `perception/audio_encoder.py` | `whisper`, `cpu`, `512` | Default backend, device, embedding dim | `AudioConfig` |
| `perception/audio_encoder.py` | `16000` | Sample rate (appears twice) | `AudioConfig.SAMPLE_RATE` |
| `perception/vision_encoder.py` | `siglip`, `cpu`, `512` | Default vision backend | `VisionConfig` |
| `voice/soniqo_bridge.py` | `5.0` | Default timeout | `SoniqoConfig.TIMEOUT` |
| `voice/soniqo_bridge.py` | `mock` | Default mode | `SoniqoConfig.MODE` |
| `flux_compat/flux_opt_codegen.py` | `16` | Default vector width | `FluxConfig.VECTOR_WIDTH` |
| `reasoning/python_bridge.py` | `5` | Default retry count | `BridgeConfig.RETRIES` |

---

## 7. Magic Numbers and Strings (P1–P2)

### P1 — `swarm/breeder_daemon_v2.py`
```python
child_slots = int(parent_slots * 0.8)          # "0.8" is a magic number
self.grid.chaos[rid] += 0.1                     # "0.1" is a magic number
if best_f >= 0.9999:                            # "0.9999" is a magic number
lineage_checker = LineageSanityChecker(max_depth=5)  # "5" is a magic number
capability_mask=0xFFFF                          # "0xFFFF" is a magic number
```

### P2 — `logos/compression_utils.py`
```python
scaled = ((flat - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)   # "255.0"
scaled = ((flat - min_val) / (max_val - min_val) * 65535.0).astype(np.uint16) # "65535.0"
header = struct.pack("!f f B", min_val, max_val, 8)  # "8"
header = struct.pack("!f f B", min_val, max_val, 16) # "16"
header = struct.pack("!f B", float(flat[0]), 32)     # "32"
```

### P2 — `perception/audio_encoder.py` & `perception/vision_encoder.py`
```python
SAMPLE_RATE = 16000     # appears twice in audio_encoder.py
CHUNK_SIZE = 64         # appears in both audio and vision
EMBED_DIM = 512         # appears in both
```

---

## 8. TODO/FIXME Comments That Need Action (P1–P2)

### P1 — `sunset/codegen.py:235`
```python
// TODO: implement via python-to-rust AST translator
```
This is a **stub** in a critical code generation path. The Rust generator currently only handles simple numeric loops and falls back to Python for anything complex. This limits performance gains.

### P1 — `swarm/breeder_daemon_v2.py` porting note
```python
"""Porting note: the existing AutoBreeder logic is wrapped as a compatibility shim so legacy callers can migrate gradually."""
```
The `auto_breed()` compatibility shim is 120+ lines of glue code. If the migration is done, remove it. If not, document the migration timeline.

### P2 — `logos/trinity_connection.py`
```python
HACK_strings = ["HACK", "TODO", "FIXME", "XXX"]
```
This is a meta-observation, but the fact that the codebase has a dedicated scanner for these markers suggests they are prevalent. The scanner itself found TODOs in `sunset/codegen.py`, `logos/codebase_state.py`, and `grammar/security_hardening.py`.

---

## 9. Security Issues (P0–P2)

### P0 — `fleet/serialization_helper.py:53` — Unsafe Pickle Deserialization
```python
def unpack(self, data: bytes, format: Optional[str] = None) -> Any:
    ...
    if fmt == "pickle":
        return pickle.loads(data)   # ⚠️ ARBITRARY CODE EXECUTION
```

**Impact:** If an attacker controls the serialized bytes, they can execute arbitrary Python code via crafted pickle payloads. This is used for inter-node communication.

**Fix:** Replace `pickle` with `msgpack` or `json` for all inter-node serialization. If pickle is absolutely needed for complex objects, add an HMAC signature and a whitelist of allowed classes.

### P0 — `logos/compression_utils.py:218` — Unsafe Pickle in DictCompressor
```python
@staticmethod
def decompress(compressed: CompressionResult) -> dict[str, Any]:
    import pickle
    raw = zlib.decompress(compressed.data)
    return pickle.loads(raw)   # ⚠️ ARBITRARY CODE EXECUTION
```

**Impact:** Same as above. `DictCompressor` is used for mesh gossip and WAL entries.

**Fix:** Replace with `json.loads()` or `msgpack`. Dictionaries of strings and numbers should never need pickle.

### P1 — `fleet/sandbox.py:98` — Eval in Sandbox
```python
def eval(self, code: str, ...):
    g = globals_dict or {}
    g["__builtins__"] = self._restricted_builtins()
    def _run():
        return eval(compile(code, "<sandbox>", "eval"), g)
    return self.run(_run, ...)
```

**Impact:** The restricted builtins help, but `compile()` + `eval()` can still escape via bytecode introspection or generator expressions. The `Sandbox` class is intended for "untrusted breeding code" but is not a true sandbox.

**Fix:** Use `subprocess` isolation with `seccomp-bpf` or replace with `ast.literal_eval()` for the limited use cases.

### P1 — `grammar/core.py:235` — Eval in Rule Scoring
```python
result = eval(
    compile(tree, "<condition>", "eval"),
    {"__builtins__": {}},
    metrics,
)
```

**Impact:** While `__builtins__` is empty, the AST whitelist is only checked *before* evaluation. A crafted AST could still reference global names or use side effects in `Constant` nodes.

**Fix:** Replace with a pure interpreter that walks the AST and computes the result without `eval()` or `compile()`.

### P2 — `sunset/codegen.py:267` — Shell Command Injection
```python
if os.system("which rustc > /dev/null 2>&1") == 0:
    self._available = True
```

**Impact:** While `rustc` is hardcoded, `os.system()` is dangerous. If `PATH` is manipulated, this could execute arbitrary commands.

**Fix:** Use `shutil.which("rustc")` instead of `os.system()`.

### P1 — Contradictory Security Posture in `grammar/core.py`
The module invests in SQL injection blacklisting (`SQLI_BLACKLIST`) and HTML tag stripping (`HTML_TAG_PATTERN`), then uses `eval()` in the same file. This is a security anti-pattern: either commit to safe evaluation or use a real sandbox, but don't mix validation theater with dangerous execution.

---

## 10. Priority Summary

| Priority | Count | Key Issues |
|----------|-------|-----------|
| **P0** | 5 | 1. Unsafe pickle in `fleet/serialization_helper.py` and `logos/compression_utils.py` (RCE)  <br>2. `swarm/breeder_daemon_v2.py:step()` — 421 lines, 15 responsibilities, 45 branches <br>3. `swarm/breeder_daemon_v2.py:_select_parents_vector()` — 190 lines, 46 branches <br>4. `simulators/tournament_sim.py` and `tournament_sweep.py` — full module duplication <br>5. Bare `except Exception: pass` in breeder daemon |
| **P1** | 12 | 1. Fleet stats boilerplate duplicated across 47+ files <br>2. `__init__.py` dead exports (grammar, triage, distill) <br>3. `logos/tide_pool_viz.py:_html_template()` — 378-line HTML string <br>4. `fleet/holodeck.py:_generate_threejs_html()` — 191-line HTML string <br>5. `experiments/distillation_demo.py:run_experiment()` — 221 lines <br>6. `flux_compat/opcode_map.py:map_opcode()` — 213 lines <br>7. `nexus/fleet_conductor_v2.py:_build_subsystems()` — 195 lines <br>8. `swarm/breeder_daemon.py:auto_breed()` — 167 lines <br>9. `a2a/server.py:_make_handler()` — 133 lines <br>10. `fleet/fleet_weather_report.py:from_conductor()` — 132 lines <br>11. Inconsistent error handling (raise vs log vs swallow) <br>12. `sunset/codegen.py` TODO — Rust generator is a stub |
| **P2** | 8 | 1. Unused imports in `pathos/moment_scorer.py`, `a2a/handlers.py`, `nerve/metronome_integration.py` <br>2. Magic numbers in `swarm/breeder_daemon_v2.py` (0.8, 0.1, 0.9999, 5, 0xFFFF) <br>3. Magic numbers in `logos/compression_utils.py` (255.0, 65535.0) <br>4. Hardcoded defaults in `perception/audio_encoder.py` and `vision_encoder.py` <br>5. `os.system()` in `sunset/codegen.py` <br>6. `scripts/demo_*.py` missing type hints <br>7. `grammar/core.py` contradictory security posture <br>8. `swarm/breeder_daemon_v2.py` AutoBreeder shim still present |

---

## Recommended Refactor Order

1. **Week 1 — Security (P0)**
   - Replace `pickle.loads()` with `msgpack` or `json` in `fleet/serialization_helper.py` and `logos/compression_utils.py`
   - Remove bare `except Exception: pass` in `swarm/breeder_daemon_v2.py`

2. **Week 2 — Complexity (P0/P1)**
   - Decompose `swarm/breeder_daemon_v2.py:step()` into 6–8 private pipeline methods
   - Extract `simulators/tournament_core.py` from duplicated simulators

3. **Week 3 — Duplication (P1)**
   - Create `FleetStatsMixin` and migrate 47 fleet modules
   - Move HTML templates from `logos/tide_pool_viz.py` and `fleet/holodeck.py` to `.html` files

4. **Week 4 — Polish (P1/P2)**
   - Add type hints to `simulators/` and `scripts/demo_*.py`
   - Extract `SimulationConfig`, `AudioConfig`, `VisionConfig` dataclasses
   - Clean up dead imports and `__init__.py` exports
   - Replace `os.system()` with `shutil.which()` in `sunset/codegen.py`

---

*End of report.*
