# Test Infrastructure Audit — Sunset Ecosystem

**Auditor:** CCC (Test Infrastructure Auditor)  
**Date:** 2026-05-31  
**Scope:** Full pytest suite under `sunset-ecosystem/tests/`  

---

## Executive Summary

The sunset-ecosystem test suite is **large but brittle**.  
- **7,194 tests** across 393 files (~78 KLOC of test code).  
- **43 source modules** have no corresponding test coverage.  
- **Critical import error** in `fleet/__init__.py` blocks entire breeder-daemon test branch.  
- Heavy use of `time.sleep()` and real network I/O makes the suite slow and flaky in CI.  
- Zero property-based or fuzz testing.  
- Flat test directory with no source-structure mirroring.

| Metric | Value |
|--------|-------|
| Total tests collected | 7,194 |
| Test files | 393 |
| Test code lines | ~78,202 |
| Collection time | 56.45 s |
| Source `.py` files | 418 |
| Untested modules (no overlap) | 43 |
| Files using `unittest.mock` | 66 |
| `xfail` markers | 2 |
| `skip` markers (non-optional) | 6 |
| `skipif` markers (optional deps) | 20+ |

---

## 1. Test Coverage by Module

### 1.1 Well-covered domains
Most core swarm, fleet, nerve, and compiler modules have dedicated test files:
- `swarm/breeder.py` → `test_breeder.py` (24 tests, all pass)
- `sunset/compiler.py` → `test_compiler.py` (12 tests, 3 skipped for numba)
- `nerve/room_grid.py` → `test_room_grid.py` (13 tests, 1 skip)
- `fleet/event_bus.py` → `test_event_bus.py`, `test_fleet_event_bus.py`
- `logos/decision_journal.py` → `test_decision_journal.py` + `test_decision_journal_integration.py`

### 1.2 Clearly untested modules (43)
These source files have **zero test file name overlap** and no clear indirect test coverage:

| Module | Path | Domain |
|--------|------|--------|
| `handlers` | `a2a/handlers.py` | A2A protocol |
| `agent_allocator` | `ethos/agent_allocator.py` | Ethos orchestration |
| `trinity_connection` | `ethos/trinity_connection.py` | Trinity bridge |
| `notifier` | `fleet/notifier.py` | Fleet notifications |
| `opcode_map` | `flux_compat/opcode_map.py` | FLUX compat |
| `v2_bytecode` | `flux_compat/v2_bytecode.py` | FLUX compat |
| `v3_module` | `flux_compat/v3_module.py` | FLUX compat |
| `flux_vm_compat` | `flux_vm_compat.py` | FLUX VM |
| `core` | `grammar/core.py` | Grammar engine |
| `decision_log` | `logos/decision_log.py` | Logos persistence |
| `trinity_connection` | `logos/trinity_connection.py` | Logos bridge |
| `a2a_conductor_integration` | `nerve/a2a_conductor_integration.py` | Nerve A2A |
| `templates` | `nerve/templates.py` | Nerve |
| `interaction_log` | `pathos/interaction_log.py` | Pathos |
| `moment_scorer` | `pathos/moment_scorer.py` | Pathos |
| `need_tracker` | `pathos/need_tracker.py` | Pathos |
| `trinity_connection` | `pathos/trinity_connection.py` | Pathos bridge |
| `audio_capture` | `perception/audio_capture.py` | Perception |
| `capture` | `perception/capture.py` | Perception |
| `personalization` | `ranking/personalization.py` | Ranking |
| `ranked_response` | `ranking/ranked_response.py` | Ranking |
| `user_ranking` | `ranking/user_ranking.py` | Ranking |
| `python_bridge` | `reasoning/python_bridge.py` | Reasoning |
| `hardware_load_profiler` | `simulators/hardware_load_profiler.py` | Simulators |
| `hardware_swarm_lite` | `simulators/hardware_swarm_lite.py` | Simulators |
| `sweep` | `simulators/sweep.py` | Simulators |
| `flux_codegen` | `sunset/flux_codegen.py` | Sunset codegen |
| `generation_runner` | `sunset/generation_runner.py` | Sunset runner |
| `seed_bank` | `sunset/seed_bank.py` | Sunset storage |
| `tensor_archive` | `sunset/tensor_archive.py` | Sunset archive |
| `trinity_scorer` | `sunset/trinity_scorer.py` | Sunset scoring |
| `superinstance_ffi_mock` | `superinstance_ffi_mock.py` | FFI mock |
| `superinstance_ffi_real` | `superinstance_ffi_real.py` | FFI real |
| `broadcast` | `swarm/broadcast.py` | Swarm messaging |
| `crdt_hdc_hybrid` | `swarm/crdt_hdc_hybrid.py` | Swarm CRDT |
| `flux_vm_runner` | `swarm/flux_vm_runner.py` | Swarm VM |
| `hardware_index` | `swarm/hardware_index.py` | Swarm hardware |
| `superinstance_ffi` | `swarm/superinstance_ffi.py` | Swarm FFI |
| `swarm_runner` | `swarm/swarm_runner.py` | Swarm runner |
| `duplicate_detect` | `triage/duplicate_detect.py` | Triage |
| `github_issues` | `triage/github_issues.py` | Triage |
| `repo_duplicate` | `triage/repo_duplicate.py` | Triage |
| `weekly` | `triage/weekly.py` | Triage |

> **Note:** `ranking/` and `distill/` modules are partially covered by the catch-all `test_new_modules.py`, but that file bundles 10+ unrelated modules into one flat test class. It is not a substitute for dedicated, focused tests.

---

## 2. Slow Tests (>1 s)

### 2.1 Identified slow calls from sampled runs

| Test | Duration | Why slow |
|------|----------|----------|
| `test_room_grid.py::test_tick_returns_stats` | 1.39 s | Likely breeds/compiles many rooms |
| `test_compiler.py::test_compiler_skips_numba` | 0.76 s | Numba JIT compilation warm-up |
| `test_bernstein_orchestrator.py::test_entries_property` | ~10 s | `time.sleep(10)` inside test |

### 2.2 Tests containing `time.sleep()` or `asyncio.sleep()`

These are **performance traps** in CI. Every `sleep()` adds wall-clock time that cannot be parallelized away.

- `test_arrow_telemetry_adapter.py` — 8 instances (0.1–0.2 s each)
- `test_process_supervisor.py` — 5 instances (0.2–0.5 s each)
- `test_thread_pool.py` — 7 instances (0.1–1.0 s each)
- `test_bernstein_orchestrator.py` — 1 instance (`time.sleep(10)`)
- `test_conflict_resolver.py` — 2 instances (0.01 s each)
- `test_snapshot_manager.py` — 1 instance (0.01 s)
- `test_lifecycle_fsm.py` — 2 instances (0.01 s)
- `test_arrow_flight_mesh.py` — 2 instances (0.001 s)
- `test_fleet_conductor_stress.py` — 1 instance (`asyncio.sleep(0.05)`)

**Recommendation:** Replace `time.sleep()` with `freezegun` or mocked clocks. The `time.sleep(10)` in `test_bernstein_orchestrator.py` alone costs more than the entire `test_cache.py` + `test_event_bus.py` + `test_config.py` suite combined.

### 2.3 Benchmark tests polluting the default run

`tests/benchmarks/test_fleet_performance.py` contains performance tests with `@pytest.mark.parametrize` over large peer/agent counts (100–1000). There is **no `slow` marker** on these tests, so they run with the default suite.

---

## 3. Flaky Tests

| Test file | Marker | Reason |
|-----------|--------|--------|
| `test_flux_gating.py:316` | `xfail` | BreederDaemonV2 integration stub — cycle does not call flux checker yet |
| `test_flux_gating.py:339` | `xfail` | Same as above |
| `test_room_grid.py:76` | `skip` | `diversity()` returns non-zero after single tick — needs algorithm review |
| `test_observer_breeder_integration.py:398` | `skip` | WAL replay transitions agent to SUNSET during replay — needs lifecycle timing fix |
| `test_lineage_checker.py:270` | `skip` | Uses fixtures from `test_breeder_daemon_v2`; cannot run standalone |
| `test_breeder_flux_integration.py:340` | `skip` | Hangs in pytest fixture context; verified manually |
| `test_breeder_flux_integration.py:355` | `skip` | Same |
| `test_breeder_flux_integration.py:372` | `skip` | Same |

**Critical:** `test_breeder_flux_integration.py` is **completely non-functional** because it imports `swarm.breeder_daemon_v2`, which fails due to the `fleet/__init__.py` `Action` import error (see §7). The skip annotations are moot — the file cannot even be collected.

---

## 4. Missing Test Patterns

| Pattern | Status | Evidence |
|---------|--------|----------|
| **Property-based (Hypothesis)** | ❌ Absent | Zero `@given` or `hypothesis` imports |
| **Fuzz testing** | ❌ Absent | No fuzz harnesses or generative inputs |
| **Load / stress testing** | ⚠️ Partial | `tests/benchmarks/` exists but is not a formal load suite; `test_fleet_conductor_stress.py` exists but is small-scale |
| **Mutation testing** | ❌ Absent | No `mutmut` or equivalent |
| **Contract / invariant testing** | ⚠️ Sparse | Only `test_adversarial_arena.py::test_zero_sum_property` uses the word "property" |
| **Snapshot / approval testing** | ❌ Absent | No `syrupy` or similar |
| **Parametrize usage** | ⚠️ Very low | Only 10 `@pytest.mark.parametrize` decorators across 393 test files |

**Recommendation:** Add Hypothesis for breeder mutation invariants, room-grid state transitions, and consensus ring convergence. Parametrize the 50+ scalar test files that currently copy-paste the same assertion with different constants.

---

## 5. Test Duplication

### 5.1 Fixture duplication
- `np.random.seed(42)` appears in **4 fixtures** inside `tests/conftest.py` (`room_grid_100`, `room_grid_1000`, `routing_layer`, `signal_64`).
- Root `conftest.py` mocks `turbovec.IdMapIndex` with a full 70-line class that duplicates behavior found in `swarm/vector_table.py`.
- Root `conftest.py` also mocks `plato_core` with 120+ lines of `_MockTrainingTile`, `_MockLamportClock`, etc. — these are repeated nowhere else but are fragile stand-ins.

### 5.2 Setup code duplication
- Multiple test files re-invent `tmpdir` + `git init` + `git commit` sequences (`test_metrics.py`, `test_drift_detect.py`).
- `test_fleet_conductor_v2.py`, `test_fleet_conductor_stress.py`, and `test_conductor_breed_coordination.py` all define nearly identical `conductor` fixtures with slightly different constructor arguments.

### 5.3 Catch-all test file
- `test_new_modules.py` (377 lines) tests **10 unrelated modules** (`ranking.*`, `distill.*`, `swarm.broadcast`, `swarm.swarm_runner`, `nerve.fiber`). This violates the one-module-per-test-file convention and makes failures hard to map.

---

## 6. Integration Test Gaps

### 6.1 Existing integration tests (14 files)
- `test_breeder_bft_qd_integration.py`
- `test_breeder_flux_integration.py` **(broken — cannot collect)**
- `test_breeder_integration.py`
- `test_compiler_integration.py`
- `test_cross_ecosystem_integration.py`
- `test_cross_repo_integration.py`
- `test_decision_journal_integration.py`
- `test_e2e_consensus_persist.py`
- `test_fleet_conductor_v2_integration.py`
- `test_flux_integration.py`
- `test_hot_swap_integration.py`
- `test_metronome_integration.py`
- `test_observer_breeder_integration.py`
- `test_room_grid_integration.py`
- `test_spread_integration.py`

### 6.2 Untested cross-module interactions
| Interaction | Gap |
|-------------|-----|
| `swarm.breeder_daemon_v2` ↔ `fleet.operational_trap` | Broken import; zero runtime integration tests |
| `sunset.compiler_integration` ↔ `nerve.room_grid_tick_integration` | No joint test |
| `nexus.fleet_conductor_v2` ↔ `swarm.breeder_daemon_v2` | `test_fleet_conductor_v2_integration.py` exists but is not running because breeder_daemon_v2 import is broken |
| `a2a.server` ↔ `a2a.handlers` | `test_a2a_server.py` tests the server, but `a2a/handlers.py` has no tests at all |
| `logos/decision_journal.py` ↔ `logos/decision_log.py` | `test_decision_journal.py` tests journal; `decision_log.py` is completely untested |
| `pathos/*` ↔ any other module | All 4 `pathos/` modules untested in isolation; no integration tests exist |
| `perception/audio_capture.py` ↔ `voice/soniqo_bridge.py` | No end-to-end audio pipeline test |
| `triage/*` ↔ `github` / `fleet` | All 4 `triage/` modules untested; `test_review_code_ci.py` tests CI review but not triage logic |

### 6.3 Conditional skipifs create hidden gaps
`test_cross_ecosystem_integration.py` has **8 `skipif` markers**. If the optional deps (`tensor_spline`, `holonomy_bridge`, `constraint_theory`, `flux_check`, `seed_bank`, `zerolang`, `superinstance.runtime`, `plato_core`) are not installed on CI, the "integration" test file runs as a hollow shell.

---

## 7. Mock Quality

### 7.1 Good: centralized mocks
- Root `conftest.py` provides a **solid `turbovec` mock** — every test that imports `swarm.vector_table` or `sunset.turbovec` gets a deterministic in-memory index.
- Root `conftest.py` also provides a **full `plato_core` mock** so `pytest.importorskip("plato_core")` tests always execute.

### 7.2 Bad: real network I/O in tests
These tests hit actual endpoints or spawn real processes:

| File | Real I/O | Risk |
|------|----------|------|
| `test_claw_fleet_bridge.py` | `urllib.request.urlopen` to external URLs | Flaky if network down; slow if remote server throttles |
| `test_a2a_server.py` | `urllib.request.urlopen` to local server started by fixture | Port collisions; zombie processes if teardown fails |
| `test_grammar_server.py` | `subprocess.Popen` + `urllib.request.urlopen` | Same as above |
| `test_metrics.py` | `subprocess.run(["git", "init"])` | Requires `git` binary; leaves temp repos |
| `test_drift_detect.py` | `subprocess.run(["git", "init"])` | Same |

### 7.3 Bad: file system I/O without `tmp_path`
Some tests use `open()` with hard-coded or non-`pytest` temp directories. `test_bernstein_orchestrator.py` reads a real file at `path = "/tmp/..."`.

### 7.4 Missing mocks for hardware
- `test_cellular_gpu.py` and `test_cuda_kernels.py` skip when CUDA is absent — this is correct.
- However, `test_hardware_nas.py`, `test_hardware_profiler.py`, and `test_cuda_benchmark.py` are marked `@pytest.mark.slow` but are **not** skipped when the underlying hardware profiler is unavailable. They may fail silently on non-GPU runners.

---

## 8. `xfail` Tests

| File | Line | Reason | Fixable? |
|------|------|--------|----------|
| `test_flux_gating.py` | 316 | BreederDaemonV2 integration stub — cycle does not call flux checker yet | ✅ **Fixable** — wire `FluxConstraintChecker` into `BreederDaemonV2.cycle()` |
| `test_flux_gating.py` | 339 | Same as above | ✅ **Fixable** |

Both xfails are **known missing wiring**, not ambiguous heisenbugs. They should be promoted to tickets and resolved rather than left to rot.

---

## 9. Test File Organization

### 9.1 Flat directory
All 393 test files live in a single `tests/` directory. There is **no mirroring** of the source tree:

```
sunset-ecosystem/
  fleet/          ← 80+ modules
  swarm/          ← 50+ modules
  nerve/          ← 15+ modules
  tests/
    test_fleet_*.py   ← 80 files
    test_swarm_*.py   ← 50 files
    test_nerve_*.py   ← 15 files
```

### 9.2 Naming convention drift
- Most files follow `test_<module>.py`.
- `test_new_modules.py` breaks this — it should be split into `test_ranking_*.py`, `test_distill_*.py`, `test_swarm_broadcast.py`, `test_nerve_fiber.py`.
- `test_fleet_event_bus.py` and `test_event_bus.py` both exist; the former tests `fleet/event_bus.py`, the latter tests `nexus/event_bus.py`. This is confusing.

### 9.3 Benchmark directory not excluded
`tests/benchmarks/` is inside `testpaths` in `pytest.ini`. Benchmarks should either:
- Move to `benchmarks/` (sibling of `tests/`), or
- Be excluded with `addopts = --ignore=tests/benchmarks` in `pytest.ini`.

---

## 10. Priority Ranking

### P0 — Blocks CI / breaks collection

| Issue | Action | Owner hint |
|-------|--------|------------|
| `fleet/__init__.py` imports `Action` from `fleet.sense_decide_act` but the class does not exist | Add `Action` to `fleet/sense_decide_act.py` or remove the import | Fleet module owner |
| `test_breeder_daemon_v2.py` and `test_breeder_flux_integration.py` fail collection because of the above | Fix import; unskip the 3 "hangs" tests once collection works | Breeder owner |
| `test_claw_fleet_bridge.py` makes real urllib calls | Mock `urllib.request.urlopen` | Bridge owner |
| `test_grammar_server.py` spawns real subprocess | Use `pytest-httpserver` or mock `subprocess` | Grammar owner |

### P1 — Slows suite significantly

| Issue | Action |
|-------|--------|
| `time.sleep(10)` in `test_bernstein_orchestrator.py` | Replace with mocked clock or `freezegun` |
| 8 `time.sleep(0.1–0.2)` calls in `test_arrow_telemetry_adapter.py` | Inject mock telemetry buffer with synthetic timestamps |
| 5 `time.sleep(0.2–0.5)` calls in `test_process_supervisor.py` | Mock `psutil.Process` or use `pytest-timeout` + fast polling |
| Benchmark tests run in default suite | Add `--ignore=tests/benchmarks` to `pytest.ini` or add `slow` marker |
| 56 s collection time for 7,194 tests | Investigate heavy imports in `conftest.py` (root turbovec mock rebuilds on every collection) |

### P2 — Coverage & quality gaps

| Issue | Action |
|-------|--------|
| 43 untested modules (see §1.2) | Create `test_<module>.py` for each; start with `ranking/`, `pathos/`, `triage/` |
| Zero property-based tests | Add `hypothesis` to dev deps; seed with `RoomGrid` state-transition invariants |
| `test_new_modules.py` catch-all | Split into dedicated files |
| Only 10 `@pytest.mark.parametrize` uses | Parametrize scalar boundary tests across `fleet/`, `swarm/` |
| 8 `skipif` deps in `test_cross_ecosystem_integration.py` | Install optional deps on CI or mark the file as `integration` and run on a nightly job |
| Duplicate `np.random.seed(42)` in fixtures | Extract a single `seed_rng` autouse fixture |
| `tests/` flat directory | Reorganize into `tests/fleet/`, `tests/swarm/`, `tests/nerve/`, `tests/sunset/` mirroring source |

---

## Appendix A: Files Referenced

- `sunset-ecosystem/pytest.ini`
- `sunset-ecosystem/conftest.py` (root)
- `sunset-ecosystem/tests/conftest.py`
- `sunset-ecosystem/fleet/__init__.py`
- `sunset-ecosystem/fleet/sense_decide_act.py`
- `sunset-ecosystem/tests/test_flux_gating.py`
- `sunset-ecosystem/tests/test_breeder_flux_integration.py`
- `sunset-ecosystem/tests/test_breeder_daemon_v2.py`
- `sunset-ecosystem/tests/test_claw_fleet_bridge.py`
- `sunset-ecosystem/tests/test_new_modules.py`
- `sunset-ecosystem/tests/benchmarks/test_fleet_performance.py`
- `sunset-ecosystem/tests/test_bernstein_orchestrator.py`
- `sunset-ecosystem/tests/test_arrow_telemetry_adapter.py`
- `sunset-ecosystem/tests/test_process_supervisor.py`
- `sunset-ecosystem/tests/test_cross_ecosystem_integration.py`

---

*End of audit.*
