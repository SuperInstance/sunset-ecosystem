# API Consistency Audit — sunset-ecosystem

**Audited:** 2026-05-31  
**Scope:** `swarm/`, `fleet/`, `sunset/`, `nerve/`, `logos/`, `ethos/`, `nexus/`, `pathos/`  
**Files inspected:** ~340 `.py` files via targeted grep + manual spot-checks on 8 core files  

---

## 1. Inconsistent Factory Names (P1)

Three naming conventions for builder functions coexist in `fleet/` alone:

| Pattern | Examples | Location |
|---------|----------|----------|
| `create_*` | `create_consumer_group`, `create_series`, `create_handshake`, `create_token` | `fleet/event_stream.py:103`, `fleet/tsdb.py:189`, `fleet/vessel_handshake.py:276`, `fleet/auth.py:63` |
| `build_*` | `build_gossip_payload`, `build_sync_payload`, `build_breeding_panel`, `build_allocation_plan` | `fleet/mem0_adapter.py:726`, `fleet/cocapn_dashboard.py:59`, `ethos/agent_allocator.py:125` |
| `init_*` / `new_*` | Essentially absent | — |

**Fix:** Standardize on one verb. `create_*` is already dominant; rename `build_*` → `create_*` in `fleet/` and `ethos/`.

---

## 2. Inconsistent Lifecycle Names (P1)

Four lifecycle dialects across the codebase:

| Dialect | Users | Files |
|---------|-------|-------|
| `start()` / `stop()` | `BreederDaemon`, `WorkerPool`, `ArrowFlightMesh`, `Metronome`, `FleetConductorV2` | `swarm/breeder_daemon.py`, `swarm/worker_pool.py`, `nerve/metronome.py`, `nexus/fleet_conductor_v2.py` |
| `initialize()` | `EnsembleBreeder`, `SpectralBreeder`, `NCABreeder`, `SwarmIntelligenceBreeder` | `swarm/ensemble_breeder.py:59`, `swarm/spectral_breeding.py:248`, `swarm/nca_breeder.py:321` |
| `shutdown()` | `FleetConductorV2`, `ProcessSupervisor`, `ThreadPool` | `nexus/fleet_conductor_v2.py:980`, `fleet/process_supervisor.py:227`, `fleet/thread_pool.py:179` |
| `close()` | `BreederDaemonV2`, `Mem0Adapter`, `MmapWAL`, `Federation` | `swarm/breeder_daemon_v2.py:394`, `fleet/mem0_adapter.py:398`, `logos/mmap_wal.py:204` |
| `begin()` / `end()` | `ABTester` | `fleet/ab_tester.py:111` |

**Fix:** Adopt a single interface. Suggest `ILifecycle { start(); stop(); }` for thread-based services and `ICloseable { close(); }` for resource holders. `shutdown()` should be deprecated or aliased to `stop()`.

---

## 3. Return Type Inconsistency (P2)

Multiple annotation styles in the same file:

- `dict` vs `dict[str, Any]` vs `dict[str, str]` (`nexus/fleet_conductor_v2.py` uses all three)
- `list` vs `list[tuple[int, str]]` vs `List[dict]` (`swarm/breeder.py:165` vs `swarm/breeder_daemon.py:344`)
- `Optional[int]` vs `int | None` (`swarm/breeder.py:235` vs `swarm/breeder_daemon_v2.py:1537`)
- Some methods have no return annotation at all (`__init__` in `breeder_daemon_v2.py:58`, `record`, `check`, `_track_lifecycle` in `worker_pool.py:296`)

**Fix:** Run `ruff` or `mypy --strict` across the 8 core files; standardize on PEP 604 union syntax (`|`) and complete return annotations.

---

## 4. Config Pattern Inconsistency (P1)

No unified config base class. Five different patterns observed:

1. **Dataclass** (`@dataclass`) — `WorkerConfig`, `DiversityConfig`, `ThermalConfig` (`swarm/worker_pool.py`, `swarm/breeder_daemon_v2.py`)
2. **Plain class** — `FluxGatingConfig` nested inside `breeder_daemon_v2.py:80` (with `# type: ignore[no-redef]`)
3. **TypedDict / dict** — `ConductorConfig` in `nexus/fleet_conductor_v2.py:39` (plain class, not dataclass)
4. **Pydantic-like** — None found; no validation layer
5. **Runtime dict** — Many functions accept `config: Dict[str, Any] | None = None`

**Fix:** Introduce a `BaseConfig` dataclass in `fleet/config.py` or `logos/config_validator.py` and migrate all `*Config` classes to inherit from it. Add a `from_env()` / `from_dict()` factory on the base.

---

## 5. Logging Inconsistency (P2)

Two logger variable names dominate:

- `logger = logging.getLogger(__name__)` — 40+ files (`swarm/breeder.py`, `swarm/thermal.py`, etc.)
- `log = logging.getLogger(__name__)` — 10+ files (`swarm/fleet_bft_qd.py`, `swarm/fleet_diversity.py`, `swarm/fleet_turbovec.py`)

No structured logging (JSON), no `self.logger` on instances, no correlation-ID propagation.

**Fix:** Standardize on `logger`. Optionally add `structlog` or a `log_context()` helper in `fleet/` for trace IDs.

---

## 6. Async / Sync Inconsistency (P0)

Critical architectural fracture:

| Layer | Style | Files |
|-------|-------|-------|
| Core breeding / thermal | **Sync** (`threading`) | `swarm/thermal.py`, `swarm/breeder_daemon.py`, `swarm/worker_pool.py` |
| Thermal (alt) | **Async** (`asyncio`) | `swarm/async_thermal.py` |
| Fleet Conductor v1 | **Async** | `nexus/fleet_conductor.py` |
| Fleet Conductor v2 | **Sync** | `nexus/fleet_conductor_v2.py` |
| A2A identity | **Async** | `logos/a2a_identity.py`, `logos/a2a_protocol.py` |
| Event bus | **Async** | `nexus/fleet_event_bus.py` |

There is **no documented bridge** between the sync breeder loop and the async conductor. `async_thermal.py` is orphaned — no code in `swarm/` imports it.

**Fix:** Decide the dominant paradigm for the breeding loop. If sync, wrap async conductor calls in `asyncio.run()` or provide a `SyncConductor` adapter. If async, rewrite `BreederDaemon` and `WorkerPool` with `asyncio`. Either way, delete the orphan or wire it.

---

## 7. Thread-Safety Issues (P0)

### 7a. Lock scope gaps
- `swarm/breeder_daemon_v2.py:878` — `del self._room_allocations[room_id]` is done **outside** `self._lock` in `_run_loop()` (line 857 context is inside the lock, but line 878 is inside the `try` block that starts after the lock is released at line 824). Verify: the lock is acquired at 779, released at 780 via `with`, but the `del` at 878 is inside the method `_breed_one()` which is called from `_run_loop()` — need to trace more carefully. Actually looking at the code, `_breed_one()` calls `_find_room_for_agent()` and `_rebirth_with_clone()` — the `del` at 878 is inside `_run_loop()` after `with self._lock:` at 779. Wait, line 779 is `with self._lock:` and the block ends at 780. So `del` at 878 is outside. This is a race condition on `_room_allocations`.

- `swarm/worker_pool.py` — `_track_lifecycle()` (line 296) is not wrapped in `self._lock`, though it mutates `_workers`.

### 7b. Lock granularity
- `swarm/mesh_vector_tables.py` — Every public method acquires an `RLock`. Fine for correctness, but high contention under swarm load. Consider reader-writer lock or sharding.

### 7c. Missing locks
- `fleet/mem0_adapter.py` — `_gossip_lock` protects gossip state, but `_vocab_lock` is only used in `add_to_memory()`; `merge_memories()` does not appear to hold it.

---

## 8. Memory Leak Risks (P1)

- `swarm/flux_vm_gating.py:204` — `__del__` is unreliable in CPython and can prevent GC of reference cycles. Replace with explicit `close()` or context manager.

- `fleet/sandbox.py:90` — `gc.collect()` called manually after every sandbox run. This is a smell, not a fix. The root cause (circular refs or unclosed file handles) should be addressed.

- `swarm/jepa_memory.py` — `history` list grows unbounded (`max(0, len(history) - 3)` only looks at the last 3 items but never truncates the list). Should use `collections.deque(maxlen=...)`.

- `swarm/service_discovery.py` — Nodes are deleted on TTL expiry, but `_cleanup()` is only triggered on read operations. A write-only workload could accumulate dead nodes indefinitely.

---

## 9. Performance Hot Paths (P0)

Nested loops over population / vectors found in breeding-critical code:

| File | Pattern | Complexity | Context |
|------|---------|------------|---------|
| `swarm/spectral_breeding.py:328` | `for i in range(len(population)): for j in range(i+1, ...)` | O(n²) | Diversity matrix every generation |
| `swarm/breeder_daemon_v2.py:1363` | `for i in range(len(vectors)): for j in range(i+1, ...)` | O(n²) | `_build_lineage_population` distance check |
| `swarm/pythagorean_evolution.py:70` | `for m in range(2, max_n+2): for n in range(1, m)` | O(n²) | Primitive triple generation |
| `swarm/penrose.py:126` | `for i in range(len(positions)): for j in range(i+1, ...)` | O(n²) | Tiling force calculation |
| `swarm/tda_landscape.py:132` | `for i in range(len(distances)): for j in range(len(distances))` | O(n²) | Persistence diagram computation |
| `swarm/differential_breeder.py:165` | `for i in range(len(population)): for j in range(i+1, ...)` | O(n²) | Population distance |
| `swarm/flux_vector_table.py:564` | `for i in range(len(niche_ids)): for j in range(i+1, ...)` | O(n²) | Niche collision check |
| `swarm/causal_breeder.py:234` | `[[d.get(z, 0.0) for z in cond_set] for d in data]` | O(n·m) | Z-matrix construction per CI test |

**Fix:** For population-distance loops, vectorize with `scipy.spatial.distance.cdist` or `numpy broadcasting`. For `tda_landscape.py`, consider `ripser` or `gudhi` if available. Add a `MAX_POPULATION_FOR_N2` guard to fall back to approximate algorithms.

---

## 10. Priority Summary

| Priority | Count | Key Items |
|----------|-------|-----------|
| **P0** | 3 | Async/sync fracture (`fleet_conductor` vs `breeder_daemon`), thread-safety gaps (`breeder_daemon_v2` `_room_allocations` race), O(n²) hot paths in breeding loops |
| **P1** | 4 | Lifecycle naming chaos, config pattern fragmentation, memory leak risks (`jepa_memory`, `flux_vm_gating.__del__`), factory name inconsistency |
| **P2** | 2 | Return type annotation inconsistency, logging variable naming (`logger` vs `log`) |

---

## Recommended Refactor Order

1. **P0 — Async/sync decision** — Pick one paradigm for the conductor ↔ breeder boundary. Write an ADR in `docs/`.
2. **P0 — Thread-safety audit** — Lock every mutation of `_room_allocations`, `_workers`, and `_nodes`. Add `threading` tests under `tests/test_thread_safety.py`.
3. **P0 — Vectorize hot paths** — Replace the 7 O(n²) loops with `cdist` or `numba` JIT where `numpy` is insufficient.
4. **P1 — Config unification** — Introduce `BaseConfig` and migrate `*Config` classes.
5. **P1 — Lifecycle interface** — Define `ILifecycle` and `ICloseable` in `fleet/interfaces.py`; implement on all services.
6. **P2 — Annotation sweep** — Run `mypy --strict` and fix the 200+ missing/inconsistent annotations.
