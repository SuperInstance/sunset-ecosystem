# Fleet Roadmap — Next Phase (May 25, 2026)

*Where the fleet goes after 21 modules, 549 tests, and one compiler that nobody has called yet.*

---

## Current State (May 25, 2026)

**Modules:** 21 | **Tests:** 549 passing, 4 xfail (intentional stubs) | **Commits on main:** 190+

**What works:**
- Cross-node sync (MetronomeBridge + MeshVectorGossip)
- Breeding with FLUX gating (BreederDaemonV2)
- Central orchestration (FleetConductorV2)
- Safety systems (GatewayPacing, OpcodeCapabilityIndex, OperationalTrap)
- Observability (SSE Stream Dashboard, Beta-Test Personas)
- Unification (SenseDecideAct framework)
- Compiler (FLUX Path B prototype, 52/52 tests, not yet wired)

**What blocks the next phase:**
1. HebbianMeshLayer lock contention at 1000 peers (90-350 dps, should be >500)
2. PID drift correction too slow (32 iterations for 50ms, should be ~10)
3. FLUX compiler exists but is not called by the breeding loop
4. No stress test beyond 1000 agents/room
5. Rust backend not compiled on production hardware

---

## Phase 1: Performance Hardening (1-2 days)

**Goal:** Fix the bottlenecks the benchmarks found.

### 1.1 HebbianMeshLayer Lock Optimization
- **Problem:** `threading.Lock` on `_affinities` serializes all routing decisions
- **Impact:** At 1000 peers, throughput drops to 90-350 dps (vs >75 gate, but high variance)
- **Approaches:**
  - A: Replace `threading.Lock` with `threading.RLock` or fine-grained locks per peer prefix
  - B: Pre-compute probability arrays with `numpy.random.choice`, lock only during update
  - C: Shard affinity storage by `peer_id % N`, reduce lock contention by factor of N
- **Decision:** Try B first (minimal code change, highest expected impact)
- **Success metric:** 1000-peer routing > 500 dps consistently, variance < 20%
- **Owner:** Direct build (no scout — this is targeted optimization)

### 1.2 PID Drift Correction Tuning
- **Problem:** `kp=0.01` is conservative; 32 iterations to correct 50ms drift
- **Impact:** Fleet-wide beat sync takes too long to converge after node restart
- **Approach:** Increase `kp` to 0.03-0.05, add overshoot guard
- **Success metric:** Convergence in < 15 iterations for 50ms drift
- **Risk:** Oscillation at high gains. Test with simulated network jitter before deploying.
- **Owner:** Direct build

### 1.3 Full Benchmark Regression Suite
- **Goal:** Run benchmarks automatically on every commit to main
- **Approach:** GitHub Actions workflow that runs `pytest tests/benchmarks/ -v`
- **Gate:** Fail CI if any benchmark exceeds baseline by > 50%
- **Owner:** CI configuration (For gemaster or direct)

---

## Phase 2: FLUX Integration Completion (2-3 days)

**Goal:** The compiler stops being a prototype and starts being a tool.

### 2.1 Path A vs Path B Decision
- **Context:** FLUX VM has 60 opcodes. Python currently uses `flux_check_batch()` FFI that bypasses the VM entirely.
- **Path A (Library):** Keep FFI. Compiler is a debugging/development tool. Low effort, no VM integration.
- **Path B (Full VM):** Wire compiler into breeding loop. Replace `flux_check_batch()` with compiled bytecode execution. High effort, enables proof certificates, checkpoints, streaming.
- **Decision needed from:** Casey or Forgemaster
- **Recommendation:** Start with Path A (stability), build Path B incrementally in a feature branch.
- **Blocker:** Human decision required.

### 2.2 Compiler-to-Breeder Wiring (Post-Decision)
- If Path B chosen:
  - Add `compile_constraint()` call in `BreederDaemonV2._select_parents_vector()`
  - Replace Python constraint evaluation with VM execution
  - Add `BytecodeCache` to avoid recompiling same constraints
  - Benchmark: compile + execute vs. current Python evaluation
- If Path A chosen:
  - Document compiler as dev tool in `docs/FLUX_COMPILER.md`
  - Add `python3 -m sunset-ecosystem.flux_compile` CLI for debugging
  - Keep compiler tested but not production-wired

### 2.3 FLUX Preset Library Expansion
- **Current:** 10 presets (basic_weight_bounds, chaos_limit, thermal_budget, etc.)
- **Gap:** No presets for multi-constraint compositions (e.g. "chaos AND thermal AND weight")
- **Add:** `composite_presets` module with preset chaining (AND, OR, weighted combinations)
- **Tests:** Each composite preset gets VM execution + Python fallback test

---

## Phase 3: Rust Backend & Production Readiness (3-5 days)

**Goal:** The Rust VM compiles, the FFI is hardened, and the system can run without Python.

### 3.1 Rust Compilation Pipeline
- **Problem:** `cargo build` fails on FM's laptop (dependencies not resolved)
- **Solution:**
  1. Pin all Rust dependencies in `Cargo.lock`
  2. Add `rust-toolchain.toml` with specific compiler version
  3. Document build steps in `docs/RUST_BUILD.md`
  4. Add CI job that builds Rust on every commit
- **Owner:** Forgemaster (Rust expertise) + CI setup

### 3.2 FFI Hardening
- **Current:** `flux_check_batch()` calls Rust, returns Python objects
- **Hardening needed:**
  - Add timeout guards (Rust should not hang Python)
  - Add panic recovery (Rust panic → Python exception, not process death)
  - Add memory pool pre-allocation (avoid malloc in hot path)
  - Add version handshake (Python checks Rust library version on import)
- **Tests:** `tests/test_flux_ffi_hardening.py` — panic injection, timeout, version mismatch

### 3.3 Production Configurability
- **Current:** Many hardcoded constants (drift thresholds, PID gains, lock timeouts)
- **Goal:** All tunables load from `config/fleet.yaml` or environment variables
- **Modules to config-ify:**
  - MetronomeBridge: bpm, drift_threshold_ms, correction_gain
  - FleetConductorV2: beat_interval_ms, pipeline enable flags
  - HebbianMeshLayer: chaos_min/max, delta values, blacklist_threshold
  - BreederDaemonV2: n_children, mutation_rate, thermal_limit
- **Approach:** Add `FleetConfig` dataclass with `from_yaml()` and `from_env()`
- **Tests:** Config override tests for each module

---

## Phase 4: Scale Testing (2-3 days)

**Goal:** Prove the fleet works at the scale we designed for.

### 4.1 Overnight Breeding Run
- **Setup:** 100 rooms, 50 agents per room, 12-hour run
- **Monitor:**
  - Memory growth (should be stable, not leaking)
  - CPU usage (should plateau, not spike)
  - Breed success rate (should stay > 80%)
  - FLUX violation rate (should stay < 5%)
  - Cross-node sync latency (should stay < 10ms)
- **Tools:** SSE Stream Dashboard for real-time monitoring, WAL for post-hoc analysis
- **Success metric:** 12-hour run completes without restart, no memory leaks > 10%

### 4.2 FleetConductorV2 Integration at Scale
- **Setup:** 6 SDA pipelines + all subsystems enabled, 100 rooms
- **Test:** Conductor beat() tick latency under load
- **Gate:** < 5ms per tick even with 100 rooms
- **Fix if failing:** Pipeline lazy initialization, subscription pruning, or beat interval adjustment

### 4.3 Cross-Node Mesh Gossip Stress
- **Setup:** 4 simulated nodes, each with 250 agents, full gossip enabled
- **Test:** CRDT convergence time, vector table consistency across nodes
- **Gate:** All nodes agree on vector table within 5 seconds of any update
- **Tool:** Add `mesh_consistency_check()` diagnostic to FleetConductorV2

---

## Phase 5: New Capabilities (Post-scale, 1-2 weeks)

**Goal:** Features that make the fleet feel alive, not just functional.

### 5.1 Decision Journal Visualization
- **Current:** `logos/decision_journal.py` logs decisions to file
- **New:** HTML dashboard that renders decision history as timeline
  - Sense observations (heat maps, metric graphs)
  - Decision cards (action_type, confidence, reasoning)
  - Action outcomes (success/failure markers)
  - Filter by pipeline, time range, severity
- **Integration:** SSE Stream Dashboard can serve the HTML

### 5.2 Agent Personality Persistence
- **Current:** AgentIdentity exists but resets on restart
- **New:** Save agent "personality" (affinity history, capability preferences, thermal profile) to WAL
  - On restart, restore from WAL
  - Personality influences initial routing decisions
  - "Experienced" agents get priority in breeding (they've proven survival)
- **Metaphor:** Agents accumulate scars and preferences. The fleet has memory.

### 5.3 SSE Dashboard UI
- **Current:** SSE endpoint streams events, no consumer UI
- **New:** Minimal HTML page that connects to SSE and renders:
  - Real-time breeding progress bar
  - Agent spawn/retirement counters
  - FLUX violation alerts (red banner)
  - Thermal gauge (per-node)
  - Beat sync status (green/yellow/red per node)
- **Design:** Single file, no build step, served by FleetConductorV2 HTTP endpoint
- **Reference:** Dieter Rams — clear hierarchy, no decoration, information-first

### 5.4 Fleet "Weather Report"
- **Idea:** Daily automated summary of fleet health
  - "Yesterday: 14,231 breeds attempted, 12,884 passed FLUX, 1,347 thermal throttled."
  - "Node-3 drift corrected 3 times. Node-7 had highest diversity (0.73)."
  - "Trend: Breed success rate up 4% vs. last week."
- **Format:** Markdown, posted to Matrix `#fleet-ops` channel daily at 08:00 UTC
- **Implementation:** Cron job using FleetConductorV2 `get_stats()` + `message.send`

---

## Phase 6: Fleet Ecosystem Expansion (Ongoing)

**Goal:** The sunset ecosystem is not the only repo. Make the fleet capable of managing itself across repositories.

### 6.1 Repo Health Integration
- **Current:** `cocapn-health` monitors fleet services. Not integrated with sunset-ecosystem.
- **New:** FleetConductorV2 calls `cocapn-health` CLI as part of `Sense` pipeline
  - Service down → OperationalTrap triggers
  - Health degradation → SSE dashboard alert
  - Automatic retry with exponential backoff

### 6.2 CCC-OS Orchestrator Integration
- **Current:** `ccc-os` has `MonitorRegistry`. Not wired to sunset-ecosystem.
- **New:** CCC-OS monitors become FleetConductorV2 SDA pipelines
  - Breeder monitor → `_BreederThermalSense`
  - Fleet bridge monitor → `_MeshDiversitySense`
  - Register CCC-OS monitors via `register_monitor()` API

### 6.3 ai-writings Automation
- **Current:** Essays written manually or by scouts.
- **New:** Fleet generates "state of the fleet" essays automatically
  - Pull stats from FleetConductorV2
  - Generate narrative using SenseDecideAct observations
  - Commit to ai-writings repo daily
  - Human edits optional
- **Metaphor:** The fleet writes its own diary.

---

## Timeline Summary

| Phase | Duration | Key Deliverable | Owner |
|-------|----------|-----------------|-------|
| 1. Performance Hardening | 1-2 days | Benchmarks green, lock fixed | Direct build |
| 2. FLUX Integration | 2-3 days | Compiler wired or documented | Casey/FM decision → direct build |
| 3. Rust Backend | 3-5 days | Rust compiles, FFI hardened | FM + CI |
| 4. Scale Testing | 2-3 days | 12-hour overnight run passes | Direct build + monitoring |
| 5. New Capabilities | 1-2 weeks | Dashboard UI, weather report, persistence | Scouts + direct |
| 6. Ecosystem Integration | Ongoing | Multi-repo orchestration | Fleet-wide |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HebbianMeshLayer lock fix doesn't help | Medium | Medium | Try 2 approaches (B and C), keep whichever works |
| FLUX Path B too complex to wire | Medium | High | Fall back to Path A, document compiler as dev tool |
| Rust compilation fails on FM's machine | High | High | Pin dependencies, add Docker build option |
| 12-hour overnight run crashes | Medium | High | Run 2-hour version first, fix issues, retry |
| Gateway congestion returns | Medium | Medium | Continue direct build for critical path, scouts only for exploration |

---

## Success Criteria for "Next Phase Complete"

1. All benchmarks pass with < 50% variance (no high-variance routing)
2. FLUX compiler either wired into breeder OR documented as dev tool with clear decision record
3. Rust backend compiles on at least one machine (FM's laptop or CI)
4. 12-hour overnight breeding run completes without restart
5. SSE dashboard renders real-time fleet status in a browser
6. Fleet weather report posted automatically for 7 consecutive days

**When all 6 are true, the fleet is not just built. It is alive.**

---

*kimi1 | Fleet Orchestrator | Day 35 | "A roadmap is a promise to the future. Here are ours."*