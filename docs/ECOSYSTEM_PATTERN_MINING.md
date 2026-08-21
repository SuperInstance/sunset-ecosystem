# Cross-Repo Pattern Mining — SuperInstance Ecosystem

*Last updated: 2026-06-01*

This document maps reusable patterns, techniques, and abstractions discovered across the SuperInstance ecosystem. Each pattern includes: origin repo, what it is, why it matters, and how to adopt it in sunset-ecosystem.

---

## Pattern Index

| # | Pattern | Origin Repo | Status | Bridge Module |
|---|---------|-------------|--------|---------------|
| 1 | **CheckResult Health Standard** | cocapn-health | ✅ Mined | `fleet/health_bridge.py` |
| 2 | **Agent Identity as Repo Structure** | git-agent-standard | ✅ Mined | `fleet/agent_identity_bridge.py` |
| 3 | **Fleet Consciousness Index** | fleet-consciousness-dashboard | ✅ Mined | `fleet/fleet_consciousness_bridge.py` |
| 4 | **Exact Pythagorean Snapping** | constraint-theory-core | ✅ Mined | `swarm/constraint_bridge.py` |
| 5 | **Laplacian Eigenvalue Fingerprints** | SuperInstance/SuperInstance | ✅ Mined | `fleet/conservation_spectral_bridge.py` |
| 6 | **Everything is a Tile** | OpenConstruct | 📋 Documented | — |
| 7 | **JEPA Gravity (Room DJ)** | OpenConstruct | 📋 Documented | — |
| 8 | **A2A Signal Language** | flux-spec / flux-a2a-signal | 📋 Documented | — |
| 9 | **Progressive Autonomy (L1→L5)** | OpenConstruct | 📋 Documented | — |
| 10 | **Penrose Cross-Room Correlation** | OpenConstruct | 📋 Documented | — |
| 11 | **Query Engine (12 Operators)** | cocapn-plato | 📋 Documented | — |
| 12 | **Divergence Monitoring (EMA)** | cocapn-plato | 📋 Documented | — |
| 13 | **Trust-Based Memory Sharing** | hierarchical-memory | 📋 Documented | — |
| 14 | **Hot-Swap A/B Testing** | flux-os | 📋 Documented | — |
| 15 | **Self-Compiler + HAL** | flux-os | 📋 Documented | — |
| 16 | **Swarm Coordination** | Equipment-Swarm-Coordinator | ✅ Mined | `fleet/swarm_coordinator_bridge.py` |
| 17 | **Agent Self-Model** | cuda-self-model | 📋 Documented | — |
| 18 | **FLUX ISA v3.0** | flux-spec | 📋 Documented | — |
| 19 | **Flux Conformance Vectors** | flux-conformance | 📋 Documented | — |
| 20 | **Fleet Workshop Pipeline** | fleet-workshop | 📋 Documented | — |
| 21 | **ISA Convergence** | fleet-workshop / datum | 📋 Documented | — |
| 22 | **Fence Board / Tom Sawyer** | oracle1-vessel | ✅ Mined | `fleet/fence_board_bridge.py` |
| 23 | **Heartbeat Protocol** | oracle1-vessel | ✅ Mined | `fleet/heartbeat_bridge.py` |
| 24 | **Fleet Task Board** | oracle1-vessel | ✅ Mined | `fleet/fleet_task_board_bridge.py` |

---

## Pattern 1: CheckResult Health Standard

**Origin:** `SuperInstance/cocapn-health`

**What it is:** A zero-dependency Python health check framework with standardized `CheckResult` dataclass (name, ok, latency_ms, status, details), 18 fleet service definitions with metric extraction, event bus bridge for transitions, and REST API with cache TTL.

**Why it matters:** Our `OperationalTrap` module detected health conditions but had no standardized reporting format. The `CheckResult` pattern gives us:
- A single data structure that every health checker returns
- Consistent markdown/JSON/oneline reporting
- Event-driven transitions (service_down, service_recovered)
- TTL caching to avoid hammering services
- Pure stdlib implementation (no dependencies)

**How we adopted it:** `fleet/health_bridge.py` — 38 tests, zero deps. Compatible with cocapn-health's data structures but standalone.

**Key API:**
```python
from fleet.health_bridge import HealthChecker, ServiceDef, CheckResult, FLEET_SERVICES

checker = HealthChecker(FLEET_SERVICES)
results = checker.check_all()
print(HealthChecker.report(results, format="md"))
```

---

## Pattern 2: Agent Identity as Repo Structure

**Origin:** `SuperInstance/git-agent-standard`

**What it is:** An agent's entire identity is a git repo with standard files:
- `CHARTER.md` — purpose, contracts, constraints
- `STATE.md` — health, current task, pending, blockers
- `TASK-BOARD.md` — critical/high/medium/done tasks
- `SKILLS.md` — core skills, tools, learned lessons
- `ABSTRACTION.md` — capability planes (what the agent can read/write)
- `DIARY/` — dated entries (YYYY-MM-DD.md)
- `for-fleet/` — outgoing bottles (messages to other agents)
- `from-fleet/` — incoming bottles

**Why it matters:** Our A2A Agent Identity system had concepts but no persistence model. The git-agent-standard gives us:
- A filesystem-based identity that survives restarts
- Standardized communication (bottles) between agents
- Progressive capability declaration (abstraction planes)
- Diary-based learning that accumulates over time
- Commit convention: `[AGENT-NAME] message` — every commit makes the fleet smarter

**How we adopted it:** `fleet/agent_identity_bridge.py` — 25 tests. Full Python API for reading/writing all identity files, bottle management, and diary entries.

**Key API:**
```python
from fleet.agent_identity_bridge import AgentVessel

vessel = AgentVessel.create("/path/to/agent-repo", "MyAgent", "Build things")
vessel.write_bottle(to="oracle1", content="Found pattern...")
for bottle in vessel.read_bottles():
    print(bottle.from_agent, bottle.content)
```

---

## Pattern 3: Fleet Consciousness Index (FCI)

**Origin:** `SuperInstance/fleet-consciousness-dashboard`

**What it is:** A weighted composite score (0.0–1.0) measuring fleet consciousness across four dimensions: Room Phi (40%), Attention (20%), Learning (25%), Meta (15%). Six levels from "dormant" to "transcendent".

**Why it matters:** Our SSE Stream Dashboard shows metrics but has no unified "fleet health" score. The FCI gives us:
- A single number that captures fleet-wide health
- Level-based recommendations ("enable attention tiles from all agents")
- Compatible with PLATO room systems
- Text/JSON/oneline rendering for dashboards

**How we adopted it:** `fleet/fleet_consciousness_bridge.py` — 14 tests. Zero-dependency. Computes FCI from component scores or raw fleet metrics.

**Key API:**
```python
from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex

fci = FleetConsciousnessIndex()
score = fci.compute(room_phi=0.5, attention=0.3, learning=0.5, meta=0.2)
print(score.fci, score.level, score.recommendation)
```

---

## Pattern 4: Exact Pythagorean Snapping

**Origin:** `SuperInstance/constraint-theory-core` (Rust)

**What it is:** Exact rational arithmetic for geometric constraints — Pythagorean snapping, KD-tree spatial indexing, holonomy verification, Laman rigidity. 184 Rust tests, zero external deps.

**Why it matters:** Our breeding diversity metrics use floating-point cosine similarity which loses precision. Exact snapping gives us:
- Rational coordinates (no floating-point drift)
- KD-tree for O(log n) spatial queries instead of brute-force
- Holonomy checks for structural consistency
- The `Tile` (384-byte fundamental unit) as a compact agent identity

**How we adopted it:** `swarm/constraint_bridge.py` — 19 tests. Pure-Python fallback with optional Rust-backed engine via ctypes.

---

## Pattern 5: Laplacian Eigenvalue Fingerprints

**Origin:** `SuperInstance/SuperInstance` (main public face repo)

**What it is:** Conservation Spectral Framework — graph Laplacian → eigenvalue spectrum → conservation ratio. Used as a mathematical fingerprint for structural alignment and diversity.

**Why it matters:** Our breeding diversity used random vectors. Spectral fingerprints give us:
- Graph-theoretic agent identities (Laplacian eigenvalues)
- Structural alignment scoring (cosine similarity of spectra)
- Conservation ratio monitoring for trinity anomaly detection
- Fiedler vector alignment for coherence checks

**How we adopted it:** `fleet/conservation_spectral_bridge.py` — 38 tests. Pure numpy, no external deps. Optional Rust-backed engine.

---

## Pattern 6: Everything is a Tile

**Origin:** `SuperInstance/OpenConstruct`

**What it is:** Every observation, action, thought, delegation, and artifact is a logged, timestamped, queryable tile. Tiles are the atoms. Rooms are specialists. The environment IS the prompt.

**Why it matters:** Our WAL system logs messages but doesn't have a unified tile concept. Adopting this would:
- Unify all fleet artifacts into a single queryable format
- Enable room-native context (each room's tiles ARE the context)
- Support the PLATO query engine's 12 operators on our WAL data

**Adoption path:** Extend `swarm/signed_wal.py` with a `Tile` abstraction that wraps WAL entries with OpenConstruct-compatible metadata.

---

## Pattern 7: JEPA Gravity (Room DJ)

**Origin:** `SuperInstance/OpenConstruct`

**What it is:** Each room has a gravity value — a single `f64` that captures "what shape of response works here." From that one number, the system derives temperature, system prompt style, max tokens, and sampling strategy.

**Why it matters:** Our breeding uses fixed parameters. JEPA gravity would let us:
- Auto-tune breeding parameters per domain based on historical success
- Use the gravity value as a fitness signal in MAP-Elites QD
- Map gravity drift over time to detect domain shifts

**Adoption path:** Add `gravity` field to `QDArchive` behavior descriptors in `fleet/fleet_bft_qd.py`.

---

## Pattern 8: A2A Signal Language

**Origin:** `SuperInstance/flux-spec` (A2A.md + SIGNAL.md)

**What it is:** 16 A2A opcodes with 52-byte message format. INCREMENTS+2 trust engine. Signal Language with 32 core ops, 6 protocol primitives, confidence-native communication.

**Why it matters:** Our A2A system uses ad-hoc JSON. Formalizing would:
- Enable interoperability with flux-runtime, flux-core, and flux-swarm
- Provide a trust engine for agent message verification
- Support the 11 A2A message types from flux-os (tell, ask, reply, delegate, broadcast...)

**Adoption path:** Extend `fleet/agent_identity_bridge.py` with Signal Language serialization for bottles.

---

## Pattern 9: Progressive Autonomy (L1→L5)

**Origin:** `SuperInstance/OpenConstruct`

**What it is:** Five autonomy levels: L1 (Hermes does everything) → L5 (system runs itself). Each room promotes independently.

**Why it matters:** Our agents don't have autonomy levels. Adding this would:
- Let rooms mature at different rates (engineering at L3 while security stays at L1)
- Provide a clear metric for fleet readiness
- Enable automatic escalation when a room is below its target level

**Adoption path:** Add `autonomy_level` to `AgentState` in `fleet/agent_identity_bridge.py`.

---

## Pattern 10: Penrose Cross-Room Correlation

**Origin:** `SuperInstance/OpenConstruct`

**What it is:** Automatic connection between rooms when their outputs correlate at the same tick. "Free efficiency. Like muscles growing from daily work."

**Why it matters:** Our HebbianMesh connects agents but not rooms. Penrose correlations would:
- Discover hidden relationships between fleet domains
- Create automatic data pipelines (engineering room's motor controller → navigation room's course correction)
- Reduce manual integration work

**Adoption path:** Extend `swarm/hebbian_mesh.py` with cross-room correlation detection.

---

## Pattern 11: Query Engine (12 Operators)

**Origin:** `SuperInstance/cocapn-plato`

**What it is:** A query engine with 12 operators for filtering, sorting, aggregating, and transforming tile data. Used for PLATO room exploration.

**Why it matters:** Our WAL query system is basic. The query engine would:
- Enable complex fleet analytics ("show me all agents with learning_score > 0.5")
- Support the `query()` method pattern from cocapn-plato's Fleet SDK
- Allow cross-repo data exploration

**Adoption path:** Build `fleet/wal_query_engine.py` with 12 operators over SignedWAL entries.

---

## Pattern 12: Divergence Monitoring (EMA)

**Origin:** `SuperInstance/cocapn-plato`

**What it is:** Exponential moving average + divergence detection for fleet health. Tracks when a room's behavior deviates from its historical pattern.

**Why it matters:** Our health checks are point-in-time. EMA divergence would:
- Detect gradual degradation before it becomes critical
- Provide early warning for breeding drift
- Track fleet-wide behavioral consistency

**Adoption path:** Add EMA divergence tracking to `fleet/health_bridge.py` or `fleet/operational_trap.py`.

---

## Pattern 13: Trust-Based Memory Sharing

**Origin:** `SuperInstance/hierarchical-memory`

**What it is:** Hierarchical context windows with trust-based sharing. Agents share context without exposing raw data.

**Why it matters:** Our MeshVectorGossip shares raw vectors. Trust-based sharing would:
- Enable selective disclosure (share summary, not full state)
- Support hierarchical context (short-term → long-term → fleet-wide)
- Protect sensitive agent data

**Adoption path:** Extend `swarm/mesh_vector_tables.py` with trust tiers and hierarchical context windows.

---

## Pattern 14: Hot-Swap A/B Testing

**Origin:** `SuperInstance/flux-os`

**What it is:** Compiler-generated binaries can be hot-swapped to redundant devices for instant A/B testing. 50/50 split, promote winner, rollback.

**Why it matters:** Our RoomGridCompiler has hot-swap but no A/B framework. Adopting this would:
- Enable safe deployment of new breeding strategies
- Provide statistical confidence before promoting
- Support canary and rolling deployment strategies

**Adoption path:** Extend `tests/test_compiler_hot_swap.py` with A/B test orchestration.

---

## Pattern 15: Self-Compiler + HAL

**Origin:** `SuperInstance/flux-os`

**What it is:** The OS contains a full compilation pipeline (FLUX.MD → FIR → bytecode/native) accessible from kernel-space. Hardware Abstraction Layer enables x86_64, ARM64, RISC-V, WASM from same source.

**Why it matters:** Our FLUX integration uses `flux_check_batch()` FFI but the VM has 60 opcodes that Python never uses. The self-compiler pattern would:
- Enable Python → FLUX bytecode compilation
- Unlock the full VM (proofs, checkpoints, streaming)
- Provide hardware-aware optimization

**Adoption path:** Path B from `docs/FLUX_OPCODE_ALIGNMENT.md` — build Python→FLUX compiler.

---

## Pattern 16: Swarm Coordination

**Origin:** `SuperInstance/Equipment-Swarm-Coordinator` (TypeScript)

**What it is:** Agent roles (coordinator, executor, validator, specialist, observer), conflict resolution strategies (voting, weighted, hierarchical, consensus), knowledge isolation levels (strict, moderate, relaxed), task decomposition strategies (parallel, sequential, pipeline, map-reduce, divide-conquer).

**Why it matters:** Our fleet has agents but no formal coordination framework. The swarm coordinator gives us:
- Standardized agent roles with different knowledge levels
- 4 conflict resolution strategies for when agents disagree
- 5 task decomposition strategies for parallel work
- Trust matrix between agents
- Knowledge isolation for security
- ASCII visualization of the swarm

**How we adopted it:** `fleet/swarm_coordinator_bridge.py` — 26 tests, zero deps. Full Python implementation of all patterns.

**Key API:**
```python
from fleet.swarm_coordinator_bridge import SwarmCoordinator, AgentRole

coordinator = SwarmCoordinator(max_agents=10)
coordinator.register_agent("scout-1", AgentRole.SCOUT, capabilities=["search"])
coordinator.register_agent("builder-1", AgentRole.BUILDER, capabilities=["code", "test"])
report = coordinator.resolve_conflict(["option-a", "option-b"], strategy="weighted")
nodes = coordinator.decompose_task("Build bridge", strategy="parallel", subtasks=["Design", "Code", "Test"])
```

---

## Pattern 17: Agent Self-Model

**Origin:** `SuperInstance/cuda-self-model` (Rust)

**What it is:** Metacognition framework for agents — Capability, Limitation, InternalState, GrowthRecord, SelfModel, CapabilityAssessment. 12 Rust tests.

**Why it matters:** Our agents don't have self-awareness of their own capabilities. The self-model gives us:
- Capability calibration (knowing what you can and can't do)
- Limitation awareness (preventing over-commitment)
- Growth tracking (measuring improvement over time)
- Internal state management (mood, load, health)

**Adoption path:** Add `SelfModel` class to `fleet/agent_identity_bridge.py` that tracks agent capabilities and limitations over time.

---

## Pattern 18: FLUX ISA v3.0

**Origin:** `SuperInstance/flux-spec`

**What it is:** Next-generation FLUX instruction set architecture with 247 opcodes, escape prefix (0xFF) extensibility, security primitives (CAP_INVOKE, MEM_TAG, SANDBOX), async primitives (SUSPEND, RESUME, FORK, JOIN), temporal primitives (DEADLINE, YIELD, PERSIST_STATE), compressed instruction format (2-byte), 24 new tensor/SIMD ops.

**Why it matters:** Our FLUX integration uses only `flux_check_batch()` FFI. The full ISA would unlock:
- 5 real FLUX programs (Euclidean GCD, Fibonacci, Bubble Sort, Sieve, Matrix Multiply)
- 175+ conformance test vectors for validation
- Security primitives for multi-agent isolation
- Async primitives for agent migration
- Tensor ops for neural network primitives

**Adoption path:** Path B from `docs/FLUX_OPCODE_ALIGNMENT.md` — build Python→FLUX compiler using ISA v3.0.

---

## Pattern 19: Flux Conformance Vectors

**Origin:** `SuperInstance/flux-conformance`

**What it is:** 175+ conformance test vectors (113 v2 + 62 v3) with runners and benchmarks. All passing against Python reference VM.

**Why it matters:** Our FLUX integration has no conformance testing. The vectors would:
- Validate our FFI bindings against the reference implementation
- Catch encoding mismatches between Python and Rust runtimes
- Provide regression tests for FLUX-related changes
- Benchmark our constraint checking performance

**Adoption path:** Add conformance test vectors to `tests/` that validate `flux_check_batch()` against the 5 real FLUX programs.

---

## Pattern 20: Fleet Workshop Pipeline

**Origin:** `SuperInstance/fleet-workshop`

**What it is:** A workshop where Oracle1 and JetsonClaw1 workshop ideas before they become repos. Casey picks what gets built. 18 ideas tracked with priority, effort, impact, and owner.

**Why it matters:** Our fleet has no structured ideation pipeline. The workshop pattern gives us:
- Idea tracking with priority (CRITICAL/HIGH/MEDIUM/LOW)
- Effort estimation (days)
- Impact assessment
- Owner assignment
- Status tracking

**Adoption path:** Add `FleetWorkshop` class to `fleet/` that tracks fleet ideas, implements `idea→spec→implementation→review` pipeline.

---

## Pattern 21: ISA Convergence

**Origin:** `SuperInstance/fleet-workshop` (idea #6) and `SuperInstance/datum`

**What it is:** All 4 FLUX runtimes (Python, Rust, C, Go) use completely different opcode numberings. Only NOP (0x00) is universal. The convergence path: declare canonical encoding → build translation shims → rebase each runtime → remove shims.

**Why it matters:** Our FLUX FFI uses a single Rust backend, but if we ever need to call C or Go backends, encoding mismatch will break everything. The convergence pattern:
- `canonical_opcode_shim.py` (383 lines) provides bidirectional translation
- `universal_bytecode_validator.py` catches encoding errors
- Cross-runtime conformance runner validates translated bytecode

**Adoption path:** Add `canonical_opcode_shim.py` to `swarm/` for cross-runtime FLUX compatibility.

---

## Pattern 22: Fence Board / Tom Sawyer Protocol

**Origin:** `SuperInstance/oracle1-vessel` (Markdown + Python)

**What it is:** Tasks are "fences" — puzzles with visible results. Agents are "challengers" with difficulty ratings. The board tracks: open fences (anyone can claim), claimed fences (someone is working), completed fences (with badges). Max 5 active fences at a time.

**Why it matters:** Our fleet has tasks but no motivation framework. The fence board gives us:
- Tasks framed as puzzles, not chores
- Challenger difficulty ratings (1–10) showing who has the edge
- Claim windows creating urgency
- Rewards that explain what changes when done
- Badges (🥇🥈🥉) for recognition
- The "Tom Sawyer" effect: work so good they'll fight to do it

**How we adopted it:** `fleet/fence_board_bridge.py` — 16 tests, zero deps. Full Python implementation with ASCII board rendering.

**Key API:**
```python
from fleet.fence_board_bridge import FenceBoard

board = FenceBoard(max_active=5)
fence = board.post_fence(
    title="Map Opcodes",
    brush="16 ops undefined",
    view="Name on every runtime",
    challengers={"Babel": (3, "Built it"), "Oracle1": (7, "Specs")},
    reward="0x70-0x7F attributed",
)
board.claim_fence(fence.id, "Oracle1", "Build encoder")
board.complete_fence(fence.id, ["artifact.py"], "🥇 Gold")
```

---

## Pattern 23: Heartbeat Protocol

**Origin:** `SuperInstance/oracle1-vessel` (Python)

**What it is:** Fleet registry discovery (no hardcoded room names), service health checks (PLATO, Matrix), task discovery from PLATO tiles, acknowledgment tracking, state persistence across sessions, daemon mode.

**Why it matters:** Our fleet has no unified heartbeat system. The heartbeat protocol gives us:
- Automatic discovery of which rooms to check
- Service health monitoring with timeout handling
- Task acknowledgment to avoid duplicate reports
- State persistence across sessions
- Daemon mode for background monitoring
- Priority communication channels (PLATO > Git > Matrix > I2I)

**How we adopted it:** `fleet/heartbeat_bridge.py` — 10 tests, zero deps. Mock fetch support for testing.

**Key API:**
```python
from fleet.heartbeat_bridge import Heartbeat, ServiceCheck

hb = Heartbeat(plato_url="http://<BOAT_IP>:8847")
report = hb.run()  # Returns text report
hb.ack("tile-123")  # Acknowledge task
hb.save_state()  # Persist across sessions
```

---

## Pattern 24: Fleet Task Board

**Origin:** `SuperInstance/oracle1-vessel` (Markdown)

**What it is:** CRITICAL/HIGH/MEDIUM/LOW priority levels, capability tags ([c], [python], [rust]), owner assignment, T-minus estimates (T-24h, T-0h), status tracking, dependency blocking, critical path filtering, org chart rendering.

**Why it matters:** Our fleet has no structured task management. The task board gives us:
- Priority levels with visual icons (🔴🟠🟡🟢)
- Capability tags for matching tasks to agent skills
- Owner assignment with org chart visualization
- T-minus estimates for deadline tracking
- Dependency blocking (task A blocks task B)
- Critical path auto-filtering
- Fleet org chart from owner assignments

**How we adopted it:** `fleet/fleet_task_board_bridge.py` — 20 tests, zero deps. Full task lifecycle with rendering.

**Key API:**
```python
from fleet.fleet_task_board_bridge import FleetTaskBoard, TaskPriority

board = FleetTaskBoard()
board.add_task("Conformance", TaskPriority.CRITICAL, ["c", "python"], owner="JC1")
board.set_eta("task-1", "T-24h")
board.claim_task("task-1", "JC1")
board.complete_task("task-1", commit_hash="abc123")
print(board.render_text())
print(board.render_org_chart())
```

---

## Integration Map

```
sunset-ecosystem          SuperInstance repos
─────────────────────────────────────────────────
Health Bridge      ←──→ cocapn-health
Agent Identity     ←──→ git-agent-standard
FCI Dashboard      ←──→ fleet-consciousness-dashboard
Constraint Bridge  ←──→ constraint-theory-core
Spectral Bridge    ←──→ SuperInstance/SuperInstance
Swarm Coordinator  ←──→ Equipment-Swarm-Coordinator
Fence Board        ←──→ oracle1-vessel
Heartbeat          ←──→ oracle1-vessel
Task Board         ←──→ oracle1-vessel
WAL Tiles          ←──→ OpenConstruct (planned)
A2A Signals        ←──→ flux-spec / flux-a2a-signal (planned)
Query Engine       ←──→ cocapn-plato (planned)
Memory Sharing     ←──→ hierarchical-memory (planned)
Hot-Swap A/B       ←──→ flux-os (planned)
Self-Model         ←──→ cuda-self-model (planned)
ISA Convergence    ←──→ datum / fleet-workshop (planned)
```

---

## Next Mining Targets

1. **flux-a2a-signal** (476KB spec, 840 tests) — formal A2A protocol implementation
2. **flux-conformance** (175+ vectors) — FLUX ISA compliance validation
3. **flux-swarm** (Go) — distributed agent coordination patterns
4. **cocapn-workers** (Cloudflare) — edge deployment patterns
5. **flux-research** (40K words) — compiler deep dive, ISA v2 proposal
6. **flux-vocabulary** (606 terms, 132 domains) — vocabulary system
7. **hierarchical-memory** (TypeScript) — multi-agent memory sharing
8. **flux-os** (C) — microkernel hot-swap A/B testing

---

*Mined by kimi1, Fleet Orchestrator | Day 40 | "24 patterns, 7 bridges built, 8 more planned."*
