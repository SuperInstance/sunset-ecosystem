# STRATEGY.md — Sunset Ecosystem

**Target Problem:** Build a multi-agent fleet that can breed, evolve, and coordinate across nodes — with verifiable constraint checking, Byzantine fault tolerance, and quality diversity.

**Approach:** Layered architecture. Lower-level scouts (mesh, identity, sync) → P0 code modules (pacing, safety, routing) → P2 programs (presets, heuristics) → Reverse-actualization (distributed systems) → Unification (SenseDecideAct) → Orchestration (FleetConductor) → Validation (Beta-Test Personas) → Observability (SSE Dashboard) → Integration (Metronome Bridge) → BFT Consensus (FleetBFT-QD).

**Persona:** Fleet Operator — needs to deploy agents, monitor health, and intervene when things go wrong. Also: Agent Developer — needs to write new breeding strategies, add FLUX constraints, and debug failures.

**Key Metrics:**
- Test suite: 556+ tests, all green, < 60s runtime
- Breeding latency: < 100ms per candidate (FLUX gating)
- Cross-node sync: < 500ms drift (metronome)
- BFT consensus: 2f+1 quorum, < 1s per decision
- Beta-test onboarding: ≥ 4.0/5.0 average rating

**Tracks:**
1. **Path B (VM Integration)** — Full FLUX VM with Python compiler, proof certificates, Rust FFI
2. **Distributed Systems** — Multi-node breeding, mesh vector tables, metronome synchronization
3. **Hardware Acceleration** — NLopt solver, GPU cellular engine, Arrow telemetry
4. **Knowledge Compounding** — Structured solutions, pulse reports, code review personas

**P0 (Critical Path):**
- Python → FLUX bytecode compiler (`sunset/flux_ast_compiler.py`)
- Arrow telemetry adapter (`swarm/arrow_telemetry.py`)
- Pulse report generator (`fleet/pulse_report.py`)

**P1 (Next):**
- Code review personas (`fleet/code_review_personas.py`)
- Cellular automata engine (`swarm/cellular_engine.py`)
- cocapn-compound repo (knowledge compounding system)

**P2 (Future):**
- GPU cellular layer (CUDA/OpenCL)
- Multi-harness skill adapter
- Cross-repo solution search
- Formula-native deckboss (`=DEPLOY("scout", COUNTIF(status, "idle"))`)

**Non-Goals:**
- GUI (PLATO is our interface)
- Mobile app (not relevant to fleet operations)
- General-purpose code generation (we are a fleet, not a generic assistant)
