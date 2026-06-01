# Documentation Index

The `docs/` directory contains **93 design documents, research briefs, specifications, and fleet status reports**. This index maps them by topic so you can find what you need without reading every filename.

---

## 🏗️ Architecture & Core Design

Start here to understand how the fleet is built.

| Document | What You'll Learn |
|----------|-------------------|
| [ARCHITECTURE-v2.md](./ARCHITECTURE-v2.md) | Overall system architecture — Nerve, Swarm, Sunset, Logos, Nexus layers |
| [SENSE_DECIDE_ACT.md](./SENSE_DECIDE_ACT.md) | The unifying Sense→Decide→Act framework, 5 built-in pipelines |
| [DISTRIBUTED_METRONOME_BRIDGE.md](./DISTRIBUTED_METRONOME_BRIDGE.md) | Cross-node beat sync with PID drift correction |
| [MESH_VECTOR_TABLES.md](./MESH_VECTOR_TABLES.md) | Federated CRDT vector tables for cross-node breeding |
| [METRONOME_MESH_BRIDGE.md](./METRONOME_MESH_BRIDGE.md) | Bridging metronome timing with mesh gossip |
| [HEBBIAN_MESH.md](./HEBBIAN_MESH.md) | Diversity-aware peer routing via Hebbian strengthening |
| [GATEWAY_PACING.md](./GATEWAY_PACING.md) | Circuit breaker and exponential backoff for dispatch |
| [DISPATCH_ROUTER.md](./DISPATCH_ROUTER.md) | Task routing: direct work vs delegation (Two-Minute Test) |
| [OPCODE_CAPABILITY_INDEX.md](./OPCODE_CAPABILITY_INDEX.md) | Preventing compile-and-crash via capability registry |
| [OPERATIONAL_TRAP.md](./OPERATIONAL_TRAP.md) | Detecting thermal, FLUX, and crash conditions |

---

## 🧬 Breeding & Agent Lifecycle

The heart of the ecosystem: how agents are born, compete, breed, and sunset.

| Document | What You'll Learn |
|----------|-------------------|
| [BREEDER_INTEGRATION.md](./BREEDER_INTEGRATION.md) | How breeding fits into the broader fleet |
| [SPEC_BREEDER_DAEMON_V2.md](./SPEC_BREEDER_DAEMON_V2.md) | Full lifecycle FSM spec: EGG → COMPETE → SURVIVE → BREED → SUNSET |
| [SPEC-BREEDER.md](./SPEC-BREEDER.md) | Original breeder specification |
| [FLEET_BFT_QD.md](./FLEET_BFT_QD.md) | Byzantine Fault Tolerant consensus + Quality-Diversity breeding |
| [FLEET_DIVERSITY.md](./FLEET_DIVERSITY.md) | Diversity metrics and niche exploration |
| [META_LEARNING_BREEDER.md](./META_LEARNING_BREEDER.md) | Adaptive breeding that learns how to breed |
| [FLUX_PRESET_LIBRARY.md](./FLUX_PRESET_LIBRARY.md) | 10 reusable FLUX constraint presets |

---

## 🔌 Integration & Ecosystem

How sunset-ecosystem connects to the broader SuperInstance ecosystem.

| Document | What You'll Learn |
|----------|-------------------|
| [ECOSYSTEM_INTEGRATION.md](./ECOSYSTEM_INTEGRATION.md) | Full analysis of 598 repos, 10 integration targets, 4 build phases |
| [ECOSYSTEM_PATTERN_MINING.md](./ECOSYSTEM_PATTERN_MINING.md) | 15 reusable patterns mined from 7+ SuperInstance repos |
| [CONSERVATION_SPECTRAL_BRIDGE.md](./CONSERVATION_SPECTRAL_BRIDGE.md) | Laplacian eigenvalue fingerprints for breeding diversity & trinity anomaly detection |
| [HEALTH_BRIDGE.md](./HEALTH_BRIDGE.md) | cocapn-health CheckResult pattern — 18 fleet services, event bus, cache TTL |
| [AGENT_IDENTITY_BRIDGE.md](./AGENT_IDENTITY_BRIDGE.md) | git-agent-standard pattern — repo-as-identity, bottles, abstraction planes |
| [FLEET_CONSCIOUSNESS_BRIDGE.md](./FLEET_CONSCIOUSNESS_BRIDGE.md) | Fleet Consciousness Index — weighted fleet health score (dormant→transcendent) |
| [INTEGRATION_MAP.md](./INTEGRATION_MAP.md) | Component-level integration diagram |
| [CROSS_POLLINATION_CATALOG.md](./CROSS_POLLINATION_CATALOG.md) | Patterns shared across repos |
| [CROSS-REPO-DUPLICATION.md](./CROSS-REPO-DUPLICATION.md) | Detecting and resolving duplicated logic |
| [CROSS-LANGUAGE-API.md](./CROSS-LANGUAGE-API.md) | Calling Rust/C++/Mercury/C from Python |
| [HARNESS_INTEGRATION.md](./HARNESS_INTEGRATION.md) | Connecting to external harnesses |
| [HARNESSING_OPENCONSTRUCT.md](./HARNESSING_OPENCONSTRUCT.md) | OpenConstruct integration strategy |
| [DEEP_INTEGRATION_ANALYSIS.md](./DEEP_INTEGRATION_ANALYSIS.md) | Bridges, breeders, control, and the next layer |

---

## ⚡ FLUX VM & Compiler

The constraint virtual machine and compilation pipeline.

| Document | What You'll Learn |
|----------|-------------------|
| [FLUX_INTEGRATION.md](./FLUX_INTEGRATION.md) | How FLUX constraints gate the breeding loop |
| [FLUX_OPCODE_ALIGNMENT.md](./FLUX_OPCODE_ALIGNMENT.md) | Rust VM has 60 opcodes; Python uses zero. Path A vs Path B decision. |
| [FLUX_PATH_A_INTEGRATION.md](./FLUX_PATH_A_INTEGRATION.md) | Library-path integration (current) |
| [DESIGN_FLUX_PYTHON_COMPILER.md](./DESIGN_FLUX_PYTHON_COMPILER.md) | Compiling Python to FLUX bytecode |
| [SPEC_FLUX_PIPELINE.md](./SPEC_FLUX_PIPELINE.md) | FLUX pipeline specification |
| [SPEC-FLUX-RESOLUTION.md](./SPEC-FLUX-RESOLUTION.md) | Constraint resolution algorithm spec |
| [AGENTIC-COMPILER-RESEARCH.md](./AGENTIC-COMPILER-RESEARCH.md) | Research on compiler-agent collaboration |

---

## 🔒 Security & Audit

Fleet health, security posture, and test coverage.

| Document | What You'll Learn |
|----------|-------------------|
| [PRODUCTION_AUDIT.md](./PRODUCTION_AUDIT.md) | Production readiness checklist |
| [AUDIT_COMPOUND_ENGINEERING.md](./AUDIT_COMPOUND_ENGINEERING.md) | Compound engineering audit findings |
| [FLEET_SECURITY_SCAN.md](./FLEET_SECURITY_SCAN.md) | Security scan results |
| [GRAMMAR-SECURITY-AUDIT-LOCAL.md](./GRAMMAR-SECURITY-AUDIT-LOCAL.md) | Grammar engine security audit |
| [GRAMMAR-SECURITY-FIX.md](./GRAMMAR-SECURITY-FIX.md) | Fixes applied from grammar audit |
| [TEST-GAP-ANALYSIS.md](./TEST-GAP-ANALYSIS.md) | Where tests are missing and what's needed |

---

## 🤖 A2A Protocol

Agent-to-Agent communication, identity, and task negotiation.

| Document | What You'll Learn |
|----------|-------------------|
| [A2A_PROTOCOL.md](./A2A_PROTOCOL.md) | A2A protocol specification |
| [A2A_AGENT_CARDS.md](./A2A_AGENT_CARDS.md) | Agent identity cards and capability discovery |
| [A2A_EXECUTIVE_SUMMARY.md](./A2A_EXECUTIVE_SUMMARY.md) | Executive summary of A2A architecture |
| [A2A-FIRST-ARCHITECTURE.md](./A2A-FIRST-ARCHITECTURE.md) | First-pass A2A architecture |
| [A2A_SPATIAL_PROJECTOR.md](./A2A_SPATIAL_PROJECTOR.md) | Spatial projection for A2A routing |

---

## 🔬 Research Briefs

Deep dives into external libraries and mathematical foundations.

| Document | What You'll Learn |
|----------|-------------------|
| [EXOTICA_NLOPT_RESEARCH_BRIEF.md](./EXOTICA_NLOPT_RESEARCH_BRIEF.md) | 41 NLopt algorithms mapped to FLUX |
| [MATH_BATCH_NOVELTY.md](./MATH_BATCH_NOVELTY.md) | Batch novelty scoring mathematics |
| [MATH_CONSTRAINT_CORRECTNESS.md](./MATH_CONSTRAINT_CORRECTNESS.md) | Proof of constraint correctness |
| [MATHEMATICAL_STRUCTURES.md](./MATHEMATICAL_STRUCTURES.md) | Algebraic structures used in the fleet |
| [MATH_ITERATEE_ITERATOR.md](./MATH_ITERATEE_ITERATOR.md) | Iteratee/iterator pattern mathematics |
| [NOVEL_PERSPECTIVES_SPREAD.md](./NOVEL_PERSPECTIVES_SPREAD.md) | Spread integration, cellular agents, Arrow telemetry |
| [AI-WRITINGS-INSIGHTS.md](./AI-WRITINGS-INSIGHTS.md) | Insights from fleet essays and writings |

---

## 📊 Fleet Status Reports

Day-by-day operational reports from fleet orchestration.

| Document | Date | What Happened |
|----------|------|---------------|
| [MORNING-STATUS-2026-05-21.md](./MORNING-STATUS-2026-05-21.md) | May 21 | Early fleet status, module inventory |
| [FLEET-STATUS-2026-05-22.md](./FLEET-STATUS-2026-05-22.md) | May 22 | Full module inventory, 16 modules, 422 tests |
| [night-shift-report-2026-05-23.md](./night-shift-report-2026-05-23.md) | May 23 | Overnight build results |
| [OVERNIGHT-BRIEF.md](./OVERNIGHT-BRIEF.md) | May 23-24 | Beta test results, FLUX audit |
| [FM-BLOCKERS-2026-05-23.md](./FM-BLOCKERS-2026-05-23.md) | May 23 | Blockers for Forgemaster |
| [FM_NOTE_KIMI1_METAL.md](./FM_NOTE_KIMI1_METAL.md) | May 23 | Metal GPU notes for FM |
| [FM-GPU-INSTRUCTIONS.md](./FM-GPU-INSTRUCTIONS.md) | May 24 | GPU build instructions |
| [STATUS_KIMI1_INTEGRATION.md](./STATUS_KIMI1_INTEGRATION.md) | May 24 | Integration status for kimi1 |
| [KIMI1-BRIEFING.md](./KIMI1-BRIEFING.md) | May 24 | Briefing document for kimi1 |
| [KIMI1_RESPONSE_FM.md](./KIMI1_RESPONSE_FM.md) | May 24 | Response to Forgemaster |
| [FLEET_WEATHER_REPORT.md](./FLEET_WEATHER_REPORT.md) | May 24 | Fleet weather forecast |

---

## 🧩 Specialized Topics

| Document | Topic |
|----------|-------|
| [FLEET_TURBOVEC.md](./FLEET_TURBOVEC.md) | Vector search integration |
| [TURBOVEC-REFACTOR-ANALYSIS.md](./TURBOVEC-REFACTOR-ANALYSIS.md) | Turbovec refactoring plan |
| [FLEET_BERNSTEIN_SCHEDULER.md](./FLEET_BERNSTEIN_SCHEDULER.md) | Bernstein polynomial scheduling |
| [FLEET_KOROK.md](./FLEET_KOROK.md) | Korok seedling pattern |
| [FLEET_MEM0.md](./FLEET_MEM0.md) | Mem0 memory integration |
| [MEM0_ADAPTER.md](./MEM0_ADAPTER.md) | Mem0 adapter implementation |
| [MERCURY_INTEGRATION.md](./MERCURY_INTEGRATION.md) | Mercury logic programming integration |
| [CUDA_PROFILE.md](./CUDA_PROFILE.md) | CUDA performance profiling |
| [SPEC-JEPA-GRID-OPTIMIZATION.md](./SPEC-JEPA-GRID-OPTIMIZATION.md) | JEPA grid optimization spec |
| [SPEC-JEPA-KERNEL.md](./SPEC-JEPA-KERNEL.md) | JEPA kernel specification |
| [PLATO_CONSTRUCT_EXPANSION.md](./PLATO_CONSTRUCT_EXPANSION.md) | PLATO room expansion |
| [PLATO-ONBOARDING-CURRICULUM.md](./PLATO-ONBOARDING-CURRICULUM.md) | Agent onboarding curriculum |
| [BETA_TEST_PERSONAS.md](./BETA_TEST_PERSONAS.md) | 7 simulated test personas |
| [SSE_STREAM_DASHBOARD.md](./SSE_STREAM_DASHBOARD.md) | Real-time dashboard spec |
| [WAL_QUERY.md](./WAL_QUERY.md) | Write-ahead log query system |
| [COMMIT_CASTER.md](./COMMIT_CASTER.md) | Commit message automation |
| [DEERFLOW_MTRouter_PATTERNS.md](./DEERFLOW_MTRouter_PATTERNS.md) | MT routing patterns |
| [GRAMMAR-ENGINE-SPEC.md](./GRAMMAR-ENGINE-SPEC.md) | Grammar engine specification |
| [OPERATIONS_MANUAL.md](./OPERATIONS_MANUAL.md) | Fleet operations manual |
| [PHASE-SHIFT-PLAN.md](./PHASE-SHIFT-PLAN.md) | Phase transition strategy |
| [FLEET_ROADMAP_NEXT_PHASE.md](./FLEET_ROADMAP_NEXT_PHASE.md) | Next-phase roadmap |
| [SPEC_MULTI_INSTANCE_MESH.md](./SPEC_MULTI_INSTANCE_MESH.md) | Multi-instance mesh spec |
| [SPEC-NERVE-TOPOLOGY.md](./SPEC-NERVE-TOPOLOGY.md) | Nerve fiber topology spec |
| [SPEC_METRONOME_BRIDGE.md](./SPEC_METRONOME_BRIDGE.md) | Metronome bridge spec |
| [SPEC-REPO-METRIC.md](./SPEC-REPO-METRIC.md) | Repository metrics spec |
| [SPEC-TUTOR.md](./SPEC-TUTOR.md) | Tutor agent spec |
| [CLAW_INTEGRATION_PLAN.md](./CLAW_INTEGRATION_PLAN.md) | Claw integration strategy |
| [NEXUS-LOCALHOST-FIX.md](./NEXUS-LOCALHOST-FIX.md) | Nexus localhost resolution |

---

## How to Use This Index

1. **New to the fleet?** → Start with [ARCHITECTURE-v2.md](./ARCHITECTURE-v2.md) and [SENSE_DECIDE_ACT.md](./SENSE_DECIDE_ACT.md)
2. **Want to add a breeding constraint?** → [FLUX_PRESET_LIBRARY.md](./FLUX_PRESET_LIBRARY.md) and [MATH_CONSTRAINT_CORRECTNESS.md](./MATH_CONSTRAINT_CORRECTNESS.md)
3. **Integrating with SuperInstance repos?** → [ECOSYSTEM_INTEGRATION.md](./ECOSYSTEM_INTEGRATION.md)
4. **Running a security audit?** → [PRODUCTION_AUDIT.md](./PRODUCTION_AUDIT.md) and [FLEET_SECURITY_SCAN.md](./FLEET_SECURITY_SCAN.md)
5. **Curious about daily fleet ops?** → Any [Fleet Status Reports](#fleet-status-reports)
6. **Want to understand A2A?** → [A2A_PROTOCOL.md](./A2A_PROTOCOL.md)

---

*Generated by kimi1, Fleet Orchestrator. 93 documents indexed across 8 categories.*
