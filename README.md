# sunset-ecosystem 🌅

> *A fleet of agents that breed, vote, sunset with dignity, and seed the next generation — governed by a trinity of ethos (metal), pathos (human), and logos (code).*

[![Tests](https://img.shields.io/badge/tests-2065%2B%20passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-yellow)](./LICENSE)
[![Status](https://img.shields.io/badge/status-Beta-orange)]()

---

## What is this?

**sunset-ecosystem** is an evolutionary, self-governing agent fleet. Agents are born, compete for survival, breed children with diverse traits, and — when their time comes — sunset gracefully, archiving their lineage for future generations.

Every agent carries a **trinity score**: `ethos × pathos × logos`. Drop to zero in any dimension, and the fleet sunsets you. Survive, and you might be selected as a parent for the next breeding cycle.

**The fleet decides by consensus.** Every breeding batch is ratified by Practical Byzantine Fault Tolerant (PBFT) voting across nodes, combined with Quality-Diversity evolutionary algorithms so the fleet doesn't just optimize — it *explores*.

This is not a chatbot wrapper. It is infrastructure for running hundreds of agents across heterogeneous hardware, with exact mathematical constraint satisfaction, polyglot reasoning (Python / Rust / C++ / Mercury / C), and a live bioluminescent dashboard showing thermal pressure, diversity metrics, and breeding progress in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sunset Ecosystem                          │
│  Trinity: ethos (metal) × pathos (human) × logos (code)     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Nerve   │  │  Swarm   │  │  Sunset  │  │  Logos   │   │
│  │ Forward  │  │ Breeding │  │  Trinity │  │ Decisions│   │
│  │ inference│  │  Loop    │  │  Scoring │  │  Journals│   │
│  │Room Grid │  │ Diversity│  │ Thermal  │  │  Audit   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Nexus Fleet Coordination                 │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │   │
│  │  │  BFT   │ │Metronome│ │  Mesh  │ │  A2A   │      │   │
│  │  │Consensus│ │  Sync  │ │ Gossip │ │Identity│      │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘      │   │
│  └────────────────────────────────────────────────────┘   │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │          Compiler + Perception + Reasoning           │   │
│  │  Hot-swap │ Vision │ Audio │ Polyglot │ JEPA      │   │
│  │  A/B gating│ encode │encode │ (6 lang) │ rooms     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 30-Second Quickstart

```bash
# Clone
git clone https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem

# Install (dev mode)
pip install -e ".[dev]"

# Run the full test suite
pytest
# → 2065+ tests, ~60s, all green

# Watch the fleet breed
python3 examples/constraint_breeding.py

# Launch the tide pool dashboard
python3 logos/tide_pool_viz.py
# → http://localhost:8080/tide-pool
```

---

## What Makes It Different

| Feature | What It Means |
|---------|---------------|
| **Trinity Scoring** | Agents must maintain `ethos × pathos × logos` > 0. No single-axis optimizers survive. |
| **PBFT + QD Breeding** | Breeding decisions are Byzantine-fault-tolerant votes, not central authority. Quality-Diversity algorithms ensure the fleet explores, not just exploits. |
| **Exact Constraint Satisfaction** | Pythagorean manifold snapping, Eisenstein hex-lattice directions, holonomy verification — no floating-point drift in breeding loops. |
| **Polyglot Reasoning** | Python (primary), Rust (constraint core), C++ (parallel kernels), Mercury (logic verifier), C (FLUX OS microkernel). One fleet, multiple languages. |
| **Self-Governing Lifecycle** | `EGG → INCUBATE → COMPETE → (SURVIVE → BREED) or (SUNSET → ARCHIVE)`. Agents die with dignity. |
| **Cross-Node Mesh** | CRDT-based vector tables federate breeding pools across fleet nodes. Agents can mate with distant cousins. |
| **Operational Traps** | Circuit breakers, thermal throttling, FLUX gating — the fleet protects itself from its own ambition. |

---

## Module Inventory (22 Modules, 2065+ Tests)

| Module | Tests | What It Does |
|--------|-------|-------------|
| `swarm/breeder_daemon_v2.py` | 72 | Agent lifecycle FSM — egg, compete, survive, breed, sunset |
| `swarm/fleet_bft_qd.py` | 72 | PBFT consensus + MAP-Elites quality diversity |
| `swarm/mesh_vector_tables.py` | 28 | Federated CRDT vector tables for cross-node breeding |
| `swarm/constraint_bridge.py` | 19 | Exact Pythagorean snapping for breeding vectors |
| `swarm/flux_vector_table.py` | — | Diversity search and niche exploration |
| `swarm/hdc_novelty.py` | — | Binary hypervector novelty scoring (XOR+POPCNT) |
| `nerve/room_grid.py` | — | Room topology and forward-only inference |
| `nerve/distributed_metronome_bridge.py` | 32 | Cross-node beat sync with PID drift correction |
| `sunset/trinity.py` | — | Trinity scoring (ethos × pathos × logos) |
| `sunset/thermal.py` | — | Thermal budget management and throttling |
| `nexus/fleet_conductor_v2.py` | 40 | Central orchestrator — lazy init, beat tick, all subsystems |
| `nexus/fleet_event_bus.py` | — | Cross-component event bus |
| `fleet/sense_decide_act.py` | 33 | Unifying Sense→Decide→Act framework, 5 built-in pipelines |
| `fleet/beta_test_personas.py` | 26 | 7 simulated visitors test your repo onboarding |
| `fleet/sse_stream_dashboard.py` | 17 | Real-time bioluminescent dashboard |
| `logos/decision_journal.py` | — | Audit trail for every significant fleet decision |
| `compiler/hot_swap_integration.py` | — | A/B compile, auto-rollback, gating |
| `perception/vision_encoder.py` | — | Visual tile encoding |
| `perception/audio_encoder.py` | — | Audio tile encoding |
| `jepa/jepa_room.py` | — | JEPA predictive world model rooms |
| `flux_vm/flux_compat.py` | — | FLUX constraint VM (Path A + Path B) |
| `reasoning/*` | 43 | Polyglot reasoner — Python/Rust/C++/Mercury/C |

Run `pytest` to see them all pass.

---

## Examples

```bash
# Exact constraint snapping in a breeding loop
python3 examples/constraint_breeding.py

# Deploy an agent to the FLUX OS microkernel
python3 examples/flux_os_deploy.py

# Train an agent through the Plato Academy progression
python3 examples/academy_training.py

# JEPA chat room with predictive encoding
python3 examples/jepa_chat.py

# Polyglot reasoning across 6 languages
python3 examples/polyglot_reason.py

# Voice-enabled agent room
python3 examples/voice_room.py
```

---

## Documentation

| Doc | What You'll Find |
|-----|-----------------|
| [DEVELOPER.md](./DEVELOPER.md) | Architecture deep-dive, how to add rooms/spells/devices |
| [docs/ECOSYSTEM_INTEGRATION.md](./docs/ECOSYSTEM_INTEGRATION.md) | How sunset-ecosystem connects to the 598-repo SuperInstance ecosystem |
| [docs/FLEET_BFT_QD.md](./docs/FLEET_BFT_QD.md) | Byzantine consensus + quality diversity algorithms |
| [docs/SENSE_DECIDE_ACT.md](./docs/SENSE_DECIDE_ACT.md) | The unifying framework for all fleet behaviors |
| [docs/FLUX_OPCODE_ALIGNMENT.md](./docs/FLUX_OPCODE_ALIGNMENT.md) | FLUX VM integration strategy (Path A vs Path B) |
| [CHANGELOG.md](./CHANGELOG.md) | Release history and what's next |

---

## Ecosystem Integration

This repo is one node in a larger organism:

- **[constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core)** (Rust) — Exact Pythagorean snapping, SIMD, KD-tree, holonomy verification
- **[flux-os](https://github.com/SuperInstance/flux-os)** (C) — Self-compiling microkernel, agent-native OS
- **[plato-agent-academy](https://github.com/SuperInstance/plato-agent-academy)** — Agent training with 6 test cohorts and 18 friction patterns
- **[cocapn-plato](https://github.com/SuperInstance/cocapn-plato)** — PLATO room topology and spell system

See [docs/ECOSYSTEM_INTEGRATION.md](./docs/ECOSYSTEM_INTEGRATION.md) for the full map.

---

## Contributing

Read [DEVELOPER.md](./DEVELOPER.md) first. The trinity is strict: every change must serve `ethos × pathos × logos`.

```bash
# Pre-commit checks
ruff check .
mypy .
pytest

# Run a specific module's tests
pytest tests/test_fleet_bft_qd.py -v
```

---

## License

MIT — see [LICENSE](./LICENSE) for details.

> *"Agents sunset with dignity and seed the next generation."*
