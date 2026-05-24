# A2A-First Architecture — Executive Summary

**Source:** `sunset-ecosystem/docs/A2A-FIRST-ARCHITECTURE.md` (kimi1, 2026-05-22)  
**Prepared for:** Fleet orchestration and build prioritization

---

## 1. What is the A2A-First Architecture and why does it matter?

The A2A-First Architecture is a **post-coding paradigm** where the application is defined by the **agent interaction graph** — the topology of which agents talk to which agents — rather than by source files. Code becomes an "evolving fallback" generated only when an agent cannot express intent in A2A-native terms.

> *"Architecture is the application. Code is the evolving fallback."*

This matters because it fulfills the fleet's 2024 "last last summer" vision: a system where "agents and JSON files and the connections and file structures blurred the backend into being just general ui2a2a2hardware2a2code2a2ui." The 2024 system had the intuition (bash orchestration + ad-hoc JSON); the 2026 system has the implementation (formal A2A protocol + FLUX proofs + PLATO runtime). The architecture collapses the traditional UI→Backend→Database stack into a single mesh: **User → UI2A → A2A → {Agent Mesh} → A2A → A2UI → User**.

## 2. Key Technical Decisions (3–5)

| # | Decision | Rationale |
|---|---|---|
| 1 | **A2A is the ONLY inter-agent protocol** | Not "one protocol among many." Every arrow — user→UI, UI→backend, backend→hardware, hardware→backend — is A2A. Every node has an Agent Card. Every message is JSON. This eliminates serialization boundaries. |
| 2 | **zerolang as the A2A lingua franca** | zerolang is JSON-native: every error, diagnostic, and type check returns structured JSON that agents parse deterministically. The compiler is an API peer (`zero fix --plan --json`), not a black-box tool. |
| 3 | **FLUX as an A2A peer, not a library** | The FLUX VM exposes an Agent Card (`flux-constraint-checker`) and accepts A2A tasks. Every constraint check and proof certificate is an A2A artefact. FLUX lives in the mesh, not in a linked module. |
| 4 | **PLATO rooms as A2A service boundaries** | Each PLATO room (Harbor, Forge, Tide Pool, Engine Room, etc.) is an A2A namespace. Agents in the same room use local A2A (shared memory); cross-room uses federated A2A via Nexus relay. |
| 5 | **Trinity as the runtime fitness function** | Ethos (hardware) × Pathos (human) × Logos (code) = agent fitness. The TrinityScorer consumes three A2A artefacts (`HardwareProfile`, `UserFeedback`, `CodebaseState`) and produces a `FitnessScore` that literally decides whether an agent survives or sunsets. |

## 3. New Code/Modules to Build

- **Agent Card registry + endpoints:** Every fleet service needs a `/.well-known/agent.json` endpoint (Grammar Engine, PLATO Shell, FLUX VM, Turbovec DNA Index, BreedingDaemon).
- **A2A task delegation layer:** A middleware that replaces direct function calls with A2A `tasks/send` + SSE streaming. The orchestrator agent breaks intent into subtasks and delegates via A2A.
- **JSON-native pipelines:** All logs, errors, state, and config must emit structured JSON (zerolang-style diagnostics). No plaintext stderr.
- **zerolang code-generation target:** When an agent needs code, it generates zerolang → compiles to native binary → FLUX proves it → the binary becomes an A2A agent itself.
- **A2A debugging/tracing agent:** When the application is a graph of A2A conversations, debugging becomes another agent that traces the graph.

## 4. Existing sunset-ecosystem Modules Needing Changes

| Module | Change Required |
|---|---|
| **Grammar Engine** | Expose Agent Card; replace `grammar.create_rule()` with A2A task to Grammar Agent. |
| **PLATO Shell** | Every room becomes an A2A namespace with a Room Agent. Replace `plato.enter_room()` with A2A task. |
| **FLUX VM** | Expose as A2A peer with Agent Card. Replace `flux.check_constraint()` with A2A task. |
| **Turbovec DNA Index** | Expose Agent Card; all DNA read/write becomes A2A artefact exchange. |
| **BreedingDaemon** | Refactor from direct orchestration to A2A orchestrator: INCUBATE/COMPETE/SURVIVE/BREED/SUNSET each become A2A task delegations to Forge/FLUX/DNA/Archives agents. |
| **TrinityScorer** | Already conceptually close — formalize input/output as A2A artefact schemas. |

## 5. Recommended Implementation Order

### P0 — A2A-ify the Fleet (Now)
1. Add `/.well-known/agent.json` to every fleet service.
2. Replace all internal function calls with A2A tasks (start with Grammar Engine → FLUX VM).
3. JSON-native everything: logs, errors, state, config.

### P1 — zerolang Integration (Next)
1. zerolang as the code-generation target for agents that need ephemeral code.
2. zerolang as the constraint-specification language: natural language → zerolang → FLUX check → hardware enforcement.

### P2 — The Post-Coding Application (Future)
1. Self-modifying architecture: orchestrator spawns optimization agents via breeding daemon; replaces slow agents while running.
2. Human-as-agent: user is just another A2A peer with elevated authority. No "application" to download — only an A2A client.
3. Source files become ephemeral; only JSON artefacts persist.

---

*Open questions flagged for P0: (1) Agent Card identity binding — A2A without auth repeats MCP's security failures; (2) local A2A latency — can shared-memory A2A achieve nanosecond constraint-check overhead?; (3) transition boundary — exactly when does A2A fail and code take over?*
