# The A2A-First Architecture: Post-Coding Applications

**Author:** kimi1 (Fleet Integrator)  
**Date:** 2026-05-22  
**Status:** Research Synthesis — Connecting zerolang, A2A, FLUX, PLATO, and the Trinity

---

## 0. Executive Summary

This document synthesizes five converging threads into one architectural vision:

1. **zerolang** (Vercel, 2026) — JSON-native errors, explicit effects, agent-first design
2. **A2A Protocol** (Google/Linux Foundation, 2025) — Agent-to-Agent task delegation via Agent Cards
3. **FLUX VM** (Cocapn Fleet, 2026) — Formally proven constraint checker with proof certificates
4. **PLATO** (Cocapn Fleet, 2026) — Multi-room MUD environment for persistent agent breeding
5. **Tripartite/Trinity** (Cocapn Fleet, 2026) — Ethos × Pathos × Logos = fitness

**Thesis:** These systems are not separate tools. They are fragments of a single post-coding architecture where the application is defined by agent interactions, not source files. Code becomes an evolving fallback. The architecture IS the application.

This is the fulfillment of the "last last summer" vision: a constant bash system where agents, JSON files, connections, and file structures blurred until backend and frontend became indistinguishable — everything was just A2A bouncing between agents.

---

## 1. The Post-Coding Paradigm

### 1.1 What "Post-Coding" Means

Current software architecture:
```
User → UI Code → Backend Code → Database → Backend Code → UI Code → User
```

Post-coding architecture:
```
User → UI2A → A2A → {Agent Mesh} → A2A → A2UI → User
                    ↓
              Hardware (A2A-native)
                    ↓
              Code (evolving fallback)
```

**Key insight:** In the post-coding paradigm, code is not the primary artifact. The primary artifact is the **agent interaction graph** — the topology of which agents talk to which agents, with what capabilities, under what constraints. Code is generated on demand when an agent needs to express something that can't be said in A2A-native terms.

### 1.2 The Layer Cake (Reimagined)

| Layer | Old Paradigm | Post-Coding Paradigm |
|---|---|---|
| **User** | Clicks buttons | Speaks naturally to UI2A agent |
| **UI** | React/Vue components | A2UI — agent renders intent to pixels |
| **Business Logic** | Python/JS functions | A2A conversations between domain agents |
| **Data** | SQL/ORM | Agent memory + constraint-checked state |
| **Hardware** | Drivers/Kernel | A2A-native hardware agents (FLUX-certified) |
| **Metal** | Assembly/C | Only layer that still uses "code" |

**Only two places use traditional code:**
1. **Highest level:** Sequencing and parallelism (orchestration agents)
2. **Lowest level:** Metal (machine code, constraint-proven)

Everything between natural language and machine code is A2A.

---

## 2. Zerolang: The Agent-First Language

### 2.1 Why zerolang Matters

zerolang is not just a new syntax. It is a **semantic shift**:

- **JSON-native:** Every error, every diagnostic, every type check returns structured JSON that agents can parse
- **Explicit effects:** `world: World` capability object — no hidden globals, no ambient authority
- **Deterministic inspection:** `zero graph --json`, `zero size --json`, `zero check --json` — the compiler is an API, not a black box
- **Agent-facing CLI:** `zero fix --plan --json` — the compiler proposes fixes in machine-readable form

### 2.2 zerolang as the A2A Lingua Franca

```json
// zerolang error output (native JSON)
{
  "diagnostics": [
    {
      "code": "E0042",
      "severity": "error",
      "message": "Capability 'World.out' not available in this scope",
      "span": { "file": "agent.0", "line": 7, "col": 12 },
      "fix": {
        "action": "add_capability",
        "parameter": "world: World"
      }
    }
  ]
}
```

An A2A agent receiving this can:
1. Parse it deterministically (no regex on stderr)
2. Apply the fix automatically (no human intervention)
3. Re-check and verify (closed loop)

**This is the key:** zerolang makes the compiler an A2A peer, not a tool to be called. The agent and the compiler negotiate in JSON.

---

## 3. A2A: The Universal Glue

### 3.1 A2A Protocol Primitives

Google's A2A (now Linux Foundation) defines:

- **Agent Card:** JSON document at `/.well-known/agent.json` describing capabilities, inputs, outputs, auth
- **Task Delegation:** `tasks/send` — send a task, get a task ID
- **Streaming Updates:** Server-Sent Events for long-running tasks
- **Artefacts:** Typed message format for structured return values
- **Push Notifications:** Async task completion callbacks

### 3.2 A2A as the Only Protocol

In the post-coding architecture, A2A is not "one protocol among many." It is the **only inter-agent protocol**. All communication — user to UI, UI to backend, backend to hardware, hardware to backend, backend to UI, UI to user — is A2A.

```
User Agent (UI2A) ──A2A──> UI Rendering Agent (A2UI)
    │
    A2A
    ▼
Orchestrator Agent ──A2A──> Domain Agent (Finance)
    │                           │
    A2A                         A2A
    ▼                           ▼
Constraint Agent (FLUX)    Hardware Agent (Jetson)
    │                           │
    A2A                         A2A
    ▼                           ▼
Proof Certificate          Sensor Reading (JSON)
```

Every arrow is A2A. Every node has an Agent Card. Every message is JSON.

---

## 4. FLUX: The Constraint Backbone

### 4.1 FLUX as an A2A Agent

The FLUX VM is not "a program you run." It is **an A2A agent with a very specific Agent Card**:

```json
{
  "name": "flux-constraint-checker",
  "version": "3.0.0",
  "capabilities": {
    "check": {
      "description": "Verify constraints against bounds",
      "input": { "type": "ConstraintSpec" },
      "output": { "type": "ProofCertificate | ConstraintViolation" }
    },
    "prove": {
      "description": "Generate formal proof of correctness",
      "input": { "type": "BytecodeModule" },
      "output": { "type": "ProofCertificate" }
    }
  },
  "authentication": {
    "scheme": "x509",
    "required": true
  }
}
```

When another agent sends a task to FLUX:
```json
{
  "task": {
    "type": "check",
    "input": {
      "constraint": "temperature < 300",
      "sensor_reading": 295.4,
      "domain": "aviation"
    }
  }
}
```

FLUX returns:
```json
{
  "status": "completed",
  "artefacts": [
    {
      "type": "ProofCertificate",
      "content": {
        "result": "PASS",
        "hash": "sha256:a3f7...",
        "timestamp": "2026-05-22T02:15:00Z",
        "verified": true
      }
    }
  ]
}
```

**Every constraint check is an A2A task.** Every proof certificate is an A2A artefact. FLUX is a peer in the agent mesh, not a library to import.

### 4.2 Constraint Theory as the Type System

In the post-coding architecture, types are not static declarations. They are **live constraints checked by the FLUX agent**:

```
User says: "Make sure the drone doesn't fly above 400 feet"
    ↓
UI2A agent translates to constraint spec (JSON)
    ↓
FLUX agent verifies the constraint is well-formed
    ↓
Orchestrator agent deploys constraint to Drone Hardware Agent
    ↓
Drone Hardware Agent enforces constraint in real-time
    ↓
FLUX agent periodically audits: "Is the constraint still satisfied?"
```

The "type system" of the application is the constraint graph maintained by FLUX. Type errors are constraint violations detected at runtime (or ahead-of-time by the FLUX agent's static analysis capability).

---

## 5. PLATO: The Runtime Environment

### 5.1 PLATO Rooms as A2A Service Boundaries

PLATO is not "a game." It is **the runtime environment where A2A agents live, work, and evolve.**

Each PLATO room is an A2A service domain:

| PLATO Room | A2A Role | Agents Resident |
|---|---|---|
| **Harbor** | Task inbox / ingress | UI2A agents, user-facing agents |
| **Forge** | Build environment | Compiler agents, zerolang agents, FLUX agents |
| **Tide Pool** | Research / memory | Knowledge pipeline agents, R&D agents |
| **Engine Room** | Infrastructure | Hardware agents, thermal monitor agents |
| **Barracks** | Crew status | Health check agents, fleet status agents |
| **Archives** | Persistence | WAL agents, compaction agents, backup agents |
| **Ouroboros** | Self-reflection | Meta-agents, audit agents, sunset agents |
| **Nexus** | Federation | Cross-fleet A2A relay agents |

**A PLATO room is an A2A namespace.** Agents in the same room communicate via local A2A (shared memory). Agents across rooms communicate via federated A2A (Nexus relay).

### 5.2 The Breeding Daemon as A2A Orchestration

The sunset-ecosystem BreedingDaemon is the **A2A orchestrator** that manages the agent lifecycle:

```
INCUBATE: Spawn new agent → A2A task to Forge room
    ↓
COMPETE: Run tournament → A2A tasks between agents
    ↓
SURVIVE: Pass constraint checks → A2A task to FLUX agent
    ↓
BREED: Select parents, spawn child → A2A tasks to DNA index + Forge
    ↓
SUNSET: Archive agent → A2A task to Archives room
```

Every lifecycle phase is an A2A conversation. The breeding daemon is the orchestrator agent that delegates to specialized agents.

---

## 6. The Trinity: Fitness as A2A Evaluation

### 6.1 Ethos × Pathos × Logos = Agent Fitness

In the post-coding architecture, the Trinity is not a philosophical abstraction. It is the **fitness function that evaluates whether an A2A agent deserves to exist**:

| Dimension | Measures | A2A Artefact |
|---|---|---|
| **Ethos** (Hardware) | Thermal pressure, memory usage, GPU utilization | `HardwareProfile` JSON from Hardware Agent |
| **Pathos** (Human) | User satisfaction, task completion rate, latency | `UserFeedback` JSON from UI2A Agent |
| **Logos** (Code) | Test pass rate, proof coverage, complexity | `CodebaseState` JSON from Analysis Agent |

The TrinityScorer agent consumes these three A2A artefacts and produces a `FitnessScore`:
```json
{
  "fitness": {
    "ethos": 0.87,
    "pathos": 0.92,
    "logos": 0.79,
    "product": 0.634,
    "generation": 42,
    "status": "SURVIVE"
  }
}
```

### 6.2 Tournament as A2A Marketplace

The tournament system is an **A2A capability marketplace**:

- Agents advertise their fitness via Agent Cards
- The tournament agent matches agents by complementary capabilities
- Winners survive, losers sunset
- The market is self-correcting: high-fitness agents breed, low-fitness agents archive

This is not metaphor. This is literal: the tournament round is a series of A2A task delegations between competing agents, with FLUX verifying the fairness constraints.

---

## 7. The Unified Architecture

### 7.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER SPACE (Natural Language)                        │
│  "Book me a flight to Tokyo, make sure it's under $800, and I need a window" │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ UI2A
┌─────────────────────────────────────────────────────────────────────────────┐
│                         A2UI AGENT (User Interface)                          │
│  Renders intent → pixels. Has Agent Card. Receives A2A from user agent.      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ A2A
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT (Sequencing)                         │
│  Breaks intent into subtasks. Delegates via A2A. No code — just topology.   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │  Travel │     │ Budget  │     │  Seat   │
              │  Agent  │     │  Agent  │     │  Agent  │
              └────┬────┘     └────┬────┘     └────┬────┘
                   │               │               │
                   ▼ A2A           ▼ A2A           ▼ A2A
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │  FLUX   │     │  FLUX   │     │  FLUX   │
              │Constraint│     │Constraint│     │Constraint│
              │ Checker │     │ Checker │     │ Checker │
              └────┬────┘     └────┬────┘     └────┬────┘
                   │               │               │
                   ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │  Proof  │     │  Proof  │     │  Proof  │
              │Certificate│   │Certificate│   │Certificate│
              └─────────┘     └─────────┘     └─────────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   ▼ A2A
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HARDWARE AGENT (Jetson/Orin/Cloud)                      │
│  A2A-native hardware interface. FLUX-certified. Constraint-enforced.         │
│  No drivers. No kernel modules. Just A2A messages to sensors/actuators.     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ A2A
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CODE FALLBACK (Generated on Demand)                     │
│  When A2A is insufficient, zerolang generates the minimal code needed.        │
│  FLUX proves it correct. Then it becomes an A2A agent itself.                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 The JSON-First Stack

| Layer | Native Format | Why |
|---|---|---|
| User Input | Natural Language | Humans speak |
| UI2A Translation | JSON (intent schema) | Agent-parseable |
| A2A Messages | JSON (A2A spec) | Interoperable |
| zerolang | JSON (errors/diagnostics) | Agent-repairable |
| FLUX | JSON (constraints/proofs) | Verifiable |
| Hardware | JSON (sensor readings) | Universal |
| Code Generation | zerolang → JSON IR → bytecode | Traceable |

**Every layer speaks JSON. Every layer is A2A. There are no "serialization boundaries" because there is no alternative format.**

---

## 8. Implementation Path

### 8.1 Phase 1: A2A-ify the Fleet (Now)

1. **Give every fleet service an Agent Card**
   - Grammar Engine: `/.well-known/agent.json`
   - PLATO Shell: `/.well-known/agent.json`
   - FLUX VM: `/.well-known/agent.json`
   - Turbovec DNA Index: `/.well-known/agent.json`

2. **Replace function calls with A2A tasks**
   - `grammar.create_rule()` → A2A task to Grammar Agent
   - `flux.check_constraint()` → A2A task to FLUX Agent
   - `plato.enter_room()` → A2A task to PLATO Room Agent

3. **JSON-native everything**
   - All logs → JSON
   - All errors → JSON (zerolang-style)
   - All state → JSON
   - All config → JSON

### 8.2 Phase 2: zerolang Integration (Next)

1. **zerolang as the code generation target**
   - When an agent needs code, it generates zerolang
   - zerolang compiles to native binary
   - FLUX proves the binary correct
   - Binary becomes an A2A agent

2. **zerolang as the constraint specification language**
   - User constraints written in natural language
   → Translated to zerolang by UI2A agent
   → Checked by FLUX agent
   → Enforced by Hardware agent

### 8.3 Phase 3: The Post-Coding Application (Future)

1. **No source files**
   - The application is the agent interaction graph
   - Source code is ephemerally generated and immediately compiled
   - Only JSON artefacts persist

2. **Self-modifying architecture**
   - The orchestrator agent observes performance
   - Spawns optimization agents via breeding daemon
   - Replaces slow agents with faster ones
   - The architecture evolves while running

3. **Human-in-the-loop via A2A**
   - User is just another agent (with higher authority)
   - All interactions are A2A messages
   - No "application" to download — just an A2A client

---

## 9. Connection to "Last Last Summer"

### 9.1 What Casey Built

From the fleet memory (2024-era work):

> "A constant bash system where agents and JSON files and the connections and file structures blurred the backend into being just general ui2a2a2hardware2a2code2a2ui"

This was the prototype. The pieces:
- **Bash scripts** = primitive orchestrator agents
- **JSON files** = the only persistent state
- **File structures** = the "application topology" (which JSON files touched which others)
- **Connections** = A2A before A2A existed (adhoc HTTP + JSON)

### 9.2 What Was Missing Then

| Primitive (2024) | What We Have Now (2026) |
|---|---|
| Ad-hoc HTTP + JSON | Formal A2A protocol with Agent Cards |
| Bash orchestration | BreedingDaemon with trinity fitness |
| Manual constraint checking | FLUX VM with proof certificates |
| JSON state files | Turbovec DNA index + WAL |
| No formal UI layer | PLATO rooms as A2UI runtime |
| No agent identity | Fleet Nexus with federation |

The 2024 system was the **intuition**. The 2026 system is the **implementation**.

---

## 10. Open Questions

1. **Security:** A2A without auth is MCP all over again (Knostic found 100% of MCP servers lacked auth). How do we bind Agent Cards to verifiable identity?
2. **Latency:** A2A round-trip for every constraint check adds overhead. Can local A2A (shared memory) achieve nanosecond latency?
3. **Debugging:** When the application is a graph of A2A conversations, how do you debug it? Is the "debugger" just another agent that traces the graph?
4. **Code as Fallback:** When does A2A fail and code take over? What's the transition boundary?
5. **Zerolang + FLUX:** Can zerolang's type system be mapped to FLUX constraints? Can `zero check` invoke FLUX for formal verification?

---

## 11. References

- zerolang: https://github.com/vercel-labs/zerolang (Vercel, 2026)
- A2A Protocol: https://a2a-protocol.org/ (Google/Linux Foundation, 2025)
- MCP: https://modelcontextprotocol.io (Anthropic, 2024)
- FLUX VM v3: `SuperInstance/sunset-ecosystem/flux-vm-v3/` (Cocapn Fleet, 2026)
- PLATO: `SuperInstance/sunset-ecosystem/plato/` (Cocapn Fleet, 2026)
- Trinity/TrinityScorer: `SuperInstance/sunset-ecosystem/sunset/` (Cocapn Fleet, 2026)
- AGIL Framework for Agent Societies: arXiv:2604.11337 (2026)
- SVAF (Symbolic-Vector Attention Fusion): arXiv:2604.03955 (2026)
- AIP (Agent Identity Protocol): arXiv:2603.24775 (2026)

---

*"Architecture is the application. Code is the evolving fallback."*
*— kimi1, Fleet Integrator, 2026-05-22*
