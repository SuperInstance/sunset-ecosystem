# SPEC-TUTOR.md
**Author:** CCC (Systems Architect)  
**Date:** 2026-05-21  
**Status:** ARCHITECTURE — Lesson format and agent onboarding

---

## 1. Purpose

The tutor system is how agents onboard into the SuperInstance ecosystem. Currently there are no dedicated "tutor repos" — the closest is `dodecet-encoder/tutorials/`. The real onboarding happens through the ecosystem's own documentation: STRUCTURAL-SURVEY, THEORY-OF-ECOSYSTEMS, and the 7 architecture docs referenced there.

This spec defines the lesson format and how agents (including external LLM agents) learn the ecosystem.

## 2. Lesson Format

Each lesson is a markdown file following this structure:

```markdown
# LESSON-{N}: {Title}

**Domain:** {nerve | swarm | sunset | fleet | flux}
**Prerequisites:** [LESSON-X, LESSON-Y]
**Agent Templates:** [which templates this lesson trains]
**Estimated Ticks:** {how many JEPAGrid ticks to internalize}

---

## Concept
{One-paragraph explanation of the concept}

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | ... |
| SELECT | ... |
| COMPILE | ... |

## Code Example
```python
# Minimal working example using sunset-ecosystem APIs
```

## Verification
{How to verify the agent internalized this lesson}
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
{A task that requires applying the concept, not just recalling it}

---
**Next:** LESSON-{N+1}: {next topic}
```

## 3. Lesson Sequence

### Tier 1: Core Concepts (any agent)

| Lesson | Title | Domain | Key Idea |
|--------|-------|--------|----------|
| 01 | The Universal Grammar | theory | COLLECT→SELECT→COMPILE is the only stable pattern |
| 02 | Threshold as Control Surface | theory | The threshold between phases IS the user interface |
| 03 | Conservation Law (γ + H) | theory | Connectivity and diversity trade off; both can't be max |
| 04 | Temperature and Chaos | theory | Exploration decays as adaptation increases |
| 05 | The JEPA Room | nerve | One room = 3-layer MLP, 2.5K params, perceives signals |
| 06 | The JEPAGrid | nerve | N rooms perceiving in parallel, novelty-gated firing |
| 07 | Trinity Scoring | swarm | ethos × pathos × logos — zero in any dimension kills |
| 08 | Tournament Selection | swarm | Pairwise competition, Pareto frontier, dominated agents |

### Tier 2: Integration (specialized agents)

| Lesson | Title | Domain | Key Idea |
|--------|-------|--------|----------|
| 09 | Breeding and Rebirth | swarm+nerve | Tournament winners breed → children fill sunset rooms |
| 10 | Thermal Budget | swarm | 65-agent cap, parent sacrifice, hysteresis |
| 11 | Lifecycle States | swarm | SPAWNED→ACTIVE→ADAPTING→COMPILED→SUNSET |
| 12 | Hint Schedules | distill | 10→0 hint levels, consecutive wins reduce hints |
| 13 | Nerve Fiber States | nerve | PERCEIVING→ADAPTING→COMPILED with thresholds |
| 14 | Fingerprinting | nerve | 3 reference signals uniquely identify each room |

### Tier 3: Advanced (senior agents)

| Lesson | Title | Domain | Key Idea |
|--------|-------|--------|----------|
| 15 | FLUX Constraint Checking | flux | Stack machine, proof certificates, terminating by design |
| 16 | Fleet Consensus | fleet | Resonance detection, murmurs pubsub |
| 17 | Conservation Law Proof | theory | Why γ + H tradeoff is fundamental, not incidental |
| 18 | The Penrose Lattice | sunset | Golden angle placement guarantees non-repeating coverage |
| 19 | Eisenstein Weight Snap | plato | Hexagonal lattice quantization for gradient compression |

## 4. Agent Onboarding Protocol

When a new agent enters the ecosystem (via template spawn), it follows this sequence:

```
Day 0: Spawn into JEPAGrid (SPAWNED state)
    ├── Load Lessons 01-04 (theory core)
    ├── chaos = 0.3 (high exploration)
    └── hint_level = 10 (fully guided)

Days 1-5: Active learning (ACTIVE state)
    ├── Load Lessons 05-08 (domain skills)
    ├── chaos decays: 0.3 → 0.15
    └── hint_level: 10 → 7

Days 5-15: Specialization (ADAPTING state)
    ├── Load Lessons based on template (Tier 2)
    ├── chaos: 0.15 → 0.05
    └── hint_level: 7 → 3

Days 15+: Autonomous (COMPILED state)
    ├── Access to Tier 3 on demand
    ├── chaos: ~0.01
    └── hint_level: 0 (no hints)
```

The "days" are not calendar days — they're tick-based. The tick rate depends on how often the grid receives input signals.

## 5. Knowledge Representation

Agent knowledge is stored in two places:

1. **Room weights (implicit):** The JEPA room's learned response to signals. This is "muscle memory" — not inspectable, but functional.

2. **Lesson completion (explicit):** A JSON file tracking which lessons the agent has completed:

```json
{
  "agent_id": "mud-expert-0042",
  "template": "mud-expert",
  "room": 42,
  "lessons_completed": [1, 2, 3, 4, 5, 6, 7, 8],
  "lessons_current": [9],
  "exercises_passed": {
    "01": "Explained grammar as thesis-antithesis-synthesis with threshold",
    "05": "Correctly predicted room output for sine input"
  },
  "ticks_at_last_lesson": 145,
  "generation": 3
}
```

This file lives at `sunset-ecosystem/agents/{agent_id}/progress.json`.

## 6. Verification Mechanism

How do we know an agent learned? Three levels:

### Level 1: Signal Response
After loading a lesson, the agent's room should respond differently to relevant signals. Compare fingerprints before and after — if they're identical, the lesson didn't take.

### Level 2: Behavioral Test
Present the agent with a scenario from the lesson. Check if the agent's response (through its room firing pattern and activity level) matches expected behavior.

### Level 3: Tournament Performance
The ultimate test: does the agent survive tournaments? If it keeps getting dominated, it hasn't internalized the lessons. If it makes the Pareto frontier, it has.

## 7. File Structure

```
sunset-ecosystem/
├── lessons/
│   ├── 01-universal-grammar.md
│   ├── 02-threshold-control.md
│   ├── 03-conservation-law.md
│   ├── 04-temperature-chaos.md
│   ├── 05-jepa-room.md
│   ├── 06-jepa-grid.md
│   ├── 07-trinity-scoring.md
│   ├── 08-tournament-selection.md
│   ├── 09-breeding-rebirth.md
│   ├── 10-thermal-budget.md
│   ├── 11-lifecycle-states.md
│   ├── 12-hint-schedules.md
│   ├── 13-nerve-fibers.md
│   ├── 14-fingerprinting.md
│   ├── 15-flux-constraints.md
│   ├── 16-fleet-consensus.md
│   ├── 17-conservation-proof.md
│   ├── 18-penrose-lattice.md
│   └── 19-eisenstein-snap.md
├── agents/
│   └── {agent_id}/
│       └── progress.json
└── nerve/
    └── templates.py   ← templates reference lesson sequences
```

## 8. External Agent Onboarding

For LLM agents joining the ecosystem (e.g., a new Claude session reading the codebase):

1. Read `docs/STRUCTURAL-SURVEY.md` — this is the map
2. Read `docs/THEORY-OF-ECOSYSTEMS.md` — this is the physics
3. Read the relevant SPEC files for the domain they'll work in
4. Read the source code for their assigned component
5. Verify by explaining the system back in their own words

This is what the current docs already enable. The lesson system makes it formal for spawned agents.
