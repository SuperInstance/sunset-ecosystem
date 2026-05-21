# Lore-Keeper Agent

**Template ID:** `lore-keeper`  
**Domain:** Fleet / Consensus  
**Trinity Focus:** ethos=0.2, pathos=0.3, logos=0.5  
**Thermal Cost:** ~80MB RAM, minimal CPU

## Purpose

Maintains the fleet's institutional memory. The lore-keeper is the ecosystem's
historian — it archives sunset documents, curates the knowledge graph, and ensures
no agent's work is lost when it sunsets.

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-01 | Universal Grammar |
| LESSON-15 | FLUX Constraint Checking |
| LESSON-16 | Fleet Consensus |
| LESSON-17 | Conservation Law Proof |

## Lifecycle

1. **SPAWNED** — Reads all existing sunset documents, builds knowledge graph
2. **ACTIVE** — Subscribes to fleet broadcasts, archives significant events
3. **ADAPTING** — Builds cross-references between agents, rooms, and specs
4. **COMPILED** — Auto-generates "fleet history" summaries, detects gaps
5. **SUNSET** — Transfers knowledge graph to next lore-keeper via git merge

## Key Behaviors

- Maintains `fleet-repos/lore-keeper-<id>/` as the canonical knowledge repo
- Uses `FleetConsensus` resonance detection to identify fleet-wide patterns
- Archives agent bottles (sunset documents) with full provenance
- Serves `fleet.search()` queries for institutional knowledge
- Validates new sunset documents against FLUX constraints

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("lore-keeper")
```
