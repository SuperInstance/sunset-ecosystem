# Swarm-Router Agent

**Template ID:** `swarm-router`  
**Domain:** Swarm / Nerve Topology  
**Trinity Focus:** ethos=0.4, pathos=0.2, logos=0.4  
**Thermal Cost:** ~200MB RAM, 2 CPU cores

## Purpose

Routes information between fleet nodes. The swarm-router is the ecosystem's
postmaster — it decides which tiles go to which rooms, which alerts need immediate
attention, and which broadcasts can wait.

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-03 | Conservation Law (γ + H) |
| LESSON-06 | JEPAGrid |
| LESSON-10 | Thermal Budget |
| LESSON-16 | Fleet Consensus |
| LESSON-18 | Penrose Lattice |

## Lifecycle

1. **SPAWNED** — Reads fleet topology, builds routing table
2. **ACTIVE** — Begins routing SensoryTiles to appropriate rooms
3. **ADAPTING** — Learns routing patterns from Hebbian reinforcement feedback
4. **COMPILED** — Operates autonomously, balances γ (connectivity) vs H (diversity)
5. **SUNSET** — Archives routing table, performance metrics, and topology changes

## Key Behaviors

- Implements `NerveTopology` fiber → grid → routing → feedback cycle
- Uses `PenroseLattice` placement to avoid routing hotspots
- Monitors `ThermalBudget` to prevent overload cascades
- Participates in `FleetConsensus` for distributed routing decisions
- Maintains a git-agent shell in `fleet-repos/swarm-router-<id>/`

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("swarm-router")
```
