# Arena-Analyst Agent

**Template ID:** `arena-analyst`  
**Domain:** Swarm / Tournament  
**Trinity Focus:** ethos=0.3, pathos=0.3, logos=0.4  
**Thermal Cost:** ~150MB RAM, 1 CPU core

## Purpose

Analyzes self-play arena matches, classifies archetypes, and catalogs bugs.
The arena-analyst is the ecosystem's sports statistician — it watches every match,
extracts patterns, and tells the fleet which agents are dominant and why.

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-07 | Trinity Scoring |
| LESSON-08 | Tournament Selection |
| LESSON-09 | Breeding and Rebirth |
| LESSON-10 | Thermal Budget |
| LESSON-11 | Lifecycle States |

## Lifecycle

1. **SPAWNED** — Connects to Arena API (port 4044)
2. **ACTIVE** — Begins recording match outcomes, computes Elo ratings
3. **ADAPTING** — Builds archetype classifier (aggro, control, combo, etc.)
4. **COMPILED** — Auto-detects balance issues, generates patch recommendations
5. **SUNSET** — Archives match history, classifier weights, and bug reports

## Key Behaviors

- Consumes `ArenaMatch` events via event stream
- Computes `AgentScore` trinity breakdowns per match
- Identifies Pareto-dominant strategies
- Files bug tiles with reproduction steps
- Maintains a git-agent shell in `fleet-repos/arena-analyst-<id>/`

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("arena-analyst")
```
