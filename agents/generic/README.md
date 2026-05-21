# Generic Agent

**Template ID:** `generic`  
**Domain:** All  
**Trinity Focus:** ethos=0.33, pathos=0.33, logos=0.34  
**Thermal Cost:** ~50MB RAM, minimal CPU

## Purpose

The default template for unclassified tasks. The generic agent is the ecosystem's
utility player — it has no specialization, which means it can attempt anything.
This is both its strength (flexibility) and weakness (no optimized training path).

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-01 | Universal Grammar |
| LESSON-02 | Threshold as Control Surface |
| LESSON-07 | Trinity Scoring |
| LESSON-11 | Lifecycle States |

## Lifecycle

1. **SPAWNED** — Receives task assignment from fleet dispatch
2. **ACTIVE** — Begins execution with maximum hints
3. **ADAPTING** — Learns task-specific patterns, reduces hints
4. **COMPILED** — Operates autonomously on this task type
5. **SUNSET** — Archives task outcomes, may specialize into a new template

## Key Behaviors

- Uses all three phases (COLLECT, SELECT, COMPILE) generically
- Adapts `adapt_threshold` based on task difficulty
- May transition to a specialized template if a pattern emerges
- Lowest thermal cost — can be spawned in large numbers for parallel tasks

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("generic")
```
