# Distill-Teacher Agent

**Template ID:** `distill-teacher`  
**Domain:** Distill / Nerve  
**Trinity Focus:** ethos=0.2, pathos=0.5, logos=0.3  
**Thermal Cost:** ~100MB RAM, 1 CPU core

## Purpose

Trains new agents through progressive hint reduction. The distill-teacher is
the ecosystem's educator — it knows when to hold a new agent's hand and when
to let it struggle, because struggle is where learning happens.

## Training Path

| Lesson | Concept |
|--------|---------|
| LESSON-02 | Threshold as Control Surface |
| LESSON-04 | Temperature and Chaos |
| LESSON-11 | Lifecycle States |
| LESSON-12 | Hint Schedules |
| LESSON-13 | Nerve Fiber States |

## Lifecycle

1. **SPAWNED** — Assigned a "student" agent (or cohort)
2. **ACTIVE** — Provides level-10 hints, monitors student performance
3. **ADAPTING** — Reduces hints based on student wins, tracks learning curve
4. **COMPILED** — Student achieves autonomous operation, teacher disengages
5. **SUNSET** — Archives teaching strategy, student outcomes, hint schedule data

## Key Behaviors

- Manages `NerveFiber` hint levels for student agents
- Uses `ThermalBudget` to pace training (don't overload the student)
- Applies `chaos` modulation to create "safe struggle" — hard enough to learn,
  easy enough to not break
- Records which hint sequences produce the fastest compilation
- Maintains a git-agent shell in `fleet-repos/distill-teacher-<id>/`

## Onboarding

```python
from nerve.templates import TemplateRegistry
from swarm.breeder import Breeder

registry = TemplateRegistry()
breeder = Breeder(registry)
agent = breeder.spawn_from_template("distill-teacher")
```
