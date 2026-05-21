# LESSON-11: Lifecycle States

**Domain:** swarm
**Prerequisites:** [09, 10]
**Agent Templates:** [all]
**Estimated Ticks:** 200

---

## Concept
SPAWNED → ACTIVE → ADAPTING → COMPILED → SUNSET. Every agent walks this path.

The lifecycle is the agent's existential journey. Each state has distinct behaviors:

- **SPAWNED**: Fresh weights, maximum chaos, full hint level. Agent explores blindly.
- **ACTIVE**: Agent has survived first tournament. Chaos begins decaying.
- **ADAPTING**: Agent is building confidence (epsilon accumulation). Hints reduce.
- **COMPILED**: Agent operates autonomously. No hints, minimal chaos. Muscle memory.
- **SUNSET**: Agent is archived. DNA stored, room freed for rebirth.

The Breeder class (swarm/breeder.py) manages transitions:
- `spawn_from_template()` → SPAWNED
- `evolve()` → transitions based on tournament results
- `tick_all()` → per-tick state updates
- `sunset_room()` → SUNSET

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from swarm.breeder import Breeder, AgentLifecycle
from nerve.templates import TemplateRegistry

registry = TemplateRegistry()
breeder = Breeder(template_registry=registry)

agent = breeder.spawn_from_template("generic")
print(f"Initial state: {agent.state.value}")

# Simulate progression
agent.state = AgentLifecycle.ACTIVE
agent.state = AgentLifecycle.ADAPTING
print(f"After adaptation: {agent.state.value}")

# Compilation threshold
if agent.confidence >= 0.95:
    agent.state = AgentLifecycle.COMPILED
    print(f"Compiled: {agent.state.value}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Map the lifecycle states to the Soft → Snap → Hard progression.
What happens to an agent that skips ADAPTING and jumps directly from ACTIVE to COMPILED?
Is this possible? Under what conditions?

---
**Next:** LESSON-12
