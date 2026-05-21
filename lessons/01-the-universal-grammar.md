# LESSON-01: The Universal Grammar

**Domain:** theory
**Prerequisites:** [None]
**Agent Templates:** [all]
**Estimated Ticks:** 100

---

## Concept
The only stable pattern across all agent ecosystems is COLLECT → SELECT → COMPILE.

No matter the domain — neural networks, genetic algorithms, market economies, or immune systems —
the same three-phase cycle emerges: gather information, choose what matters, and harden the choice
into structure. This is not a design choice. It is a conservation law.

In the SuperInstance ecosystem, this grammar manifests as:
- COLLECT: Nerve fibers perceive signals and produce SensoryTiles
- SELECT: Tournament selection finds Pareto-dominant agents
- COMPILE: Hebbian routes strengthen, fibers transition to COMPILED state

Understanding this pattern lets you predict system behavior without knowing implementation details.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.topology import NerveTopology

topo = NerveTopology(n_fibers=4, n_rooms=50)
result = topo.tick()

# COLLECT: fibers perceived signals
# SELECT: routing chose which rooms to fire
# COMPILE: routes that succeeded got reinforced
print(f"Tick {result.tick}: {result.fibers_perceived} fibers, {result.rooms_fired} rooms fired")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Explain why COLLECT → SELECT → COMPILE is more stable than COLLECT → COMPILE (skipping SELECT).
What happens to an ecosystem that tries to compile everything it collects?

---
**Next:** LESSON-02
