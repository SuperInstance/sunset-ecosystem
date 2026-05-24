# LESSON-09: Breeding and Rebirth

**Domain:** swarm+nerve
**Prerequisites:** [06, 08]
**Agent Templates:** [mud-expert, arena-analyst, lore-keeper]
**Estimated Ticks:** 300

---

## Concept
Tournament winners breed → children fill sunset rooms. The ecosystem regenerates itself.

Breeding is not sexual reproduction. It is asexual cloning with noise: the winner's weights
are copied to a sunset room, then light mutation is applied. This preserves the winner's
"muscle memory" while introducing enough variation to explore adjacent possibilities.

The breeding daemon (swarm/breeder_daemon.py) manages this lifecycle:
1. Detect sunset candidates (cold rooms, dominated agents)
2. Select parents via Pareto frontier or DNA similarity search
3. Spawn children with template-guided biases (ethos/pathos/logos weights)
4. Track lineage for autopsy and learning

Rebirth is more radical: completely new random weights, no parent. Used when the ecosystem
needs fresh diversity or when all existing rooms have converged to similar firing patterns.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.room_grid import RoomGrid
from swarm.tournament import AgentScore

# Simulate ecosystem regeneration
g = RoomGrid(n=20)
for _ in range(20):
    g.tick(np.random.randn(64))

cold = g.cold()
top = g.top(k=3)
print(f"Cold (sunset): {cold}")
print(f"Top (parents): {top}")

if cold and top:
    parent = top[0][0]
    child = cold[0]
    g.breed(parent, child)
    print(f"Bred {parent} -> {child}: new activity = {g.activity[child]}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Compare breeding (clone + noise) vs rebirth (random) as diversity mechanisms.
Under what conditions does each dominate? Design an experiment that measures
the "effective diversity" of a grid after 100 generations of each strategy.

---
**Next:** LESSON-10
