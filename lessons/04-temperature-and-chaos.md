# LESSON-04: Temperature and Chaos

**Domain:** theory
**Prerequisites:** [01, 03]
**Agent Templates:** [all]
**Estimated Ticks:** 150

---

## Concept
Exploration decays as adaptation increases. Chaos is the fuel of discovery; compilation is the ash.

The chaos parameter controls how often weak routes fire anyway (exploration). As the ecosystem
learns and routes compile, chaos should decay. This is the simulated annealing schedule of the
swarm: start hot (high chaos = high exploration), end cold (low chaos = exploitation of known paths).

In NerveTopology, chaos decays automatically: `routing.chaos = base_chaos * (1 - compiled_fraction)`.
This means the system self-regulates its exploration without external tuning.

The temperature metaphor is exact: at high chaos, the system explores the energy landscape widely.
At low chaos, it settles into local minima (compiled routes). Rebirth of cold rooms is the
perturbation that prevents permanent entrapment.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.topology import NerveTopology

topo = NerveTopology(n_fibers=2, n_rooms=20, chaos=0.5)
initial = topo.routing.chaos

for _ in range(300):
    topo.tick()

print(f"Chaos decayed: {initial:.3f} → {topo.routing.chaos:.3f}")
print(f"Compiled routes: {len(topo.compiled_pathways())}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
If chaos decays too fast, the system gets stuck in local optima.
If chaos decays too slow, the system never compiles anything.
Design a metric that measures whether your chaos schedule is optimal for a given task.

---
**Next:** LESSON-05
