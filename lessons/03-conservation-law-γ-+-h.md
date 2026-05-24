# LESSON-03: Conservation Law (γ + H)

**Domain:** theory
**Prerequisites:** [01, 02]
**Agent Templates:** [all]
**Estimated Ticks:** 200

---

## Concept
Connectivity (γ) and diversity (H) trade off. Both cannot be maximized simultaneously.

This is the fundamental constraint of the ecosystem. If every agent connects to every other agent
(maximum connectivity), then all agents see the same information and diversity collapses.
If every agent is completely isolated (maximum diversity), then no information flows and
the system cannot coordinate.

The sum γ + H ≈ constant. The ecosystem finds its operating point on this tradeoff curve
through the tournament selection pressure. High selection pressure → high diversity, low connectivity.
Low selection pressure → high connectivity, low diversity.

The Penrose lattice placement (golden angle) is the spatial embodiment of this law: each room
is placed to maximize coverage without repeating patterns, guaranteeing non-uniform connectivity.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
import numpy as np
from swarm.penrose import assign_positions

positions = assign_positions([f"agent-{i}" for i in range(50)])
# Coverage increases with count, but spacing stays non-repeating
angles = [p.angle for p in positions]
print(f"Golden angle spacing: {np.diff(angles)[:5]} rad")
# The non-repeating pattern is the diversity guarantee
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Prove that a fully-connected graph (γ = N²) has Shannon entropy H = 0 for agent outputs.
Then show that a graph with no edges (γ = 0) has maximum H but zero coordination.
Where is the optimal point?

---
**Next:** LESSON-04
