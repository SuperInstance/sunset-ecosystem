# LESSON-18: The Penrose Lattice

**Domain:** sunset
**Prerequisites:** [03, 06]
**Agent Templates:** [swarm-router, arena-analyst]
**Estimated Ticks:** 300

---

## Concept
Golden angle placement guarantees non-repeating coverage. Nature's quasicrystal in agent space.

The Penrose lattice places agents using the golden angle (≈137.5°), the same angle that
governs sunflower seed spirals, pinecone scales, and nautilus shells. This placement has
a remarkable property: it is maximally non-repeating while maintaining uniform density.

Mathematical basis:
- The golden angle = 2π(1 - φ) where φ = (1+√5)/2 ≈ 1.618
- Sequential placement at this angle fills a disk with no two points at the same angle
- The radial spacing grows as √n to maintain constant density
- The pattern is a quasicrystal: no translational symmetry, but perfect rotational order

In the ecosystem, this means:
- No two agents have identical relative positions
- Local neighborhoods are similar but never identical
- Coverage is uniform without grid artifacts
- The pattern extends infinitely without repetition

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from swarm.penrose import assign_positions
import numpy as np

positions = assign_positions([f"room-{i}" for i in range(200)])
angles = [p.angle for p in positions]
radii = [p.radius for p in positions]

# Verify golden angle
from math import pi, sqrt

golden_angle = 2 * pi * (1 - (1 + sqrt(5)) / 2 + 1)  # 2π(2-φ)
actual_diff = np.mean(np.diff(angles))
print(f"Golden angle: {golden_angle:.6f} rad")
print(f"Mean angle diff: {actual_diff:.6f} rad")
print(f"Match: {np.isclose(actual_diff, golden_angle, atol=0.001)}")

# Verify radial growth
expected_r = sqrt(np.arange(1, 201))  # r ∝ √n
print(f"Radius correlation with √n: {np.corrcoef(radii, expected_r)[0, 1]:.4f}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Prove that the Penrose lattice maximizes the minimum distance between any two
agents for a given density. Compare with hexagonal and square grids. Under what
conditions does the Penrose lattice lose its advantage?

---
**Next:** LESSON-19
