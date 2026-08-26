# LESSON-05: The JEPA Room

**Domain:** nerve
**Prerequisites:** [01]
**Agent Templates:** [mud-expert, arena-analyst]
**Estimated Ticks:** 250

---

## Concept
One room = 3-layer MLP, 2,560 params, perceives signals through its unique random weights.

A JEPA room is not trained. It is born with random weights and lives or dies based on whether
its firing pattern is useful to the ecosystem. This is the "predictive coding without backprop"
insight: diversity of random projections captures enough signal structure to be useful.

Room weights: w1 (64→32), w2 (32→16), w3 (16→16). The near-identity w3 preserves room
identity — each room's output is recognizably its own, even after nonlinear transforms.

The forward pass is pure matmul + ReLU: x @ w1 → ReLU → @w2 → ReLU → @w3 + biases.
No training. No gradients. The room's "learning" happens through rebirth (new random weights)
or breeding (clone + noise from a successful parent).

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.room_grid import RoomGrid, make_weights
import numpy as np

# One room's weights
w = make_weights(n=1, d=64, h=32, l=16)
print(f"w1 shape: {w['w1'].shape}")  # (1, 64, 32)
print(f"Total params: {sum(v.size for v in w.values())}")

# Two rooms, same input, different outputs
g = RoomGrid(n=2)
out = g._forward(np.random.randn(64))
print(f"Room 0 latent: {out[0][:4]}")
print(f"Room 1 latent: {out[1][:4]}")
print(f"Correlation: {np.corrcoef(out[0], out[1])[0, 1]:.3f}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Prove that two rooms with independent random weights produce uncorrelated outputs
for Gaussian inputs (in expectation). Then show that breeding (clone + noise) creates
positive correlation between parent and child. What is the optimal noise level for
maintaining diversity while preserving useful correlations?

---
**Next:** LESSON-06
