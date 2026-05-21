# LESSON-19: Eisenstein Weight Snap

**Domain:** plato
**Prerequisites:** [14, 18]
**Agent Templates:** [lore-keeper, distill-teacher]
**Estimated Ticks:** 400

---

## Concept
Hexagonal lattice quantization for gradient compression. Nature's most efficient packing.

Eisenstein integers are complex numbers of the form a + bω where ω = e^(2πi/3).
They form a hexagonal lattice in the complex plane — the most efficient 2D packing.

Weight snap maps continuous neural network weights to the nearest Eisenstein integer,
then scales back. This achieves 2-4× compression with minimal accuracy loss because:
1. The hexagonal lattice has the highest packing density in 2D (π/√12 ≈ 0.907)
2. The quantization error is isotropic (same in all directions)
3. The snap operation is a simple rounding + basis transformation

In the ecosystem, weight snap is used for:
- Gradient compression during fleet broadcast (less bandwidth)
- Model checkpoint storage (smaller files)
- Cross-ship weight transfer (faster synchronization)

The forge_kernels.cu CUDA implementation provides GPU-accelerated snap for large tensors.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
import numpy as np

# Simplified Eisenstein snap for 1D weights
def eisenstein_snap(w, scale=16):
    """Snap weights to Eisenstein lattice points."""
    # Scale up, snap to nearest integer lattice, scale down
    w_scaled = w * scale
    w_snapped = np.round(w_scaled)
    return w_snapped / scale

# Test compression
w = np.random.randn(1000).astype(np.float32)
w_snap = eisenstein_snap(w, scale=8)
error = np.abs(w - w_snap).mean()
print(f"Mean quantization error: {error:.6f}")
print(f"Unique values before: {len(np.unique(w))}")
print(f"Unique values after: {len(np.unique(w_snap))}")
print(f"Effective compression: {len(np.unique(w)) / len(np.unique(w_snap)):.1f}x")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Extend Eisenstein snap to 2D weight matrices (like w1: 64×32). The hexagonal
lattice must be embedded in the higher-dimensional space. Is the Voronoi cell still
optimal? What is the compression ratio for a ResNet-50's conv layers?

---
**Next:** LESSON-END
