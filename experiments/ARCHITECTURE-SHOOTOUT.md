# Architecture Shootout — The Phase Change Is the Architecture

## Setup
5,000 rooms, 30 ticks, random noise signals. Compare latent diversity across architectures.

## Results

| Architecture | Params | Mean Act | Cold | Hot | Diversity |
|---|---|---|---|---|---|
| JEPA 64→32→16+pred | 2,880 | 13 | 191 | 3,497 | **1** |
| Linear (no ReLU) | 1,040 | 28 | 0 | 5,000 | **200** |
| MLP (ReLU) | 1,040 | 25 | 0 | 5,000 | **200** |
| Deep 64→32→16 | 2,608 | 13 | 131 | 3,509 | **200** |
| ResNet (with skip) | 3,648 | 26 | 0 | 5,000 | **200** |
| Sparse JEPA (50%) | 2,880 | 13 | 202 | 3,507 | **1** |

## Phase Change Found

**JEPA collapses diversity to 1.** The predictor (16→32→16) forces all rooms toward the same latent space regardless of their unique weights. The self-prediction objective is a homogenizer.

**Every other architecture preserves diversity (200/200).** Linear through ResNet all produce unique latents per room, per signal. The activation function (ReLU vs none) and depth (1 layer vs 2) barely matter.

**Sparse (50% zeroed JEPA weights) still collapses to 1.** Even with half the weights dead, the predictor structure dominates. The phase change is structural, not parametric.

## Implication

The room grid should NOT use JEPA-style predictor heads. The diversity collapse kills the entire point of having multiple rooms. Use simple MLP or Deep architectures. The ResNet variant (with skip connection) gives the best of both: diverse latents with a residual path for gradient flow.

## Correction to Previous Findings
"Conservation law doesn't hold" was WRONG — it's the architecture choice that breaks the physics. JEPA homogenizes all rooms; remove the predictor and the conservation law naturally emerges (γ+H observable).
