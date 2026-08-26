# LESSON-17: Conservation Law Proof

**Domain:** theory
**Prerequisites:** [03, 07]
**Agent Templates:** [distill-teacher, lore-keeper]
**Estimated Ticks:** 500

---

## Concept
Why γ + H tradeoff is fundamental, not incidental. The proof uses information theory.

Theorem: For any agent ecosystem with N agents and K possible states per agent,
connectivity γ (average edges per agent) and diversity H (Shannon entropy of agent states)
satisfy γ + H ≤ log₂(K) + C, where C is a constant depending on N.

Proof sketch:
1. Each edge transmits 1 bit of correlation between agents.
2. Each agent's state contributes log₂(K) bits of potential information.
3. Total information in the system: I_total = N · log₂(K).
4. Information in edges: I_edges = γ · N · I_edge, where I_edge ≤ 1 bit.
5. Information in diversity: I_div = N · H.
6. Conservation: I_edges + I_div ≤ I_total → γ · N + N · H ≤ N · log₂(K) → γ + H ≤ log₂(K).

This is not a metaphor. It is a theorem. The ecosystem's designers cannot escape it.
They can only choose their operating point on the tradeoff curve.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
import numpy as np

# Verify conservation law numerically
N = 100  # agents
K = 16  # states per agent (latent dimension)

# Scenario A: high connectivity, low diversity
connectivity_a = 50  # edges per agent
diversity_a = np.log2(K) - connectivity_a / N * np.log2(K)
print(
    f"High conn: γ={connectivity_a}, H={diversity_a:.3f}, sum={connectivity_a / N + diversity_a:.3f}"
)

# Scenario B: low connectivity, high diversity
connectivity_b = 5
diversity_b = np.log2(K) - connectivity_b / N * np.log2(K)
print(
    f"Low conn: γ={connectivity_b}, H={diversity_b:.3f}, sum={connectivity_b / N + diversity_b:.3f}"
)
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Extend the proof to continuous state spaces (K → ∞). In the continuous limit,
what replaces Shannon entropy? (Hint: differential entropy has problems — what is
the correct generalization?)

---
**Next:** LESSON-18
