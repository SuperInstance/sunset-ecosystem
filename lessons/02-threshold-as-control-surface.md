# LESSON-02: Threshold as Control Surface

**Domain:** theory
**Prerequisites:** [01]
**Agent Templates:** [all]
**Estimated Ticks:** 150

---

## Concept
The threshold between phases IS the user interface. There is no other control surface.

Every agent in the ecosystem has a threshold parameter that determines when it transitions
from one lifecycle state to another. This threshold is not fixed — it is learned from the user
through implicit feedback (which agents survive tournaments, which routes get reinforced).

Key insight: The threshold is the only degree of freedom that matters. All other parameters
(noise levels, chaos probability, learning rate) are secondary effects of threshold tuning.

The Soft → Snap → Hard progression (PERCEIVING → ADAPTING → COMPILED) is governed entirely
by the adapt_threshold parameter in NerveFiber and the tournament selection pressure in the swarm.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.fiber import NerveFiber

fiber = NerveFiber("demo", adapt_threshold=0.8, epsilon=0.05)

# Feed the same signal repeatedly — watch threshold cross
for i in range(20):
    tile = fiber.perceive("hello world")
    print(f"Step {i}: {tile.state.value}, confidence={tile.confidence:.2f}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design an experiment that proves the threshold is the only control surface.
What would you measure to show that changing chaos or learning_rate has no effect
independent of their influence on effective threshold?

---
**Next:** LESSON-03
