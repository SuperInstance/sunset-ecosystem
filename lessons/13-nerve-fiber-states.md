# LESSON-13: Nerve Fiber States

**Domain:** nerve
**Prerequisites:** [05, 11]
**Agent Templates:** [mud-expert, lore-keeper]
**Estimated Ticks:** 200

---

## Concept
PERCEIVING → ADAPTING → COMPILED with thresholds. The fiber's journey from raw to routine.

Nerve fibers are the sensory organs of the ecosystem. Their lifecycle mirrors human perception:

- **PERCEIVING**: Full attention, raw feature extraction. Like feeling every edge of a new shoe.
- **ADAPTING**: Pattern recognition building. Like learning to walk in the shoe.
- **COMPILED**: Automatic processing, no conscious attention. Like forgetting you're wearing shoes.
- **NOVELTY_ALERT**: Something changed — full attention returns. Like a rock in the shoe.

The transitions are threshold-gated by confidence (epsilon accumulation) and novelty detection.
A COMPILED fiber that sees a novel signal snaps back to PERCEIVING via NOVELTY_ALERT.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.fiber import NerveFiber, FiberState

fiber = NerveFiber("demo", adapt_threshold=0.95, epsilon=0.05)

# Repeated signal → compilation
signal = "the quick brown fox"
for i in range(30):
    tile = fiber.perceive(signal)
    if tile.state == FiberState.COMPILED:
        print(f"Compiled at step {i}!")
        break

# Novel signal → novelty alert
tile = fiber.perceive("completely different input")
print(f"Novelty response: {tile.state.value}, confidence={tile.confidence:.2f}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
A fiber that never sees novel signals will stay COMPILED forever.
Is this desirable or dangerous? Design a "scheduled novelty injection" system
that periodically presents synthetic novel signals to prevent fossilization.

---
**Next:** LESSON-14
