# LESSON-12: Hint Schedules

**Domain:** distill
**Prerequisites:** [11]
**Agent Templates:** [distill-teacher, generic]
**Estimated Ticks:** 150

---

## Concept
10 → 0 hint levels, consecutive wins reduce hints. Guidance fades as competence grows.

The hint schedule is the training wheels of the ecosystem. New agents get maximum hints
(level 10): explicit instructions, example outputs, error correction. As the agent wins
tournaments and compiles patterns, hints reduce.

The reduction schedule:
- Win 3 tournaments → hint level 7
- Win 5 more → hint level 3
- Win 10 total → hint level 0 (autonomous)

This is not punishment — it is weaning. The agent must eventually operate without
external guidance. A teacher that never reduces hints creates dependency, not competence.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
# Hint schedule simulation
def hint_level(wins):
    if wins < 3: return 10
    if wins < 8: return 7
    if wins < 15: return 3
    return 0

for wins in [0, 2, 5, 10, 20]:
    print(f"Wins: {wins:2d} → Hint level: {hint_level(wins)}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design a hint schedule where the agent can REQUEST hints when stuck,
but each request reduces its tournament score (it is admitting weakness).
Does this create honest signaling or gaming behavior?

---
**Next:** LESSON-13
