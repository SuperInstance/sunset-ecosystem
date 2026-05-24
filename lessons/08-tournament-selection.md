# LESSON-08: Tournament Selection

**Domain:** swarm
**Prerequisites:** [07]
**Agent Templates:** [all]
**Estimated Ticks:** 250

---

## Concept
Pairwise competition, Pareto frontier, dominated agents sunset. Natural selection without fitness landscapes.

Tournament selection avoids the need for a global fitness function. Instead of asking
"what is the absolute best agent?", we ask "which agent wins head-to-head?". This is
robust to non-stationary environments where "best" changes over time.

The tournament proceeds in rounds:
1. Pair agents randomly
2. Compare trinity scores
3. Winners advance, losers enter sunset pool
4. Frontier agents (non-dominated) always survive
5. Breed winners to fill sunset slots

The dynamic cap (65 agents) is enforced by tournament size: if population exceeds cap,
the bottom fraction is auto-sunset regardless of pairwise results. This prevents infinite
growth while preserving selection pressure.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from swarm.tournament import breed, sunset_candidates
from nerve.room_grid import RoomGrid

# Simulate a tournament round
g = RoomGrid(n=10)
for _ in range(5):
    g.tick(np.random.randn(64))

# Cold rooms are sunset candidates
cold = g.cold()
print(f"Sunset candidates: {cold}")

# Breed a winner into a sunset slot
if len(cold) > 0 and len(g.top(k=1)) > 0:
    winner = g.top(k=1)[0][0]
    loser = cold[0]
    g.breed(winner, loser)
    print(f"Bred room {winner} into {loser}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design a tournament variant where agents can form coalitions.
A coalition's score is the product of its members' scores, but the coalition
must agree on all decisions. Does this increase or decrease diversity? Why?

---
**Next:** LESSON-09
