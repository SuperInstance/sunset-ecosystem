# LESSON-06: The JEPAGrid

**Domain:** nerve
**Prerequisites:** [05]
**Agent Templates:** [mud-expert, arena-analyst, swarm-router]
**Estimated Ticks:** 300

---

## Concept
N rooms perceiving in parallel, novelty-gated firing. The grid IS the ecosystem's attention mechanism.

The JEPAGrid runs all rooms simultaneously via batched matrix multiply. Each room computes
its own latent vector. A room fires if its output is novel (cosine distance from recent history > 0.5)
or if chaos probability triggers it.

Novelty gating is critical: without it, every room fires every tick and the system is just
a dense random projection layer. With novelty, only rooms that see something new participate.
This is sparse attention — the grid selects which rooms matter for each input.

The Rust kernel achieves ~2.35ms for 10K rooms via multi-threading. The numpy fallback
is ~5ms. Both use the same weight layout (room-major flat arrays) so results are identical.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.room_grid import RoomGrid
import numpy as np
import time

g = RoomGrid(n=1000)
x = np.random.randn(64)

# Warmup
g.tick(x)

t0 = time.perf_counter()
for _ in range(50):
    g.tick(x)
elapsed = (time.perf_counter() - t0) / 50

print(f"1000 rooms: {elapsed*1000:.2f} ms/tick")
print(f"Active rooms: {g.stats['active']}")
print(f"Cold rooms: {g.stats['cold']}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design an input signal that activates exactly 10% of rooms after 10 ticks.
What properties must the signal have? (Hint: think about the history threshold and chaos decay.)

---
**Next:** LESSON-07
