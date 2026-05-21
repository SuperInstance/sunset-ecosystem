# LESSON-14: Fingerprinting

**Domain:** nerve
**Prerequisites:** [05, 06]
**Agent Templates:** [mud-expert, arena-analyst]
**Estimated Ticks:** 150

---

## Concept
3 reference signals uniquely identify each room. The fingerprint is the room's identity.

Every room responds differently to three canonical test signals:
- **sine**: A pure sinusoid — tests frequency response
- **noise**: White noise — tests broadband sensitivity
- **step**: A step function — tests edge detection and latency

The fingerprint is the concatenation of the three output latents. Two rooms with
different fingerprints are guaranteed to have different weight configurations.

Fingerprints enable:
- Duplicate detection (identical fingerprints = identical rooms)
- Lineage tracking (parent/child fingerprint similarity)
- Grid health monitoring (fingerprint diversity = grid diversity)

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from nerve.room_grid import RoomGrid

g = RoomGrid(n=5)
prints = g.fingerprints(n=5)

for p in prints:
    print(f"Room {p.i}: activity={p.activity}")

# Fingerprint diversity
diffs = []
for i, a in enumerate(prints):
    for j, b in enumerate(prints):
        if i < j:
            diffs.append(a.diff(b))
print(f"Mean pairwise diff: {sum(diffs)/len(diffs):.3f}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Design a "fingerprint hash" that maps room fingerprints to a compact ID
for fast duplicate detection. The hash must be: (1) deterministic, (2) sensitive
to small weight changes, (3) collision-resistant for 10K rooms.

---
**Next:** LESSON-15
