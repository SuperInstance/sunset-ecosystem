# LESSON-16: Fleet Consensus

**Domain:** fleet
**Prerequisites:** [07, 11]
**Agent Templates:** [swarm-router, generic]
**Estimated Ticks:** 350

---

## Concept
Resonance detection, murmurs pubsub. The fleet agrees without a leader.

Fleet consensus is how distributed agents agree on shared state without a central coordinator.
The mechanism is resonance detection: agents that see the same pattern "hum" at the same frequency.
The hum spreads via the murmurs pubsub channel. When enough agents hum the same tune,
the pattern is accepted as fleet consensus.

This is not Byzantine fault tolerance — it is ecological consensus. The system does not
try to agree on a single truth. It agrees on a distribution of beliefs, and the dominant
belief becomes the "consensus" for practical purposes.

The Federated Nexus (port 4047) is the fleet's heartbeat. Each ship registers itself
and publishes status. Down ships are detected by missing heartbeats. Divergence is detected
by comparing hash summaries of shared state.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
# Simulated fleet consensus via resonance
agents = [f"agent-{i}" for i in range(10)]
patterns = {"alpha": 0, "beta": 0, "gamma": 0}

# Agents "hum" patterns based on their local state
for agent in agents:
    # Simplified: each agent picks a random pattern
    import random

    p = random.choice(list(patterns.keys()))
    patterns[p] += 1

# Consensus = pattern with most hums
consensus = max(patterns, key=patterns.get)
print(f"Pattern counts: {patterns}")
print(f"Fleet consensus: {consensus} ({patterns[consensus]}/{len(agents)} agents)")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Prove that ecological consensus is robust to Sybil attacks (one agent pretending
to be many) if and only if the resonance threshold is set to require >50% of
IDENTIFIED agents. What identification mechanism would you use in a permissionless fleet?

---
**Next:** LESSON-17
