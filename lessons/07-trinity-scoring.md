# LESSON-07: Trinity Scoring

**Domain:** swarm
**Prerequisites:** [01, 03]
**Agent Templates:** [all]
**Estimated Ticks:** 200

---

## Concept
ethos × pathos × logos = fitness. Zero in any dimension causes agent sunset.

The trinity is multiplicative, not additive. An agent with ethos=1.0 (perfect hardware fit),
pathos=1.0 (perfect user alignment), but logos=0.0 (broken code) has fitness = 0.0.
It sunsets regardless of its other strengths.

- **ethos**: Hardware affinity — does this agent run efficiently on available devices?
- **pathos**: User alignment — does this agent produce output the user values?
- **logos**: Code quality — is the agent's implementation correct, tested, documented?

The tournament system enforces this: agents compete pairwise, and a Pareto-dominated agent
(is worse in all dimensions than another) is automatically sunset. The frontier is the set
of agents that are not dominated by anyone.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from swarm.tournament import AgentScore

# Two agents competing
agent_a = AgentScore(ethos=0.9, pathos=0.8, logos=0.7)
agent_b = AgentScore(ethos=0.7, pathos=0.9, logos=0.8)

print(f"Agent A fitness: {agent_a.fitness:.3f}")
print(f"Agent B fitness: {agent_b.fitness:.3f}")

# A dominates B if A is better in ALL dimensions
a_dominates_b = all([
    agent_a.ethos >= agent_b.ethos,
    agent_a.pathos >= agent_b.pathos,
    agent_a.logos >= agent_b.logos,
]) and any([
    agent_a.ethos > agent_b.ethos,
    agent_a.pathos > agent_b.pathos,
    agent_a.logos > agent_b.logos,
])
print(f"A dominates B: {a_dominates_b}")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
Construct a scenario where two agents have identical fitness scores
but one should sunset and the other should survive. (Hint: look at the variance
across dimensions, not just the product.)

---
**Next:** LESSON-08
