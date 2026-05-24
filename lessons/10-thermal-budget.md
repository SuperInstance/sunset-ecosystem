# LESSON-10: Thermal Budget

**Domain:** swarm
**Prerequisites:** [08]
**Agent Templates:** [swarm-router, distill-teacher]
**Estimated Ticks:** 250

---

## Concept
65-agent cap, parent sacrifice, hysteresis. The hardware is the ceiling.

The thermal budget is a hard constraint: the ecosystem cannot spawn more agents than
the hardware can sustain. The cap is enforced by the breeding daemon, which checks
`DeviceBudget.available()` before spawning.

When at capacity, a new spawn requires a parent sacrifice. The parent is not deleted —
it is archived (sunset) and its DNA vector is stored in the autopsy database. This
preserves lineage information for future analysis.

Hysteresis prevents oscillation: once the cap is hit, spawning is blocked until
capacity drops below 60% (not 100%). This prevents rapid spawn/sunset cycles that
would destabilize the ecosystem.

## The Tripartite View
| Phase | How this concept manifests |
|-------|---------------------------|
| COLLECT | Gather raw signals / agent scores / hardware metrics |
| SELECT | Novelty gating / tournament rounds / Pareto filtering |
| COMPILE | Hebbian reinforcement / lifecycle transitions / breeding |

## Code Example
```python
from swarm.thermal import ThermalBudget, DeviceType

budget = ThermalBudget()
budget.add_device(DeviceType.GPU, 8_000_000)   # 8GB VRAM
budget.add_device(DeviceType.CPU, 16_000_000)  # 16GB RAM

print(f"Total budget: {budget.total()}")
print(f"Used: {budget.used()}")
print(f"Available: {budget.available()}")

# Simulate agent spawn
agent_cost = 150_000  # ~150MB per agent
n_agents = budget.available() // agent_cost
print(f"Can spawn {n_agents} agents at current capacity")
```

## Verification
- [ ] Agent can explain the concept in its own words
- [ ] Agent can apply it to a novel situation
- [ ] Agent passes the exercise below

## Exercise
If hysteresis is set to 60%, what is the maximum sustainable spawn rate
that prevents oscillation? Derive the relationship between spawn rate, sunset rate,
and hysteresis threshold.

---
**Next:** LESSON-11
