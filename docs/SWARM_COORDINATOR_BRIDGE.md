# Swarm Coordinator Bridge

*Integration target: `Equipment-Swarm-Coordinator`*

Brings the Equipment-Swarm-Coordinator patterns into sunset-ecosystem as a zero-dependency Python module.

## What It Does

- **Agent roles** — coordinator, executor, validator, specialist, observer, scout, builder, auditor
- **Conflict resolution** — voting, weighted, hierarchical, consensus
- **Task decomposition** — parallel, sequential, pipeline, map-reduce, divide-conquer
- **Trust matrix** — per-agent trust scores
- **Knowledge isolation** — strict, moderate, relaxed
- **Task assignment** — match tasks to agents by capabilities
- **ASCII visualization** — render the swarm

## Quick Start

```python
from fleet.swarm_coordinator_bridge import SwarmCoordinator, AgentRole

# Create a swarm with 10 agents max
coordinator = SwarmCoordinator(max_agents=10, knowledge_isolation="moderate")

# Register agents with roles and capabilities
coordinator.register_agent("scout-1", AgentRole.SCOUT, capabilities=["search", "fetch"])
coordinator.register_agent("builder-1", AgentRole.BUILDER, capabilities=["code", "test"])
coordinator.register_agent("boss", AgentRole.COORDINATOR, hierarchy_level=5)

# Set trust scores
coordinator.set_trust("boss", "builder-1", 0.9)
coordinator.set_trust("boss", "scout-1", 0.7)

# Decompose a task into parallel subtasks
nodes = coordinator.decompose_task(
    "Build a bridge",
    strategy="parallel",
    subtasks=["Design", "Code", "Test"],
)

# Assign a task to the best agent
assigned = coordinator.assign_task(nodes[0], required_caps=["code"])
print(f"Assigned to: {assigned}")  # builder-1

# Resolve conflicts between options
report = coordinator.resolve_conflict(
    ["option-a", "option-b"],
    strategy="weighted",
    agent_votes={"boss": "option-a", "builder-1": "option-b"},
)
print(report.winner, report.confidence)

# Visualize the swarm
print(coordinator.render_ascii())
```

## Agent Roles

| Role | Knowledge Level | Typical Capabilities |
|------|----------------|---------------------|
| coordinator | Full | manage, delegate, decide |
| executor | Partial | code, test, build |
| validator | Partial | review, audit, check |
| specialist | Limited | specific domain expertise |
| observer | Minimal | monitor, report, log |
| scout | Minimal | search, explore, discover |
| builder | Partial | code, implement, construct |
| auditor | Partial | review, verify, validate |

## Conflict Resolution Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| voting | Democratic vote | Equal-weight agents |
| weighted | Weighted by trust × performance | Trusted agents with varying skill |
| hierarchical | Highest hierarchy wins | Clear chain of command |
| consensus | > 50% agreement required | Collaborative decisions |

## Task Decomposition Strategies

| Strategy | Description | Dependencies |
|----------|-------------|------------|
| parallel | Independent subtasks | None |
| sequential | Ordered subtasks | Previous subtask |
| pipeline | Staged processing | Previous stage |
| map-reduce | Split → process → merge | Map → Reduce |
| divide-conquer | Split → conquer → merge | Divide → Conquer → Merge |

## Knowledge Isolation

```python
# Strict — only knowledge at or below agent's knowledge_level
SwarmCoordinator(knowledge_isolation="strict")

# Moderate — knowledge at or below agent's hierarchy_level + 1
SwarmCoordinator(knowledge_isolation="moderate")

# Relaxed — all knowledge accessible
SwarmCoordinator(knowledge_isolation="relaxed")
```

## Serialization

```python
d = coordinator.to_dict()
# JSON-serializable dict with agents, trust matrix, and config
```

## Tests

```bash
python3 -m pytest tests/test_swarm_coordinator_bridge.py -v
```

26 tests covering agent registration, trust matrix, all 4 conflict resolution strategies, all 5 decomposition strategies, task assignment, knowledge isolation, serialization, ASCII visualization.

---

*Zero dependencies. Compatible with Equipment-Swarm-Coordinator patterns.*
