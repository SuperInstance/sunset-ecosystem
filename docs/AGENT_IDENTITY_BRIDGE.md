# Agent Identity Bridge

*Integration target: `git-agent-standard`*

Brings the git-agent-standard repo structure into sunset-ecosystem as a Python API for agent identity management, bottle communication, and diary tracking.

## What It Does

- Agent identity as repo structure (`CHARTER.md`, `STATE.md`, `TASK-BOARD.md`, `SKILLS.md`)
- Bottle communication (`for-fleet/` → outgoing, `from-fleet/` → incoming)
- Abstraction planes (capability declaration: what the agent can read/write)
- Diary entries (`DIARY/YYYY-MM-DD.md`)
- Full save/load roundtrip

## Quick Start

```python
from fleet.agent_identity_bridge import AgentVessel

# Create a new agent
vessel = AgentVessel.create("/path/to/my-agent", "Scout7", "Explore new repos")

# Check state
print(vessel.charter.name)  # Scout7
print(vessel.charter.purpose)  # Explore new repos
print(vessel.state.health)  # 🟢 ACTIVE

# Write a bottle to another agent
bottle = vessel.write_bottle(
    to="oracle1", content="Found a pattern in constraint-theory-core KD-tree..."
)

# Read incoming bottles
for bottle in vessel.read_bottles():
    print(f"From {bottle.from_agent}: {bottle.content}")

# Write diary
vessel.write_diary(date="2026-06-01", content="Today I built a bridge")

# Get summary
print(vessel.summary())
```

## Repo Structure

```
my-agent/
├── CHARTER.md           # Purpose, contracts, constraints
├── STATE.md             # Health, current task, pending, blockers
├── TASK-BOARD.md        # Critical / High / Medium / Done tasks
├── SKILLS.md            # Core skills, tools, learned lessons
├── ABSTRACTION.md       # Capability planes (JSON)
├── DIARY/
│   └── 2026-06-01.md    # Dated entries
├── for-fleet/
│   └── BOTTLE-TO-oracle1-2026-06-01.md
└── from-fleet/
    └── MESSAGE-FROM-ccc-2026-06-01.md
```

## Charter

```python
vessel.charter.name = "Scout7"
vessel.charter.purpose = "Explore new repos"
vessel.charter.contracts = ["Report findings", "Don't modify production"]
vessel.charter.constraints = ["No direct commits to main"]
vessel.save()
```

## State

```python
vessel.state.health = "🟢 ACTIVE"
vessel.state.current_task = "Mining patterns"
vessel.state.pending = 3
vessel.state.blockers = ["Waiting for API key"]
vessel.save()
```

## Abstraction Planes

Abstraction planes declare what an agent can read and write:

```python
from fleet.agent_identity_bridge import AbstractionPlane

plane = AbstractionPlane(
    primary=4,  # This agent operates at plane 4
    reads_from=[3, 4, 5],  # Can read planes 3-5
    writes_to=[2, 3, 4],  # Can write planes 2-4
    floor=2,  # Lowest accessible plane
    ceiling=5,  # Highest accessible plane
)

assert plane.can_read(4) is True
assert plane.can_read(6) is False
assert plane.can_write(3) is True
```

## Bottles

Bottles are async messages between agents. They auto-generate filenames:

```python
bottle = vessel.write_bottle(to="oracle1", content="Hello")
print(bottle.filename())  # BOTTLE-TO-oracle1-2026-06-01.md
```

## Task Board

```python
vessel.task_board.critical = ["Fix test flakiness"]
vessel.task_board.high = ["Add new bridge"]
vessel.task_board.medium = ["Refactor old code"]
vessel.task_board.done = ["Initial setup"]
vessel.save()
```

## Skills

```python
from fleet.agent_identity_bridge import SkillEntry

vessel.skills.core_skills = [
    SkillEntry(
        name="Pattern Mining", description="Extract reusable patterns from repos"
    ),
    SkillEntry(
        name="Bridge Building", description="Connect sunset-ecosystem to external repos"
    ),
]
vessel.skills.tools = ["Git", "Pytest", "kimi_search"]
vessel.skills.learned = ["Always test first", "Push often"]
vessel.save()
```

## Loading an Existing Agent

```python
vessel = AgentVessel.from_repo("/path/to/existing-agent")
print(vessel.charter.name)
print(vessel.state.health)
```

## Summary

```python
summary = vessel.summary()
# {
#   "name": "Scout7",
#   "purpose": "Explore new repos",
#   "health": "🟢 ACTIVE",
#   "current_task": "Mining patterns",
#   "pending": 3,
#   "blockers": ["Waiting for API key"],
#   "primary_plane": 4,
#   "bottles_out": 1,
#   "bottles_in": 0,
#   "diary_entries": 1,
# }
```

## API Reference

### `AgentVessel(repo_path)`

Load an agent vessel from a repo path. Creates default identity files if missing.

### `AgentVessel.create(repo_path, name, purpose)`

Class method. Create a new agent with default identity.

### `AgentVessel.from_repo(repo_path)`

Class method. Load an existing agent.

### `save()`

Write all identity files to disk.

### `write_bottle(to, content, **metadata)`

Write an outgoing bottle to `for-fleet/`. Returns `Bottle`.

### `read_bottles()`

Read incoming bottles from `from-fleet/`. Returns list of `Bottle`.

### `write_diary(date=None, content="")`

Write a diary entry to `DIARY/`. Returns `Path`.

### `read_diary()`

Read all diary entries. Returns list of `(date, content)` tuples.

### `summary()`

Return JSON-serializable summary dict.

## Tests

```bash
python3 -m pytest tests/test_agent_identity_bridge.py -v
```

25 tests covering charter/state/task/skills parsing, bottle read/write, diary, summary, abstraction planes.

---

*Zero dependencies. Compatible with git-agent-standard repo structure.*
