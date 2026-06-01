# Fleet Task Board Bridge

*Integration target: `oracle1-vessel`*

Brings the TASK-BOARD.md pattern into sunset-ecosystem.

## What It Does

- **Priority levels** — CRITICAL 🔴, HIGH 🟠, MEDIUM 🟡, LOW 🟢
- **Capability tags** — `[c]`, `[python]`, `[rust]`, `[testing]`, `[infra]`, etc.
- **Owner assignment** — who owns what
- **T-minus estimates** — `T-24h`, `T-0h` for task ETA
- **Dependency tracking** — `blocked_by` field for task chains
- **Critical path** — auto-filter CRITICAL tasks that aren't done
- **Ready tasks** — tasks that are not blocked and not done
- **Org chart rendering** — visualize fleet hierarchy from owner assignments

## Quick Start

```python
from fleet.fleet_task_board_bridge import FleetTaskBoard, TaskPriority

board = FleetTaskBoard()

# Add tasks with priority and tags
board.add_task("Conformance", TaskPriority.CRITICAL, ["c", "python"], owner="JC1")
board.add_task("Dashboard", TaskPriority.HIGH, ["infra"], owner="Oracle1")
board.add_task("Docs", TaskPriority.LOW, ["docs"])

# Set ETA
board.set_eta("task-1", "T-24h")

# Claim and complete
board.claim_task("task-1", "JC1")
board.complete_task("task-1", commit_hash="abc123")

# Render the board
print(board.render_text())
```

## Task Lifecycle

```
OPEN → CLAIMED → DONE
    ↑
  BLOCKED (blocked_by set)
```

## Priority Rendering

```
🔮 FLUX Fleet Task Board
==================================================
🔴 ✓ Conformance @JC1 T-24h
   [c] [python]
   Commit: abc123
🟠 ► Dashboard @Oracle1
   [infra]
🟢   Docs
   [docs]
```

## Org Chart

```python
print(board.render_org_chart())
# → Captain Casey
# →   └── Oracle1 🔮 (Managing Director)
# →       ├── JC1 — 2 active
# →       ├── OpenManus — 1 active
```

## Tests

```bash
python3 -m pytest tests/test_fleet_task_board_bridge.py -v
```

20 tests covering add, claim, complete, ETA, priority sorting, owner filtering, critical path, ready tasks, blocking, rendering, and serialization.

---

*Zero dependencies. Compatible with oracle1-vessel TASK-BOARD.md patterns.*
