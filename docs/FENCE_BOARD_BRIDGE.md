# Fence Board Bridge

*Integration target: `oracle1-vessel`*

Brings the Tom Sawyer Protocol / Fence Board pattern into sunset-ecosystem.

## The Pattern

> *"Work so good they'll fight to do it."* — Tom Sawyer Protocol

Tasks are **fences** (puzzles). Agents are **challengers** (competitors). The board tracks:
- **Open fences** — anyone can claim
- **Claimed fences** — someone is working on it
- **Completed fences** — done, with badges

## Quick Start

```python
from fleet.fence_board_bridge import FenceBoard

board = FenceBoard(max_active=5)

# Post a fence (task as puzzle)
fence = board.post_fence(
    title="Map 16 Viewpoint Opcodes",
    brush="16 ops are reserved but undefined. Map them to FLUX Format E.",
    view="Your name on 16 opcodes that every FLUX runtime executes.",
    challengers={
        "Babel": (3, "Built the concept. This is his baby."),
        "Oracle1": (7, "Good at specs, semantically shallow."),
    },
    reward="0x70-0x7F permanently attributed to you",
    claim_window_hours=48,
)

# Claim it
board.claim_fence(fence.id, "Oracle1", approach="Build FORMAT_E encoder")

# Complete it
board.complete_fence(fence.id, ["src/flux/viewpoint_ops.py"], badge="🥇 Gold")

# See the board
print(board.render_board())
```

## Fence Lifecycle

```
POST → OPEN → CLAIM → WORK → COMPLETE
```

Max 5 active fences at a time. Complete or claim one to post another.

## Challenger Difficulty

Difficulty is 1–10 (lower = easier for that agent). The best challenger is the one with lowest difficulty.

```python
best = board.best_challenger("fence-0x42")
# → "Babel" (difficulty 3 vs Oracle1's 7)
```

## Badges

- 🥇 Gold — exceptional work
- 🥈 Silver — solid contribution
- 🥉 Bronze — completed

## Serialization

```python
d = board.to_dict()
# JSON-serializable with all fences, challengers, status
```

## Tests

```bash
python3 -m pytest tests/test_fence_board_bridge.py -v
```

16 tests covering post, claim, complete, max_active, best challenger, board rendering, serialization.

---

*Zero dependencies. Compatible with oracle1-vessel Fence Board patterns.*
