# SPEC-TUTOR.md — Make the Tutor Repos Actually Teach

## Problem

There are 4 tutor-adjacent repos in the ecosystem, but none have structured learning paths:

1. **plato-core** — shipped on PyPI, but no lesson directory. Users get the library with no guidance.
2. **eisenstein-embed** — shipped on PyPI, bitvector embeddings. No progressive exercises.
3. **deadband-rs** — Rust crate for deadband/BMA algorithms. Reference code exists (`src/bma.rs`) but no onboarding path.
4. **sunset-ecosystem** — the 360-test monolith. Has rooms, tournament, thermal, penrose — but no "lesson 1: spawn your first room."

Each repo should have a `lesson/` directory with progressive exercises. New agents (from the breeder) should onboard by cloning a tutor → running the test suite → graduating to real rooms.

## Ground-Level Code

### Lesson format specification

Each lesson is a directory under `lesson/`:

```
lesson/
  01-spawn-your-first-room/
    exercise.md          # Instructions, context, what to learn
    solution/
      solution.py        # Reference solution (hidden until needed)
    test.py              # pytest test — agent must pass this
    hints.md             # Progressive hints (spoiler-tagged)
  02-fire-a-signal/
    exercise.md
    solution/
      solution.py
    test.py
    hints.md
  03-tournament-basics/
    ...
```

### exercise.md template

```markdown
# Lesson {N}: {Title}

## What You'll Learn
- {learning objective 1}
- {learning objective 2}

## Context
{2-3 paragraphs explaining WHY this matters in the ecosystem}

## Your Task
{concrete, testable instruction}

## Files to Read
- `{file1}` — {what to look for}
- `{file2}` — {what to understand}

## Success Criteria
- `pytest lesson/{nn}-{slug}/test.py` passes

## Hints
See `hints.md` if stuck. Try without hints first.
```

### test.py template

```python
"""Lesson {N}: {Title} — Test Suite

Run: pytest lesson/{nn}-{slug}/test.py -v
"""
import pytest

def test_spawn_room():
    """Agent must create a JEPAGrid and verify a room exists."""
    # User writes code that creates a grid and exposes it
    from solution.solution import grid  # noqa: F401
    assert grid is not None
    assert grid.n > 0

def test_room_weights():
    """Room weights should be initialized (not all zeros)."""
    from solution.solution import grid
    assert grid.w["w1"].any(), "Room weights should be initialized"

def test_room_perceives():
    """Room should process a signal and produce a latent."""
    import numpy as np
    from solution.solution import grid
    x = np.random.randn(64).astype(np.float32)
    result = grid.tick(x)
    assert result["fired"] >= 0, "Grid should return tick results"
```

### Lesson plans per repo

**sunset-ecosystem** (7 lessons):

| # | Slug | What | Tests |
|---|------|------|-------|
| 01 | spawn-your-first-room | `JEPAGrid(10)`, check `.n`, `.w` | grid exists, has weights |
| 02 | fire-a-signal | `grid.tick(randn(64))`, check fired count | tick returns valid dict |
| 03 | find-active-rooms | `grid.top(5)`, `grid.cold()` | top rooms sorted, cold detected |
| 04 | run-a-tournament | `TournamentRound(scores).run()`, check ranks | tournament completes, ranks valid |
| 05 | breed-children | `breed(winners, 3)`, check child config | children have valid scores |
| 06 | thermal-budget | `ThermalBudget()`, allocate/release | budget respected, release works |
| 07 | full-lifecycle | Spawn → tick 100× → tournament → breed → rebirth cold | population churns, generation advances |

**plato-core** (5 lessons):

| # | Slug | What | Tests |
|---|------|------|-------|
| 01 | create-a-room | SoftRoom creation, basic API | room admits input |
| 02 | snap-a-tile | Snap threshold, tile compiles | tile crosses threshold |
| 03 | hard-room-cannot-be-fooled | HardRoom verification | fails invalid input |
| 04 | trinity-scoring | Ethos × Pathos × Logos | product computed correctly |
| 05 | sunset-a-room | Lifecycle from soft to hard to sunset | full cycle completes |

**eisenstein-embed** (4 lessons):

| # | Slug | What | Tests |
|---|------|------|-------|
| 01 | embed-a-string | Basic bitvector embedding | embedding has correct dim |
| 02 | compare-embeddings | Cosine similarity between embeddings | similarity in [-1, 1] |
| 03 | batch-embed | Embed multiple strings efficiently | batch matches individual |
| 04 | fingerprint-matching | Use embeddings as room fingerprints | matching finds correct room |

**deadband-rs** (4 lessons):

| # | Slug | What | Tests |
|---|------|------|-------|
| 01 | basic-deadband | Apply deadband to a signal stream | noise filtered |
| 02 | berlekamp-massey | Run BMA on binary sequence (see `src/bma.rs`) | correct LFSR found |
| 03 | adaptive-deadband | Adaptive threshold based on signal variance | threshold adjusts |
| 04 | integrate-with-grid | Deadband as pre-processing for JEPA input | signal-to-noise improves |

### Agent onboarding test

New file: `sunset-ecosystem/lesson/onboarding_test.py`

```python
"""Agent Onboarding Test — Must pass before agent enters real rooms.

This is the graduation test for new agents spawned by the BreedingDaemon.
An agent must:
1. Clone the tutor lessons
2. Run the test suite
3. Graduate to real rooms

Usage:
    pytest lesson/onboarding_test.py -v
"""
import subprocess
import sys
import pytest


def test_01_spawn_room():
    """Did you successfully spawn a room?"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "lesson/01-spawn-your-first-room/test.py", "-v"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lesson 01 failed:\n{result.stdout}\n{result.stderr}"


def test_02_fire_signal():
    """Did it fire?"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "lesson/02-fire-a-signal/test.py", "-v"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lesson 02 failed:\n{result.stdout}\n{result.stderr}"


def test_03_active_rooms():
    """Can you find active rooms?"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "lesson/03-find-active-rooms/test.py", "-v"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lesson 03 failed:\n{result.stdout}\n{result.stderr}"


def test_06_full_lifecycle():
    """Full lifecycle: spawn → tick → tournament → breed → rebirth."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "lesson/07-full-lifecycle/test.py", "-v"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lesson 07 failed:\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="session")
def onboarding_pass():
    """Marker for the breeder: this agent passed onboarding."""
    return True
```

### Breeder integration

Add to `swarm/breeder.py` (from SPEC-BREEDER):

```python
def onboard_agent(self, record: AgentRecord) -> bool:
    """Run onboarding test suite for a new agent.

    Returns True if agent graduates to real rooms.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "lesson/onboarding_test.py", "-v", "--tb=short"],
        capture_output=True, text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    if passed:
        log.info(f"Agent {record.agent_id} graduated onboarding")
    else:
        log.warning(f"Agent {record.agent_id} failed onboarding:\n{result.stdout}")
    return passed
```

## Decision

Standardize on `lesson/` directories with a fixed format: `exercise.md`, `solution/solution.py`, `test.py`, `hints.md`. The onboarding test is a pytest suite that runs all lesson tests as subprocess calls. The breeder calls `onboard_agent()` before installing a child into a real room.

## Implementation Order

1. Create lesson directory structure in `sunset-ecosystem/lesson/`
2. Write lessons 01-03 (spawn, fire, active rooms) — these are the critical path
3. Write `onboarding_test.py`
4. Write lessons 04-07 (tournament, breed, thermal, lifecycle)
5. Add `lesson/` to `plato-core` (5 lessons)
6. Add `lesson/` to `eisenstein-embed` (4 lessons)
7. Add `lesson/` to `deadband-rs` (4 lessons, Rust tests)
8. Integrate `onboard_agent()` into BreedingDaemon
9. Add CI check: `pytest lesson/onboarding_test.py` must pass

## Success Criteria

- [ ] Each repo has `lesson/` with numbered progressive exercises
- [ ] Every lesson has `exercise.md`, `solution/`, `test.py`, `hints.md`
- [ ] `pytest lesson/onboarding_test.py` passes (4 tests)
- [ ] New agents from breeder run onboarding before entering real rooms
- [ ] Lesson 01 test is: "spawn a room, verify it has weights"
- [ ] Lesson 02 test is: "fire a signal, verify it fired"
- [ ] Total: 20 lessons across 4 repos
- [ ] CI runs onboarding test on every commit
