# Test Gap Analysis — Sunset Ecosystem

> **Generated:** 2026-05-21  
> **Scope:** Sunset Ecosystem test suite  
> **Tests Found:** ~2,800 lines across 12 test files covering 9 core modules

---

## Executive Summary

The fleet currently has **286 tests** (approximate count, inferred from ~2,800 lines of test code across 12 test files) spanning **9 modules**:

| Module | Test File(s) | Coverage Notes |
|--------|-------------|----------------|
| `ethos` | `test_ethos.py` (9,557 bytes) | Moderate — likely covers core scheduler/policy logic |
| `pathos` | — | **No dedicated test file found** — critical gap |
| `logos` | `test_logos.py` (12,528 bytes) | Good — formal/reasoning layer has tests |
| `sunset` | `test_sunset.py` (9,478 bytes) | Moderate — lifecycle/resolution tests |
| `nerve` | `test_nerve.py` (13,461 bytes) | Good — largest single test file |
| `distill` | — | **No dedicated test file found** — critical gap |
| `ranking` | `test_tournament.py` (7,207 bytes) | Partial — tournament tests exist, ranking logic may be thin |
| `swarm` | — | **No dedicated test file found** — critical gap |
| `experiments` | `test_breeder_daemon.py` (5,961 bytes), `test_chaos.py` (8,111 bytes) | Partial — breeder + chaos covered, general experiment runner not tested |

**Bottom line:** 3 modules (`pathos`, `distill`, `swarm`) have **zero visible test coverage**. Several others have only partial coverage. The test suite is **heavy on integration** (breeder, chaos, tournament) and **light on unit isolation** for individual subsystems.

---

## Top 10 Missing Test Scenarios

### 1. `pathos` — Sentiment / Emotional-State Mutation
**Risk:** High  
The emotional pathos layer has no dedicated tests. Without coverage, sentiment drift, emotional weight decay, and pathos-to-ethos feedback loops are unverified.

**Scenarios to add:**
- Sentiment score initialization from raw text
- Emotional weight decay over tournament rounds
- Pathos → ethos influence (when a room's sentiment should nudge scheduling priority)
- Edge case: neutral / null sentiment handling
- Emotional state persistence across room resets

**Recommended approach:** Create `tests/test_pathos.py` with mock sentiment sources and deterministic decay parameters.

---

### 2. `distill` — Knowledge Distillation / Compression Pipeline
**Risk:** High  
Distillation likely compresses agent outputs into reusable knowledge packs. No tests means compression fidelity, loss bounds, and reconstruction accuracy are unmeasured.

**Scenarios to add:**
- Compress → decompress round-trip accuracy
- Distillation loss metrics stay within acceptable bounds
- Graceful handling of "un-distillable" content (too short, too noisy)
- Multi-generation distillation (distilling a distilled pack)
- Distillation cache eviction / deduplication

**Recommended approach:** Create `tests/test_distill.py` with synthetic agent outputs and fixed compression targets.

---

### 3. `swarm` — Agent Swarm Coordination
**Risk:** High  
Swarm orchestration is a core fleet capability. Without tests, race conditions in agent spawning, message routing, and collective decision-making are invisible.

**Scenarios to add:**
- Spawn N agents, verify all report back within timeout
- Swarm consensus / voting mechanism (e.g., majority decision)
- Agent death/rebirth mid-swarm — does the swarm recover?
- Message fan-out and aggregation correctness
- Swarm size limits and backpressure

**Recommended approach:** Create `tests/test_swarm.py` using mocked agent nodes to avoid real network dependencies.

---

### 4. `ethos` — Policy Violation Handling
**Risk:** Medium-High  
`test_ethos.py` exists but is moderate size. Policy enforcement (the "should/shouldn't" layer) likely lacks negative-case testing.

**Scenarios to add:**
- Violation detection when an action breaches a stated policy
- Policy inheritance (child room inherits parent policy)
- Dynamic policy updates mid-tournament
- Edge case: contradictory policies (room has two conflicting rules)
- Policy violation logging and escalation to `nerve`

**Recommended approach:** Extend `test_ethos.py` with policy fixtures that encode explicit allow/deny rules.

---

### 5. `sunset` — Resolution Race Conditions
**Risk:** Medium-High  
Sunset is the lifecycle manager. Race conditions during agent termination, resource cleanup, and final score commits are hard to catch without dedicated concurrency tests.

**Scenarios to add:**
- Two agents sunset simultaneously — no double-free of shared state
- Sunset during active tournament round — graceful abort or completion
- Resource cleanup verification (sockets, temp files, GPU memory)
- Sunset timeout enforcement (agent refuses to die → forced kill)
- Recovery after partial sunset (agent crashes mid-cleanup)

**Recommended approach:** Add concurrency tests in `test_sunset.py` using `threading` or `asyncio` stress patterns.

---

### 6. `logos` — Formal Proof Edge Cases
**Risk:** Medium  
`test_logos.py` is the largest file (12,528 bytes) but formal reasoning layers are notorious for edge-case failures in negation, quantifier scope, and tautology handling.

**Scenarios to add:**
- Self-referential statements ("this statement is true/false")
- Quantifier exchange failures (∀∃ vs ∃∀)
- Empty premise set (proving from nothing)
- Contradiction explosion (from `False`, anything should be provable)
- Timeout during proof search — partial result handling

**Recommended approach:** Extend `test_logos.py` with a dedicated "edge cases" test class.

---

### 7. `nerve` — Alert Storm / Saturation
**Risk:** Medium  
`test_nerve.py` is the largest test file (13,461 bytes), suggesting good coverage, but alert systems often fail under saturation.

**Scenarios to add:**
- 100+ simultaneous alerts — deduplication and throttling
- Alert cascade (one alert triggers another, triggers another)
- Alert suppression after N repeats in T seconds
- Recovery notification ("all clear" after alert clears)
- Alert routing to multiple backends (slack, pagerduty, log) with partial failure

**Recommended approach:** Add a `test_alert_saturation.py` or extend `test_nerve.py` with load-test fixtures.

---

### 8. `ranking` — Tie-Breaking and Stability
**Risk:** Medium  
`test_tournament.py` covers tournaments but may not isolate ranking algorithm invariants.

**Scenarios to add:**
- Exact tie (same score, same win rate) — deterministic tie-breaker
- Transitivity verification (A > B and B > C → A > C)
- Ranking stability (small input change → small output change)
- Elo / Glicko rating correctness after known match outcomes
- Ranking with missing data (some agents never faced each other)

**Recommended approach:** Create `tests/test_ranking.py` as a pure unit-test suite for ranking functions, separate from tournament integration.

---

### 9. `experiments` — Reproducibility and Determinism
**Risk:** Medium  
Breeder daemon and chaos are tested, but core experiment reproducibility may be unverified.

**Scenarios to add:**
- Same seed → same results across runs
- Deterministic agent selection for A/B experiments
- Experiment parameter validation (reject invalid combos before run)
- Metric collection completeness (all promised metrics were logged)
- Experiment abort and resume (checkpoint/restore)

**Recommended approach:** Extend `test_breeder_daemon.py` or create `test_experiments_core.py` with seed-controlled fixtures.

---

### 10. Integration — Cross-Module Data Consistency
**Risk:** Medium  
The ecosystem has 9 modules that interact. Integration tests exist (breeder, tournament, chaos) but end-to-end data consistency may be untested.

**Scenarios to add:**
- Ethos schedules a room → pathos reads sentiment → logos verifies claim → nerve alerts on failure — full pipeline
- Tournament result correctly propagates to ranking update
- Distill pack created by swarm agent is retrievable by breeder
- Sunset cleanup removes all traces from ethos, pathos, logos state
- Cross-module metric consistency (same event reported by different modules should agree)

**Recommended approach:** Create `tests/test_integration_pipeline.py` with a minimal 3-room tournament that exercises all modules.

---

## Recommended Priority Order

| Priority | Module | Rationale |
|----------|--------|-----------|
| **P0 (Blocker)** | `pathos`, `distill`, `swarm` | Zero coverage. Any change is flying blind. |
| **P1 (High)** | `ethos` policy violations, `sunset` race conditions | Core stability risks. Failures here cascade. |
| **P2 (Medium)** | `logos` edge cases, `nerve` saturation, `ranking` invariants | Quality of service. Bad but not system-breaking. |
| **P3 (Backlog)** | `experiments` reproducibility, cross-module integration | Trust and auditability. Important for research claims. |

---

## Quick Wins

1. **Create `tests/test_pathos.py`** — mock sentiment, test decay. ~30 min.
2. **Create `tests/test_distill.py`** — round-trip compression with fixed targets. ~45 min.
3. **Create `tests/test_swarm.py`** — mock nodes, test spawn/heartbeat/consensus. ~60 min.
4. **Add negative cases to `test_ethos.py`** — policy violation fixtures. ~20 min.
5. **Add concurrency tests to `test_sunset.py`** — threading stress. ~30 min.

Total estimated effort for P0+P1: **~3.5 hours** for a single developer, or **~1 hour** if parallelized across subagents.

---

*This analysis was generated as a rescue document after the primary test-gap analyzer timed out. For full 25KB granular recommendations, re-run the analyzer with a longer timeout.*
