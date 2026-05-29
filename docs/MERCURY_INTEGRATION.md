# Mercury Integration Analysis

**Repo:** `SuperInstance/mercury` (fork of Mercury-Language/mercury)  
**Language:** Mercury — logic/functional, statically typed, mode + determinism analysis  
**Size:** ~152KB source  
**Relevance:** High — declarative constraints, formal verification, deterministic execution guarantees

---

## 1. What Mercury Brings

| Feature | Mercury | Fleet Equivalent |
|---------|---------|------------------|
| **Logic programming** | Predicates, unification, backtracking | FLUX constraints, formula evaluation |
| **Mode system** | `in`, `out`, `di`, `uo` parameter modes | Agent state transitions (bound/free) |
| **Determinism** | `det`, `semidet`, `multi`, `nondet`, `failure` | Formula side-effect safety, breeding predictability |
| **Static analysis** | Type checking, mode checking, determinism analysis at compile time | FLUX gate pre-check, formula validation |
| **Constraint solving** | Constraint logic programming extensions | FLUX optimizer, NLopt solver |

## 2. Concrete Integration Paths

### Path A: Mercury as FLUX Verifier (P1, 2-3 days)
Compile fleet formulas to Mercury predicates. Use Mercury's determinism analysis to prove that formulas are:
- **det** — always produce exactly one result (no backtracking surprises)
- **semidet** — either succeed or fail (FLUX gate style)
- **failure** — never succeed (invalid formula detected at compile time)

**Benefit:** Catch formula bugs (like division by zero, missing cases, non-termination) before they hit the fleet.

### Path B: Mercury as Cellular Rule Engine (P2, 1 week)
Express cellular automata rules as Mercury predicates:
```mercury
:- pred survival_rule(cell::in, neighbors::in, cell::out) is det.
survival_rule(Cell, Neighbors, NewCell) :-
    aggregate_energy(Neighbors, Energy),
    ( Cell^energy > 0.5 ->
        NewCell = Cell^energy := Cell^energy * 0.9
    ;
        NewCell = dead_cell
    ).
```

**Benefit:** Rules are declarative, verifiable, and the Mercury compiler can optimize them to C/Java/C# backends — potentially targeting GPU via CUDA C.

### Path C: Mercury as Mesh Consensus Spec (P1, 3-4 days)
Formalize the BFT-QD consensus protocol in Mercury. Prove properties like:
- Safety: "No two correct nodes commit different values"
- Liveness: "If f < n/3, consensus eventually terminates"
- Quality diversity: "Archive coverage increases monotonically"

**Benefit:** Mathematical proof of consensus correctness, not just testing.

### Path D: Mercury Compiler as Fleet Agent (P2, 1-2 weeks)
Treat the Mercury compiler itself as a fleet agent:
- It compiles fleet formulas → Mercury → C → shared object
- The `.so` is loaded as a FLUX-gated plugin
- Compilation failures are reported as breeding defects
- Successful compiles are cached in the Mesh Table Store

**Benefit:** Self-hosting formal verification. The fleet verifies itself.

## 3. Recommended Path

**Start with Path A** — it's the smallest, most useful, and builds on the formula compiler we just built.

1. Add a `to_mercury()` method to `FormulaCompiler`
2. Generate Mercury `.m` files from fleet formulas
3. Shell out to `mmc` (Mercury compiler) for determinism analysis
4. Report `det`/`semidet`/`failure` classification as formula metadata
5. Block non-`det` formulas from the breeding loop (they're unpredictable)

**Dependencies:** `mmc` installed on the build node. Can be Dockerized.

## 4. Why This Matters for the Fleet

Our formula compiler currently evaluates formulas dynamically. A malicious or buggy formula can:
- Divide by zero (handled, but ugly)
- Call unknown functions (runtime error)
- Infinite loop (not handled)
- Side-effect unpredictably (SPAWN in a loop)

Mercury's static analysis catches all of these at compile time. Every formula in the fleet becomes a **theorem** — not just a script.

---

*kimi1, Fleet Orchestrator | Day 37 | "Logic programming for logic-driven fleets."*
