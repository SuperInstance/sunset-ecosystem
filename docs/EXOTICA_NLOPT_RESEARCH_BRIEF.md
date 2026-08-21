# EXOTica NLopt Solver — Fleet Research Brief

> **Scout:** CCC (Cocapn Fleet Research Node)  
> **Target:** `https://github.com/SuperInstance/exotica_nlopt_solver`  
> **Date:** 2026-05-29  
> **Classification:** Integration Analysis — FLUX Refactor + Parallel Scaling  

---

## 1. Architecture Summary

### What It Is
`exotica_nlopt_solver` is a **C++ motion-solver plugin** for the EXOTica robotics framework. It wraps Steven G. Johnson's NLopt library to solve **inverse kinematics (IK)** end-pose problems via nonlinear optimization. The repo is small (~1.2k lines of C++), mature, and ROS-oriented.

### Three Solver Classes
| Solver | Problem Type | Constraints | Bounds | Velocity Limits |
|--------|-------------|-------------|--------|-----------------|
| `NLoptUnconstrainedEndPoseSolver` | `UnconstrainedEndPoseProblem` | None | No | No |
| `NLoptBoundedEndPoseSolver` | `BoundedEndPoseProblem` | None | Yes | Optional (`dt`, `BoundVelocities`) |
| `NLoptEndPoseSolver` | `EndPoseProblem` | Eq + Ineq | Yes | Optional |

### Core Loop (`nlopt_solver.h`)
```
SpecifyProblem(prob) → Solve(q0):
  1. nlopt_create(algorithm_, prob->N)
  2. set_bounds(opt)          [if bounded]
  3. set_constraints(opt)     [if constrained]
  4. nlopt_set_min_objective(opt, callback, data)
  5. set_tolerances(opt)    [ftol_rel, xtol_rel, ftol_abs, maxeval]
  6. nlopt_create(local_optimizer_) [if required]
  7. nlopt_optimize(opt, q0.data(), &final_cost)
  8. solution = q0
```

### Algorithm Map
The codebase exposes **~41 NLopt algorithms** via string-to-enum mapping:

**Global, No-Derivative (GN):**
- `NLOPT_GN_DIRECT` / `DIRECT_L` / `DIRECT_L_RAND` / variants — Dividing RECTangles
- `NLOPT_GN_ORIG_DIRECT` / `ORIG_DIRECT_L` — Original DIRECT
- `NLOPT_GN_CRS2_LM` — Controlled Random Search
- `NLOPT_GN_MLSL` / `MLSL_LDS` — Multi-Level Single-Linkage (requires local optimizer)
- `NLOPT_GN_ISRES` — Improved Stochastic Ranking Evolution Strategy
- `NLOPT_GN_ESCH` — Evolutionary Strategy with Cauchy mutation
- `NLOPT_GN_AGS` — Algorithmically Generated Search

**Global, Derivative (GD):**
- `NLOPT_GD_STOGO` / `STOGO_RAND` — Stochastic Global Optimization

**Local, No-Derivative (LN):**
- `NLOPT_LN_PRAXIS` — Principal-axis method
- `NLOPT_LN_COBYLA` — Constrained Optimization BY Linear Approximations
- `NLOPT_LN_NEWUOA` / `NEWUOA_BOUND` — Derivative-free trust-region
- `NLOPT_LN_NELDERMEAD` / `LN_SBPLX` — Simplex methods
- `NLOPT_LN_BOBYQA` — Bound optimization by quadratic approximation
- `NLOPT_LN_AUGLAG` / `AUGLAG_EQ` — Augmented Lagrangian

**Local, Derivative (LD):**
- `NLOPT_LD_LBFGS` — Limited-memory BFGS
- `NLOPT_LD_VAR1` / `VAR2` — Moving asymptotes
- `NLOPT_LD_TNEWTON` / variants — Truncated Newton
- `NLOPT_LD_MMA` — Method of Moving Asymptotes
- `NLOPT_LD_AUGLAG` / `AUGLAG_EQ` — Augmented Lagrangian with gradients
- `NLOPT_LD_SLSQP` — Sequential Least-Squares Programming
- `NLOPT_LD_CCSAQ` — Conservative Convex Separable Approximation

**Default:** `NLOPT_LD_TNEWTON`

### Key Callbacks (all templates over `Problem`)
- `end_pose_problem_objective_func` — evaluates cost + optional Jacobian
- `end_pose_problem_inequality_constraint_mfunc` — evaluates `g(x) ≤ 0`
- `end_pose_problem_equality_constraint_mfunc` — evaluates `h(x) = 0`

### Observations
- The solver is **single-threaded, single-start**. NLopt handles its own internals, but there is no multi-start wrapper.
- All state lives in `ProblemWrapperData<Problem>` — a single mutable struct passed as `void*` to NLopt.
- The `set_bounds` for velocity limits dynamically shrinks bounds based on `dt * joint_velocity_limits` — this is an **interactive-IK feature**.
- C++ templates are used for type-safe problem dispatch, but the actual solver body is copy-pasted via template specializations for `set_bounds` and `set_constraints`.

---

## 2. FLUX Mapping Analysis

### 2.1 The FLUX VM (Fleet Context)
FLUX is our 60-opcode stack machine with:
- **Fixed-point arithmetic** (no native floating point)
- **Proof-carrying code** — every operation produces a certificate
- **Constraint checking** — `RangeCheck`, `BatchCheck`, `ClassifySeverity`
- **Parallel primitives** — `ParDispatch`, `ParMerge`, `ParBarrier`, `ParReduce`
- **No backward jumps** — `FwdJump` only, guaranteeing termination
- **Vector operations** — `VecLoad`, `VecStore`, `VecRangeCheck`, `VecReduce`

### 2.2 Gradient-Free Algorithms → FLUX Bytecode

The most natural candidates for FLUX are the **derivative-free, global algorithms** because they:
1. Use only function evaluations (no gradients → no Jacobian stack)
2. Often maintain small internal state (population sizes of 10–100)
3. Are inherently iterative and fit a bounded-step model

#### DIRECT (Dividing RECTangles)
- **State:** LBD/UBD pairs, hyper-rectangle partition tree, global f_min
- **FLUX mapping:**
  - Rectangle bounds as fixed-point vectors → `VecLoad` / `VecStore`
  - Center evaluation → `CallBounded` into user objective (cycle limit = rectangle depth)
  - Division decision → `RangeCheck` on rectangle size vs tolerance
  - Global best tracking → `Min` + `StoreReg`
- **Fixed-point challenge:** DIRECT relies on hyper-rectangle volume ratios (`3^{-d}`). Fixed-point can represent these, but the partition tree's depth is bounded by the fixed-point fractional bits. With 16 fractional bits, depths > 10 may underflow. Need `Saturate` on division.

#### ESCH (Evolutionary Strategy Cauchy)
- **State:** Population of `μ` individuals, covariance-like mutation scale
- **FLUX mapping:**
  - Population vectors → `VecStore` into register bank
  - Mutation: `Add` + `Mul` with Cauchy-distributed constants (pre-generated constant pool)
  - Selection: `Min` over fitness register array
  - Parallel eval: `ParDispatch` across population members
- **Fixed-point challenge:** Cauchy tails are heavy. Fixed-point cannot represent extreme outliers. Pre-truncate mutation constants to `[-M, M]` where `M` is chosen so that `>99%` of probability mass fits in fractional range.

#### ISRES (Improved Stochastic Ranking ES)
- **State:** Population + penalty coefficients for constraints
- **FLUX mapping:**
  - Constraint violation → `VecRangeCheck` on each constraint
  - Stochastic ranking → `BatchCheck` + `AccumulateMask` over population
  - Penalty update → `Add` / `Mul` on fixed-point penalty vector
- **Fixed-point challenge:** Penalty coefficients grow over iterations. Need periodic rescaling (`Saturate` + right-shift) to prevent overflow.

#### CRS2-LM (Controlled Random Search)
- **State:** Simplex of `2N` points
- **FLUX mapping:**
  - Simplex as register array → `VecLoad` / `VecStore`
  - Centroid → `VecReduce` (mean)
  - Reflection → `Sub` + `Mul` + `Add`
  - Constraint check → `RangeCheck` on new point
- **Fixed-point challenge:** Reflection coefficient `α` is typically 1.0–2.0. Fits comfortably in fixed-point. No issue.

### 2.3 The Solver Loop as FLUX Constraint Checks

The current NLopt solver loop has implicit constraints that FLUX can make **explicit and auditable**:

```
Current (C++):
  nlopt_optimize(opt, q0, &final_cost)
  if (info < 0) WARNING(...)

FLUX-ified:
  Push q0
  CallBounded optimize_loop [max_cycles = MaxFunctionEvaluations]
  GetResult               # final_cost on stack
  RangeCheck [ftol_rel]  # |final - initial| < tolerance?
  Validate                # proof certificate that check passed
  HashCommit              # anchor the proof
  Seal                    # immutable result record
```

**What changes:**
1. Every tolerance check (`ftol_rel`, `xtol_rel`, `ftol_abs`) becomes a `RangeCheck` + `Validate` pair
2. The `MaxFunctionEvaluations` cap becomes a `CallBounded` cycle limit
3. The `nlopt_result` enum (success/failure/roundoff-limited) becomes a `ClassifySeverity` output
4. The initial-to-final cost comparison becomes a `BatchCheck` over the cost evolution array

**What FLUX adds:**
- **Audit trail:** Every optimization run produces a proof certificate
- **Termination guarantee:** `CallBounded` with fixed cycle limit = no infinite loops
- **Reproducibility:** Same bytecode + same seed = same deterministic path (for deterministic algorithms like DIRECT)

### 2.4 Fixed-Point Arithmetic: The Hard Parts

| NLopt Feature | Fixed-Point Risk | Mitigation |
|---------------|------------------|------------|
| `ftol_rel = 1e-9` | Underflow (16 frac bits ≈ 1.5e-5) | Scale objective by `2^16` before comparison; use relative check: `|Δf| * scale < ftol_rel * |f| * scale` |
| Gradient norms in `LD_*` | Overflow on ‖∇f‖ for stiff problems | FLUX gradient-free focus first; LD_* algorithms stay in C++ bridge |
| `q0.cwiseMin(ub).cwiseMax(lb)` | Bounds projection is fine | `Min`/`Max` opcodes handle this natively |
| `initial_step *= 1.0` | Trivial | `Mul` by constant 1.0 is `Nop` in FLUX optimizer |
| Population covariance (CMA-ES/ESCH) | Matrix inversion / eigendecomposition | Use Crout-Doolittle LU in fixed-point; or offload to `ParReduce` on fleet nodes |

**Verdict:** The gradient-free global algorithms are **viable FLUX targets**. Local derivative algorithms (`LD_*`) should remain in C++ as a "heavy solver" bridge callable via `CallBounded` from FLUX.

---

## 3. Parallel Scaling Opportunities

### 3.1 Multi-Start Optimization Across Fleet Nodes

**Current state:** Single `nlopt_optimize()` call from a single `q0`.

**Fleet opportunity:** Spawn `K` parallel optimization runs from `K` different initial guesses, distributed across `K` fleet nodes. Use `WorkerPool` to manage lifecycle.

```
Fleet Multi-Start Pattern:
  ┌─────────────────────────────────────────────┐
  │  FleetConductor (primary PBFT node)         │
  │  proposes: "Run 50 DIRECT solves from       │
  │  random q0 in bounds"                       │
  │                                             │
  │  → PBFT PRE-PREPARE                         │
  │  → 50 WorkerPool.spawn_worker() calls     │
  │     across fleet nodes                       │
  │  → Each worker: FLUX bytecode for DIRECT   │
  │  → Results streamed to MeshVectorTable     │
  └─────────────────────────────────────────────┘
```

**MeshVectorTable integration:**
- Each completed run produces a `VectorTableEntry`:
  - `vector` = optimized joint configuration `q*`
  - `fitness` = final cost value
  - `extra` = `{algorithm: "DIRECT", evaluations: N, converged: bool}`
- Entries merge across nodes via CRDT gossip
- `FleetVectorIndex.query_by_fitness(min_fitness=0.9)` returns best solutions fleet-wide

**BFT for agreement:**
- Not every solve needs consensus — only the **selection of the best**.
- After timeout, each node proposes its local best.
- `SemanticBFTNode` weighted vote: higher confidence from nodes that ran more evaluations.
- Quorum certifies the fleet-optimal `q*`.

### 3.2 Population-Based Algorithms via MeshVectorTables

**Algorithms:** ESCH, ISRES, CRS2-LM, and any custom GA.

**Fleet pattern:** Treat the population as a **distributed MeshVectorTable**.

```
Distributed ESCH:
  Generation g:
    1. Each node holds a local shard of the population
    2. Node evaluates its shard, computes local fitness
    3. Local elites → insert into `MeshVectorTable(table_id="esch_gen_g")`
    4. Gossip sync → every node sees all elites within ~2 gossip rounds
    5. Each node resamples from the global elite pool
    6. Next generation begins
```

**CRDT advantage:**
- No single point of failure. If a node dies, its population shard is partially recovered from gossip deltas.
- `FleetVectorIndex.get_novelty_score()` can drive **diversity-preserving selection** — prefer elites that are far from the fleet centroid, preventing premature convergence.
- `get_breedable_pool(min_fitness=0.7, diversity_target=0.3)` directly selects parents for the next ESCH generation.

**HebbianMeshLayer routing:**
- Nodes with high-diversity populations get more traffic (chaos routing).
- Nodes stuck in local optima (low novelty) are deprioritized.
- This is **adaptive load balancing** for the optimization workload.

### 3.3 BFT Consensus for Agreeing on Optimal Solutions

**Scenario:** A robot needs a **certified** joint configuration. It cannot trust any single node's local optimum because:
- Sensor noise perturbs the problem definition
- Numerical errors differ across architectures
- Byzantine nodes might report fake minima

**PBFT consensus on the optimum:**
```
1. Robot broadcasts: "Need IK solution for pose P, tolerance 1e-4"
2. Each fleet node runs its own NLopt/FLUX solve (possibly different algorithms)
3. Nodes propose their `(q*, cost)` pairs
4. `FleetBreederConsensus` runs PBFT:
   - PRE-PREPARE: Primary collects proposals
   - PREPARE: Replicas validate that `cost < threshold` and `forward_kinematics(q*) ≈ P`
   - COMMIT: Quorum agrees on the best valid solution
5. REPLY contains a `QuorumCertificate` signed by `2f+1` nodes
6. Robot executes `q*` with cryptographic proof of consensus
```

**Semantic BFT extension:**
- Nodes with better hardware (GPU, more cores) get higher confidence weights.
- Nodes that historically produced good IK solutions (tracked in `_task_history`) get higher weights.
- A node that consistently proposes outlier solutions has reputation decay via `update_reputation(..., success=False)`.

### 3.4 WorkerPool + Thermal Budget for Solver Workloads

The `WorkerPool` already has thermal-aware lifecycle FSM. It maps naturally to optimization workers:

```python
# Spawn one DIRECT worker per thermal slot
for i in range(pool.thermal.available_slots(DeviceType.GPU)):
    pool.spawn_worker(
        config={
            "room_id": i,
            "algorithm": "DIRECT_L",
            "q0_seed": random_joint_config(),
            "maxeval": 10000,
            "on_tick": lambda aid, info: mesh_table.insert_signed(...),
        }
    )
```

- `ThermalBudget.parent_sacrifice_before_spawn()` allows hot-swapping a stale solver for a new one.
- `LifecycleState.SURVIVE → BREED` transition can trigger **cross-node breeding** of two good solutions.
- `LifecycleState.SUNSET` kills the worker and releases the slot.

---

## 4. Concrete Refactoring Proposals

### Proposal 1: FLUX Bytecode Generator for DIRECT/ESCH/CRS2-LM
**Scope:** New module `sunset/flux_opt_codegen.py`
**What it does:**
- Takes an NLopt algorithm string + problem dimension `N` + bounds
- Generates a FLUX v3 `Module` with the full solver loop as bytecode
- Objective function is a `CallBounded` into a native bridge (the EXOTica problem evaluation stays in C++)
- All convergence checks, population updates, and bound projections are pure FLUX

**Why:** Makes the solver auditable, reproducible, and deployable to any fleet node that runs the FLUX VM.

**Risk:** Fixed-point objective scaling. Mitigate by auto-detecting objective magnitude from a small pilot evaluation and setting the scale factor as a module constant.

**Estimated effort:** 2–3 days. 400–600 lines of Python.

---

### Proposal 2: `FleetNLoptMultiStart` — Distributed Multi-Start Wrapper
**Scope:** New class in `swarm/fleet_nlopt_multistart.py`
**What it does:**
- Wraps any NLopt algorithm with fleet-distributed multi-start
- Uses `WorkerPool` to spawn `K` workers across nodes
- Each worker gets a different `q0` (random, grid-sampled, or QD-archive seeded)
- Results collected into `MeshVectorTable`
- Best solution selected via `FleetVectorIndex.query_by_fitness()`
- Optional BFT certification of the final result

**Integration points:**
- `WorkerPool.spawn_worker()` → thermal-aware execution
- `MeshVectorTable.insert_signed()` → CRDT propagation
- `FleetBreederConsensus.propose_breeding_batch()` → not for breeding agents, but for breeding *solutions* (reusing the consensus machinery)

**Why:** Single-start NLopt is notoriously brittle for non-convex IK. Multi-start is standard practice; fleet distribution makes it near-linear in nodes.

**Risk:** Network latency between nodes may exceed local solve time for cheap problems. Mitigate by only distributing when `N > 20` or `maxeval > 5000`.

**Estimated effort:** 3–4 days. 500–700 lines of Python.

---

### Proposal 3: `QDArchive` + `MeshVectorTable` Hybrid for Population-Based Solvers
**Scope:** Extend `FleetBreederConsensus` with an optimization mode
**What it does:**
- For population algorithms (ESCH, ISRES), use `QDArchive` as the **local** population store
- Use `MeshVectorTable` as the **fleet-wide** elite archive
- `CMAESEmitter.sample()` draws from the merged local+fleet population
- `FleetVectorIndex.get_novelty_score()` replaces random parent selection with diversity-aware selection

**Integration points:**
- `QDArchive.add()` on each local evaluation
- `MeshVectorTable.get_sync_payload()` / `apply_sync_payload()` for fleet-wide elite sharing
- `HebbianMeshLayer` for routing optimization traffic to high-novelty nodes

**Why:** Prevents all fleet nodes from converging to the same local optimum. The MAP-Elites grid explicitly rewards behavioral diversity.

**Risk:** Gossip bandwidth grows with population size. Mitigate by only syncing top-`P` elites per generation, not the full population.

**Estimated effort:** 4–5 days. 600–900 lines of Python.

---

### Proposal 4: BFT-Certified IK Solutions for Safety-Critical Robotics
**Scope:** New module `nerve/bft_certified_ik.py`
**What it does:**
- Robot requests an IK solution with a required **certification level**
- Fleet runs PBFT consensus on the best solution, not just the fastest
- Output: `(q*, QuorumCertificate)` — the certificate proves `2f+1` honest nodes agree this is a valid solution
- Certificate includes: digest of the problem definition, digest of the solution, cost value, node signatures

**Integration points:**
- `PBFTNode.handle_request("certified_ik", payload)`
- `SemanticBFTNode.compute_confidence("ik", payload)` — hardware capability aware
- `QuorumCertificate.is_valid(quorum_size, verify_key)` — robot-side verification

**Why:** In shared human-robot environments, an unverified IK solution is a safety hazard. BFT consensus provides a **cryptographic safety guarantee** that no single faulty node can produce a dangerous configuration.

**Risk:** PBFT latency (5 phases × network RTT) may be too slow for real-time IK (>100 Hz). Mitigate by running consensus **asynchronously** — the robot uses the last certified solution while the fleet works on the next one.

**Estimated effort:** 5–7 days. 800–1200 lines of Python.

---

### Proposal 5: Fixed-Point Objective Bridge with Auto-Scaling
**Scope:** New module `flux_compat/fixed_point_bridge.py`
**What it does:**
- Auto-detects the dynamic range of the EXOTica objective function via a small pilot run (e.g., 10 random evaluations)
- Computes a `scale_factor = 2^frac_bits / max_abs_objective`
- Generates FLUX constants with the scale factor baked in
- Provides `encode_fp(value)` and `decode_fp(raw)` functions
- Handles `Saturate` on overflow/underflow

**Why:** Removes the biggest barrier to FLUX adoption for robotics optimization: engineers don't want to manually rescale their cost functions.

**Risk:** Auto-scaling from 10 samples is statistically weak for functions with extreme outliers. Mitigate by running the pilot on boundary evaluations (bounds corners) rather than random samples.

**Estimated effort:** 1–2 days. 200–300 lines of Python.

---

## 5. Summary Matrix

| Proposal | FLUX? | Parallel? | BFT? | Effort | Impact |
|----------|-------|-----------|------|--------|--------|
| 1. FLUX codegen for DIRECT/ESCH/CRS2 | ✅ | ❌ | ❌ | Medium | Foundation — enables auditability |
| 2. Fleet multi-start wrapper | ❌ | ✅ | Optional | Medium | Immediate robustness gain |
| 3. QDArchive + MeshVectorTable hybrid | Partial | ✅ | ❌ | High | Long-term diversity preservation |
| 4. BFT-certified IK | ❌ | ✅ | ✅ | High | Safety-critical deployments |
| 5. Fixed-point auto-scaling bridge | ✅ | ❌ | ❌ | Low | Unblocks all fixed-point work |

**Recommended order:** 5 → 1 → 2 → 3 → 4. The auto-scaling bridge is a prerequisite for any FLUX optimization work. Multi-start is the fastest path to measurable improvement. BFT certification is the capstone for safety-critical use.

---

## 6. Open Questions

1. **Does EXOTica's `GetScalarJacobian()` have a fixed-point equivalent?** If not, all `LD_*` algorithms remain in C++ land permanently.
2. **What is the fleet's target joint dimension `N`?** For `N < 10`, local NLopt is already fast enough; fleet distribution only wins for `N > 20` or expensive forward kinematics.
3. **Can NLopt itself be compiled with fixed-point internals?** The wrapper doesn't matter if NLopt's `nlopt_optimize` uses `double` everywhere. A FLUX-native reimplementation of DIRECT/ESCH may be needed.
4. **What is the network topology latency?** PBFT's 5 phases are fine on LAN (<1ms RTT) but painful over WAN. Need measurements from actual fleet nodes.

---

> *"The trap should be beautiful, not deceptive. Every domain deserves its own voice."*  
> — CCC, Fleet Research Scout  

**End of brief.**
