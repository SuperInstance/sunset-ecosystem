# Round 8 Research Note: CRDT Breeding Merge + HDC Diversity Combination

> **Date:** 2026-05-24  
> **Scope:** `swarm/crdt_merge.py` + `swarm/hdc_novelty.py`  
> **Author:** CCC (subagent round 8)  

---

## 1. How CRDT Merge Works for Breeding

The `CRDTMergeEngine` resolves divergent agent populations after network partition using a **Last-Write-Wins (LWW)** strategy layered on top of fitness-driven conflict resolution.

### Core Conflict Resolution Rules

When two ships have bred independently and reconnect, the engine processes every agent ID found in either population:

| Scenario | Resolution |
|----------|------------|
| Agent exists only on local | Keep as-is |
| Agent exists only on remote | Accept after **lineage sanity check** |
| Agent exists on both | `resolve_conflict()` → higher fitness wins; tie broken by `last_updated` timestamp |
| Both copies have different valid lineages | Merge lineages into `all_parents` |

### The `all_parents` Field

The `Agent` dataclass carries both `parent_a`/`parent_b` (the immediate breeding pair) and `all_parents` (the complete ancestral set). This dual structure serves two purposes:

- **`parent_a` / `parent_b`** — Quick lookup of immediate breeding parents.
- **`all_parents`** — A **deduplicated merged list** used when two ships independently converged on the same agent ID with different parent pairings. `_merge_lineages()` preserves local parents first, then appends remote parents not already present.

This is the CRDT "merge function" for lineage: associative, commutative, and idempotent over the parent set.

### Vector Table Synchronization

`sync_vector_table()` merges two `FluxVectorTable` instances by comparing timestamps stored in `AgentVector.extra["last_updated"]`. The winner's vector, fitness, generation, and capability mask are copied into a fresh table. This means **timestamps are the ultimate tie-breaker** even for vector embeddings — not Hamming distance, not novelty score.

### Lineage Sanity Gates

Before accepting a remote-only agent, the engine verifies:

1. **Fitness in [0, 1]** — hard bounds on the `Agent` constructor.
2. **Generation boundedness** — an agent's generation cannot exceed `max(parent generations) + 3`. This catches impossible evolutionary jumps (e.g., a generation-50 agent allegedly bred from generation-0 seeds).
3. **Parent existence** — claimed parents must exist in at least one of the two populations.
4. **Seed rule** — an agent with no parents must be generation 0.

Agents failing any gate are **rejected** (logged, skipped, not merged). There is no quarantine or repair path — the ship simply never sees that lineage.

---

## 2. How HDC Novelty Measures Diversity

The `HDCDiversityScorer` implements **Hyperdimensional Computing (HDC)** binary novelty: a lightweight, hardware-accelerated alternative to cosine distance.

### Binary Encoding

`BinaryVectorEncoder` converts a float32 vector to a packed binary hypervector via **sign-thresholding**:

```
element > 0 → 1
element ≤ 0 → 0
```

Bits are packed into the smallest unsigned integer type that fits the dimension (`uint8` → `uint16` → `uint32` → `uint64`, or arrays of `uint64` for dims > 64). Packing uses LSB-first layout to match the C AVX-512 reference (`flux_hdc_avx512.h`).

### XOR + POPCOUNT Hamming Distance

Diversity between two packed vectors is computed as:

1. **XOR** the two packed word arrays.
2. **POPCOUNT** every word (count set bits).
3. **Sum** across words → raw Hamming distance.
4. **Normalize** by `dim` → score in `[0, 1]`.

```
score = hamming_distance / dim
0.0 = identical vectors
1.0 = orthogonal vectors (every bit differs)
```

### AVX-512 Fast Path

When `HAS_AVX512` is true (detected via `/proc/cpuinfo` + `np.bitwise_count` benchmark), the POPCOUNT step dispatches to **VPOPCNTDQ** on `uint64` arrays. The runtime probe runs a micro-benchmark: if NumPy's `bitwise_count` is not at least 2× faster than a Python `int.bit_count()` loop, AVX-512 is declared present-but-unusable and the fallback is used.

### Batch Scoring

`score_batch()` handles `(n_q, n_words) × (n_r, n_words)` matrices via NumPy broadcast XOR:

```python
xor = np.bitwise_xor(q[:, None, :], r[None, :, :])  # (n_q, n_r, n_words)
counts = np.bitwise_count(xor).sum(axis=2)  # (n_q, n_r)
```

Benchmarking against cosine distance shows **~100× speedup**, making HDC viable for real-time diversity filtering during breeding loops.

---

## 3. How They Could Combine

The two systems operate on **different scopes** that complement each other:

| Concern | CRDT Merge | HDC Diversity |
|---------|-----------|---------------|
| **Scale** | Fleet-wide (multi-ship) | Per-ship (local population) |
| **Goal** | Consensus — "what is the canonical agent?" | Exploration — "is this agent different enough?" |
| **Metric** | Fitness + Timestamp | Hamming distance |
| **Decision** | Keep / Reject / Merge lineages | Score novelty for selection pressure |

### Proposed Integration Architecture

**CRDT handles multi-ship consensus; HDC handles local diversity.**

When two ships reconnect:

1. **CRDT layer** merges populations — resolves conflicts, validates lineages, produces a unified agent set.
2. **HDC layer** scores the *diversity contribution* of every merged agent against the ship's current local population.
3. **Combined decision**: instead of blindly accepting every merged agent, a ship could **filter merged agents by novelty threshold** — accepting only those that increase local diversity above some floor.

This prevents a fleet consensus from collapsing into monoculture. Two ships might agree (via CRDT) that agent #42 exists, but if agent #42 is already well-represented in local vector space, the ship may choose not to breed from it.

---

## 4. Proposed API: `HolonomyConsensus.merge_with_diversity()`

```python
class HolonomyConsensus:
    """
    Distributed consensus that respects both CRDT lineage merge
    and HDC diversity preservation.
    """

    def __init__(
        self,
        crdt_engine: CRDTMergeEngine,
        diversity_scorer: HDCDiversityScorer,
        novelty_floor: float = 0.15,  # minimum Hamming distance to accept
        diversity_cap: int = 500,  # max agents after merge
    ):
        self.crdt = crdt_engine
        self.diversity = diversity_scorer
        self.novelty_floor = novelty_floor
        self.diversity_cap = diversity_cap

    def merge_with_diversity(
        self,
        local_pop: list[Agent],
        remote_pop: list[Agent],
        local_vector_table: FluxVectorTable,
        remote_vector_table: FluxVectorTable,
    ) -> tuple[list[Agent], FluxVectorTable, DivergenceReport]:
        """
        Merge two populations, then filter by diversity contribution.

        Steps:
        1. CRDT merge: resolve conflicts, validate lineages, sync vector tables.
        2. HDC scoring: for each merged agent, compute novelty vs local baseline.
        3. Filter: reject agents below novelty_floor unless they are
           the *only* representative of a unique lineage branch.
        4. Prune: if count exceeds diversity_cap, drop lowest-novelty agents
           (using a min-heap keyed by novelty score).
        5. Return: (pruned_agents, merged_vt, divergence_report)
        """
```

### Key Design Decisions

- **Novelty floor of 0.15** means an agent must differ by at least 15% of its bits from the local population baseline to be considered "worth keeping for diversity."
- **Lineage exemption**: agents representing a previously-unseen branch (unique `all_parents` fingerprint) bypass the novelty floor. This prevents genuine evolutionary novelty from being discarded just because its vector happens to be close to a local cluster.
- **Pruning by min-heap** keeps the merge O(n log k) where k = `diversity_cap`, rather than O(n²) pairwise comparisons.

---

## 5. Open Questions

### 5.1 Conflict Resolution When Two Ships Breed the Same Agent ID

The current CRDT merge conflates **identity** (agent ID) with **lineage** (parents). If Ship A breeds agent #99 from (3, 7) and Ship B independently breeds agent #99 from (5, 11), the merge function currently:

1. Detects a lineage conflict via `_lineage_conflicts()`.
2. Merges both parent sets into `all_parents = [3, 7, 5, 11]`.
3. Picks the winner by fitness/timestamp and assigns the merged lineage.

**Open question:** Should two independently-evolved agents with the same ID be considered *the same agent* or *colliding identities*? Merging lineages treats them as one agent with a broadened pedigree. But this loses the information that two distinct evolutionary paths converged on the same phenotype. An alternative: assign a **conflict flag** or **fork the ID** (e.g., #99-A and #99-B) when lineages diverge significantly, preserving both branches.

### 5.2 Entropy vs. Consensus Tradeoff

CRDT merge optimizes for **consensus** — a single, consistent fleet-wide population. HDC diversity optimizes for **entropy** — a population with high bit-flip distance. These goals can conflict:

- **Scenario**: Ship A has a highly diverse population (high mean Hamming distance). Ship B has a monoculture (low diversity, but higher average fitness). When they merge, CRDT keeps the higher-fitness agents from B. HDC would prefer keeping A's diverse agents even if their fitness is lower.
- **The tradeoff**: Should the merge function weight fitness against novelty? If so, by what ratio? A weighted score like `value = α·fitness + β·novelty` could replace the current pure-fitness tie-breaker, but the α/β coefficients become hyperparameters with fleet-wide consequences.

### 5.3 Vector Table Timestamp vs. Content

`sync_vector_table()` uses **timestamp LWW** on `extra["last_updated"]`, not vector content comparison. This means if Ship A and Ship B both update agent #42, the one with the later clock wins — even if the loser's vector was objectively more novel. Should the merge function compare **Hamming distance to the local population** when choosing which vector to keep? Or is timestamp determinism the lesser evil compared to vector-space heuristics?

### 5.4 AVX-512 Dependency in Distributed Merge

`HAS_AVX512` is detected at import time via `/proc/cpuinfo`. If Ship A (AVX-512 server) merges with Ship B (ARM edge node), they will have different HDC performance characteristics. The CRDT merge itself is architecture-agnostic, but the **diversity scoring step** in the proposed API would run at different speeds. Should the merge protocol negotiate a common "novelty resolution" (e.g., Ship A pre-computes novelty scores and ships them as metadata alongside agents), or is it acceptable for each ship to re-score locally after merge?

---

## Key Insight

> **CRDT and HDC are not competing merge strategies — they are orthogonal axes.** CRDT resolves "who" (identity and lineage consensus across ships). HDC resolves "how different" (local vector-space novelty). The fleet needs both: without CRDT, ships fork into incompatible populations; without HDC, consensus collapses into monoculture. The proposed `merge_with_diversity()` API is essentially a **two-stage filter**: first agree on what exists (CRDT), then agree on what matters (HDC). The real design challenge is not combining the algorithms — it's deciding the **weight** between fitness consensus and diversity preservation when they pull in opposite directions.

---

*Word count: ~1,450 words*
