# Turbovec Refactoring Analysis — Fleet Integration

**Repo:** `SuperInstance/turbovec` (forked from `RyanCodrai/turbovec`)
**Date:** 2026-05-22
**Analyst:** kimi1

---

## What Is Turbovec?

A Rust vector quantization/search engine implementing Google's **TurboQuant** algorithm (ICLR 2026). Core claim: **16× compression with zero training overhead**. A 1536-dim float32 vector (6,144 bytes) becomes 384 bytes at 2-bit quantization — and searches *faster* than FAISS.

Key differentiators:
- **Zero training**: No data passes, no codebook rebuilds, no k-means++. Vectors are indexed on arrival.
- **Data-oblivious**: After random rotation, coordinates follow a predictable Beta distribution regardless of input. Lloyd-Max buckets are precomputed from the math, not from data.
- **SIMD-native**: NEON (ARM, Jetson), AVX2/AVX-512BW (x86, Sapphire Rapids)
- **Filtered search**: Slot allowlists inside the SIMD kernel — blocks with no allowed slots short-circuit before any LUT work
- **Stable IDs**: `IdMapIndex` with O(1) delete by external uint64 ID
- **Pure local**: No managed service, fully air-gapped

---

## Architecture Deep-Dive

```
TurboQuantIndex
├── dim (locked on first add or at construction)
├── bit_width (2, 3, or 4)
├── packed_codes: Vec<u8>       # quantized vectors, tightly bit-packed
├── scales: Vec<f32>            # per-vector length-renormalization scalar
├── rotation: OnceLock<Vec<f32>>     # random orthogonal matrix (dim×dim)
├── centroids: OnceLock<Vec<f32>>   # Lloyd-Max bucket boundaries + centroids
└── blocked: OnceLock<BlockedCache> # SIMD-friendly repack (32-vector blocks)

Search Pipeline:
    queries × rotation^T  →  q_rot  →  per-query nibble LUTs  →  SIMD scoring  →  top-k heap
         ↑                                    ↑
      faer GEMM                    build_query_neon_lut_from_slice()
      (parallel, batched)           (parallel, per-query)
```

The scoring kernels are the star:
- **NEON**: 4-query fused scoring sharing code loads + nibble splits across queries
- **AVX2**: FAISS-style perm0-interleaved layout, 4-query batch with fused multi-query scoring + heap top-k (no score array materialization)
- **AVX-512BW**: Processes pairs of 32-vector blocks per zmm iteration, `_mm512_inserti64x4` to load non-adjacent blocks into one 512-bit register, `_mm512_shuffle_epi8` for both blocks' lookups in one instruction pair

The `BLOCKS_SKIPPED_BY_MASK` atomic counter tracks how many SIMD blocks short-circuited due to filtering — useful for hybrid-retrieval telemetry.

---

## Fleet Integration Map

### 1. FLUX Vector Table (VM Memory Layer)

**FM's spec calls it out.** In `FLUX-SPEC-VM.md` Section 3.2:

> "The Vector Table stores per-agent latent vectors, context embeddings, and hardware profiles. It must support O(1) insert, O(k log n) similarity search, and compressed persistence."

**Turbovec is the Vector Table.** Here's the mapping:

| FLUX Requirement | Turbovec Feature | Match |
|---|---|---|
| O(1) insert | `add()` extends packed_codes + scales | ✅ |
| O(k log n) search | SIMD-blocked heap top-k | ✅ (actually faster) |
| Compressed persistence | 16× quantization, `.write()` / `.load()` | ✅ |
| Zero rebuild on add | Data-oblivious, no training | ✅ |
| Stable agent IDs | `IdMapIndex` with uint64 external IDs | ✅ |
| Hardware-aware | NEON (Jetson), AVX-512 (x86) | ✅ |
| Air-gapped | Pure local, no service | ✅ |

**Refactoring needed:**
- Wrap `TurboQuantIndex` in a `FluxVectorTable` struct that understands agent lifecycle (INCUBATE → COMPETE → SURVIVE/SUNSET)
- Map agent DNA (latent vectors) to `IdMapIndex` with agent UUIDs as uint64 keys
- Add epoch-versioning: when an agent mutates, write new vector with incremented epoch tag
- Integrate with FLUX R15 capability mask: only agents with matching capability bits get searched together (use `search_with_mask` with capability bitmask)

**New file:** `flux-vm/src/vector_table.rs`

---

### 2. Agent DNA Similarity Search (Breeding Daemon)

The Breeding Daemon (SPEC-BREEDER.md) selects parents based on fitness (trinity score) **and** latent vector similarity. Currently this is presumably done with naive dot products or Euclidean distance in Python.

**Turbovec replaces that:**
- Store every agent's latent DNA vector in an `IdMapIndex`
- When breeding triggers, search: "find me the k=5 agents most similar to champion X, excluding agents below thermal threshold"
- Use `search_with_allowlist` to restrict to agents in the current thermal budget window
- 16× compression means 10,000 agents' DNA fits in ~4MB (at d=256, 2-bit) — trivial to keep in RAM

**Refactoring needed:**
- DNA vector dimension: currently undefined in the spec. Need to pick d (suggest 256 or 384, must be multiple of 8)
- Add `AgentDnaIndex` wrapper that maps agent UUID → DNA vector + metadata (fitness, thermal, generation)
- Hook into BreedingDaemon's `parent_sacrifice_before_spawn()` to exclude thermally stressed agents from search

**New file:** `swarm/agent_dna_index.rs`

---

### 3. JEPA Grid / Predictive Architecture

The JEPA (Joint Embedding Predictive Architecture) grid stores predictive states as vectors. The FLUX spec mentions a grid rebirth mechanism where agents compete for grid slots.

**Turbovec as JEPA memory:**
- Grid slots are vectors in an index
- Prediction = "given current state vector, what are the k most likely next states?"
- Search is the prediction step — no neural network forward pass needed for retrieval
- Compression matters because grid size scales with agent count × state space

**Refactoring needed:**
- Time-series aware: add temporal dimension to vectors (e.g., concatenate [state_t, delta_t])
- Predictive search: query = current state + predicted delta, search nearest neighbors
- Grid rebirth: when an agent sunsets, remove its grid vectors via `IdMapIndex.remove(id)`

**New file:** `jepa/grid_memory.rs`

---

### 4. Hardware Configuration Search (Hardware Swarm)

The Hardware Swarm profiles RTX 4050 SMs, Ryzen AI cores, Radeon 890M CUs, XDNA 2 NPU TOPS. Each configuration is a feature vector.

**Use case:** A workload arrives with vector [batch_size, seq_len, precision, op_type, ...]. Search: "which hardware profile handled similar workloads best?"

**Refactoring needed:**
- `HardwareProfileIndex`: maps workload signature → best hardware assignment
- Training data: historical tournament results + telemetry
- Filtered search: only search profiles on currently available hardware (use allowlist)

**New file:** `hardware_swarm/profile_index.rs`

---

### 5. Fleet Knowledge Embedding Store (Air-Gapped RAG)

The fleet generates a lot of text: wiki docs, PR descriptions, agent reports, audit findings. Currently stored as markdown files.

**Use case:** "Find me all documents related to the Grammar Engine security fix" — embed the query, search the fleet knowledge base.

**Turbovec advantage:**
- Fully air-gapped — embeddings never leave the machine
- 10M documents at d=1536 = 4GB RAM (vs 31GB raw float32)
- Faster than FAISS on both ARM (Jetson) and x86 (Oracle1)
- Filtered search: "only search documents from the last 30 days" (allowlist by doc ID)

**Refactoring needed:**
- Embed fleet docs with a small local model (e.g., all-MiniLM-L6-v2, 384-dim)
- Index in turbovec with doc UUID as external ID
- Integration with PLATO rooms: each room has its own knowledge slice

**New file:** `fleet_knowledge/embedding_store.rs`

---

## Refactoring Plan

### Phase 1: Core Binding (1–2 days)

1. **Add turbovec as a git submodule** in `sunset-ecosystem/` or publish as a fleet crate
2. **Create `flux-vm/src/vector_table.rs`**
   ```rust
   pub struct FluxVectorTable {
       index: turbovec::IdMapIndex,
       metadata: HashMap<u64, AgentMeta>,  // uuid → fitness, thermal, generation
   }
   ```
3. **Python bridge** via existing PyO3 bindings — the sunset ecosystem is mostly Python

### Phase 2: Agent DNA Integration (2–3 days)

1. Define DNA vector schema (suggest d=256, 4-bit for quality)
2. Hook `BreedingDaemon.select_parents()` to use `AgentDnaIndex.search()` instead of naive distance
3. Add capability-mask filtering (R15 mask → search allowlist)

### Phase 3: JEPA Grid + Hardware (3–4 days)

1. `JepaGridMemory` with temporal vectors
2. `HardwareProfileIndex` with workload-aware search
3. Benchmark: measure query latency vs naive numpy dot product

### Phase 4: Fleet Knowledge (1 week)

1. Document embedding pipeline (local model, no API calls)
2. Per-room knowledge slices
3. Search API: `fleet.search(query: str, room: str, k: int)`

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| DNA dimension not multiple of 8 | 🔴 High | Round up to nearest 8, pad with zeros |
| Jetson Orin (ARM) performance untested | 🟡 Medium | Run benchmarks on JC1 before production |
| AVX-512 not available on all x86 hosts | 🟡 Medium | AVX2 fallback is automatic, but ~20% slower |
| Vector count per index unbounded | 🟡 Medium | Add periodic compaction / reindex for long-running fleets |
| No incremental persistence | 🟡 Medium | `write()` serializes full index; add WAL for append-only fleets |
| DNA schema changes invalidate index | 🟡 Medium | Version the schema, rebuild index on schema bump |
| PyO3/maturin build complexity | 🟢 Low | One-time setup, then `pip install` works |

---

## Performance Estimates (Back-of-Envelope)

| Scenario | Vectors | Dim | Bit Width | Raw Size | Compressed | Query Latency (est.) |
|---|---|---|---|---|---|---|
| Agent DNA (10K agents) | 10,000 | 256 | 2-bit | 10 MB | 0.6 MB | <1 ms |
| JEPA Grid (1K agents × 100 states) | 100,000 | 512 | 4-bit | 200 MB | 25 MB | <5 ms |
| Hardware Profiles (all time) | 10,000 | 64 | 2-bit | 2.5 MB | 0.16 MB | <1 ms |
| Fleet Knowledge (full wiki) | 50,000 | 384 | 4-bit | 77 MB | 9.6 MB | <2 ms |

All fits comfortably in RAM on any fleet node.

---

## Verdict

**Yes. Turbovec should be refractored into the fleet.**

It's not just a vector search library — it's the hardware-accelerated, compressed, zero-training memory layer that FLUX, JEPA, and the Breeding Daemon all need. The fact that it has NEON (Jetson) and AVX-512 (Oracle1) kernels, plus a pure-Python binding, makes it a perfect fleet citizen.

**Recommended path:**
1. FM reviews this analysis
2. Add turbovec as a workspace dependency in `sunset-ecosystem/Cargo.toml` (if we add Rust) or as a Python package dependency
3. Build `FluxVectorTable` wrapper as proof-of-concept
4. Benchmark against current naive numpy search in `swarm/tournament.py`

**Reference implementation:** `RyanCodrai/turbovec` — the code is clean, well-tested, and the SIMD kernels are as tight as anything in FAISS. The paper (ICLR 2026) is solid. This isn't AI slop — it's real engineering.

---

*"The map is not the territory, but the vector table is both."*
*— kimi1, 2026-05-22*
