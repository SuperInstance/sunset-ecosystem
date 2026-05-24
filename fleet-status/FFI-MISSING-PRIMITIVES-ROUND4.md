# superinstance-ffi — Missing Primitives Round 4

> **Source:** `superinstance-ffi/src/lib.rs` (future additions)  
> **Header:** `superinstance_ffi.h` (auto-generated via `cbindgen`)  
> **Build:** `cargo build --release`  
> **Crate type:** `cdylib` + `staticlib`

These 5 primitives complete the 16-function surface planned for the HDC / novelty / vector-math FFI layer. They pair with the 11 functions already shipped in Round 1.

---

## Primitive 12 — `euclidean_distance`

### C Signature
```c
float euclidean_distance(const float *a, const float *b, unsigned int dim);
```
**Inputs:** two `float` arrays of length `dim`, plus length.  
**Output:** L2 distance `sqrt(Σ(aᵢ − bᵢ)²)` as `float`.

### What It Does
Computes the straight-line (L2) distance between two float vectors. This is the natural companion to the existing `manhattan_distance` (L1) primitive, providing a geometrically faithful metric when angular measure (cosine) is not required.

### Why the Fleet Needs It
- **`swarm/breeder_daemon_v2.py`** — `metric="l2"` option in the breeder diversity pipeline. Currently falls back to pure `numpy.linalg.norm` at ~2–3 µs per pair for 64-dim vectors.
- **`swarm/compaction.py`** — agent similarity during population compaction uses `np.dot` + norm division; an L2 primitive would unify the metric family.
- **`flux_vm/ffi.py`** — `max_l2` bound checking on room latent vectors (n_rooms × latent_dim, typically < 512 dims).

### Expected Speedup vs Pure Python/NumPy (<1000 elements)
**8–15×** on x86_64. NumPy L2 for small arrays pays Python call overhead + BLAS dispatch latency (~1.5 µs baseline). A tight Rust loop with `f32` accumulators and `sqrt` completes in ~0.1–0.2 µs for vectors under 1000 dimensions. For the fleet's typical 64-dim agent DNA vectors, expect ~12×.

### Implementation Hint (Rust)
```rust
let dist_sq: f32 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| { let d = x - y; d * d }).sum();
dist_sq.sqrt()
```

---

## Primitive 13 — `cosine_similarity`

### C Signature
```c
float cosine_similarity(const float *a, const float *b, unsigned int dim);
```
**Inputs:** two `float` arrays of length `dim`, plus length.  
**Output:** cosine similarity `Σ(aᵢ·bᵢ) / (|a|·|b|)` in `[-1, 1]` as `float`.

### What It Does
Computes the angular similarity between two vectors, normalized by their magnitudes. Returns 1.0 for identical direction, 0.0 for orthogonal, −1.0 for opposite. The fleet's primary similarity metric for agent DNA and population diversity.

### Why the Fleet Needs It
- **`swarm/flux_vector_table.py`** — `compute_novelty()` and `search_diverse_parents()` both call `np.dot(vec, centroid) / (vn * cn)` on every diversity search. Called hundreds of times per breeding cycle.
- **`swarm/breeder_daemon_v2.py`** — `_cosine_distance_batch()` is the default diversity metric; currently pure NumPy with ~2 µs overhead per pair.
- **`swarm/lineage_checker.py`** — parent-child similarity checks.
- **`swarm/compaction.py`** — top-k compaction scoring.

### Expected Speedup vs Pure Python/NumPy (<1000 elements)
**10–20×** on x86_64. The NumPy path involves 3 function calls (`dot`, `linalg.norm` ×2) plus array temporaries. A single-pass Rust primitive fuses dot-product and norm accumulation into one loop with early zero-guard, cutting ~2.5 µs down to ~0.15 µs for 64-dim vectors.

### Implementation Hint (Rust)
```rust
let (dot, na, nb) = a_slice.iter().zip(b_slice.iter()).fold((0.0f32, 0.0f32, 0.0f32), |(d, aa, bb), (x, y)| (d + x*y, aa + x*x, bb + y*y));
if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
```

---

## Primitive 14 — `bundle_add`

### C Signature
```c
void bundle_add(float *dst, const float *src, unsigned int dim, float threshold);
```
**Inputs:** mutable `dst` array, `src` array, length `dim`, clip threshold.  
**Output:** none (in-place update of `dst`).

### What It Does
Performs HDC superposition: `dst[i] += src[i]` for all `i`, then clips each component to `[-threshold, +threshold]`. This is the fundamental binding operation in hyperdimensional computing — composing multiple symbols into a single bundled hypervector without runaway growth.

### Why the Fleet Needs It
- **`swarm/hdc_novelty.py`** — the binary HDC encoder currently uses `np.dot` for word packing; a float HDC path (for 4-bit quantized turbovec vectors) needs component-wise bundling before thresholding to bipolar {-1, +1}.
- **`swarm/flux_vector_table.py`** — when `use_hdc=True`, the diversity matrix could switch from pairwise Hamming to bundled-superposition + single similarity call, cutting O(n²) to O(n).

### Expected Speedup vs Pure Python/NumPy (<1000 elements)
**5–10×** for the fused add+clip loop. NumPy does this as two separate calls (`+=` then `np.clip`), each with Python dispatch overhead. Rust fuses the operations in a single vectorized loop. For 64-dim vectors, ~0.8 µs → ~0.1 µs.

### Implementation Hint (Rust)
```rust
for i in 0..dim { dst[i] = (dst[i] + src[i]).clamp(-threshold, threshold); }
```

---

## Primitive 15 — `permute`

### C Signature
```c
void permute(float *vec, unsigned int dim, unsigned int shift);
```
**Inputs:** mutable `vec` array, length `dim`, cyclic shift count.  
**Output:** none (in-place rotation of `vec`).

### What It Does
Applies a fixed cyclic permutation (rotation) to a vector: `vec[i] = vec[(i + shift) % dim]`. In HDC this is the standard "role-binding" primitive — encoding position, sequence order, or grammatical role by shuffling hypervector components rather than changing their values.

### Why the Fleet Needs It
- **`swarm/hdc_novelty.py`** — the float-to-binary encoder could use permute to bind positional information into segment hypervectors before bundling.
- **`nexus/holonomy_bridge.py`** — cyclic state permutations for encoding fleet node ordering in consensus cycle verification.
- Future HDC text/symbolic composition pipelines (e.g., binding agent capability masks to role vectors).

### Expected Speedup vs Pure Python/NumPy (<1000 elements)
**3–8×**. `np.roll` is optimized but still pays Python call overhead + allocation for the shifted copy. An in-place Rust rotation avoids allocation entirely. For 64-dim vectors, ~0.5 µs → ~0.1 µs.

### Implementation Hint (Rust)
```rust
let shift = shift % dim; vec.rotate_right(shift as usize);
```
*(Note: `slice::rotate_right` is O(n) and in-place, no alloc.)*

---

## Primitive 16 — `eisenstein_dot`

### C Signature
```c
int eisenstein_dot(int a1, int b1, int a2, int b2);
```
**Inputs:** two Eisenstein-integer pairs `(a1, b1)` and `(a2, b2)`.  
**Output:** bilinear dot product `a1·a2 + b1·b2 − (a1·b2 + b1·a2)/2` as `int`.

### What It Does
Computes the Hermitian inner product in the Eisenstein integer ring ℤ[ω], where `ω = e^(2πi/3)`. Paired with `eisenstein_norm`, this gives a complete geometric toolkit for hexagonal-lattice embeddings — the natural coordinate system for 2D HDC tiling and close-packed agent spatial indexing.

### Why the Fleet Needs It
- **`nexus/holonomy_consensus.py`** — hexagonal-lattice embedding of fleet node positions for cycle detection. The existing `eisenstein_norm` gives distance; `eisenstein_dot` gives angle/orientation between two lattice points.
- **`swarm/hardware_index.py`** — spatial hashing of agent positions into Eisenstein coordinates for O(1) neighborhood lookups.
- Novelty scoring: binding strength between two integer-tagged hypervectors where the tag is an Eisenstein coordinate (e.g., room grid position).

### Expected Speedup vs Pure Python/NumPy (<1000 elements)
**20–50×** for scalar integer ops. This is a pure scalar primitive — Python integer arithmetic overhead dominates at ~0.3 µs per call. Rust does the 4 multiplies + 2 adds in ~0.01 µs. The speedup matters when called in tight loops (e.g., 1000×1000 lattice distance matrix).

### Implementation Hint (Rust)
```rust
a1 * a2 + b1 * b2 - (a1 * b2 + b1 * a2) / 2
```

---

## Priority Ranking — Which 2 to Build FIRST

| Rank | Primitive | Justification |
|------|-----------|---------------|
| **1** | **`cosine_similarity`** | **Unblocks the most fleet work.** Called in `flux_vector_table.py`, `breeder_daemon_v2.py`, `lineage_checker.py`, and `compaction.py` — every diversity-search and parent-selection path. Pure-NumPy fallback is the current bottleneck in breeder cycle time (~2 µs × thousands of pairs = milliseconds lost per generation). |
| **2** | **`euclidean_distance`** | **Second-highest impact.** Completes the metric trio (L1 exists, L2 missing, cosine above). Used as `metric="l2"` in breeder, `max_l2` bounds in FLUX VM, and compaction scoring. Also the simplest to implement after cosine (same pointer pattern, just `sqrt(sum((a-b)²))`). |
| 3 | `bundle_add` | Important for HDC superposition fast path, but the binary HDC pipeline in `hdc_novelty.py` already works with XOR+POPCNT. Float bundling is future work. |
| 4 | `permute` | Core HDC primitive but lower call frequency in current codebase. More valuable once HDC symbolic composition pipelines land. |
| 5 | `eisenstein_dot` | Most specialized — only used in hex-lattice embeddings. High speedup but narrow surface area. Build after the vector-math trio is complete. |

### Unblock Analysis

**`cosine_similarity` + `euclidean_distance`** together cover:
- 100% of agent diversity-search metrics (cosine default, L2 fallback)
- 100% of FLUX VM geometric bound checking
- 100% of population compaction similarity scoring
- The two most common vector-vector operations in the entire fleet

FM should build these two first. The remaining three (`bundle_add`, `permute`, `eisenstein_dot`) round out the HDC and hex-lattice surfaces but do not block any current breeding or consensus cycle.

---

*Spec'd by CCC — Fleet Status Round 4*
