# SCOUT Report — Low-Level Hardware Patterns for RoomGrid

**Run 1**: PTX kernel, Tucker decomposition, AVX-512 HDC
**Date**: 2026-05-21

---

## 1. Eisenstein Snap PTX Kernel (`eisenstein_snap_sm89.ptx`)

### What it does that RoomGrid doesn't

The PTX kernel maps arbitrary float32 (x,y) coordinates to **Eisenstein integers** — points on a hexagonal lattice with basis vectors `(1, 0)` and `(-1/2, √3/2)`. Each input point gets snapped to the nearest lattice point, returning `(a, b)` Eisenstein coordinates plus a `delta` (distance from the snap).

Key machinery (lines 40–75):
1. `b_f = y * (1 / 2√3)` → round to integer `b`
2. `a_f = x + b * 0.5` → round to integer `a`
3. Reconstruct snapped point: `sx = a - b*0.5`, `sy = b * √3/2`
4. `delta = sqrt((x-sx)² + (y-sy)²)` — snap error

The sm_89 variant processes **2 points per thread** via ILP (instruction-level parallelism), exploiting the larger 64K register file on Ada. Block size = 128 threads (lower occupancy, higher per-thread throughput).

### How to adapt it

**A. Eisenstein weight snapping in `breed()`.** Currently `breed()` (room_grid.py line 136–144) adds Gaussian noise to float32 weights. Instead: snap each weight to the nearest Eisenstein integer after mutation. This gives:
- **Exact arithmetic**: weight updates land on lattice points, no floating-point drift
- **6-fold symmetry**: the hexagonal lattice naturally produces diverse but structured weights
- **Deterministic breeding**: same (a,b) Eisenstein pair always produces the same weight

Implementation: after the existing `+= rng.randn() * 0.005` mutation, snap each weight element:
```python
# Snap weight w to Eisenstein integer
b = round(w * inv_2sqrt3)
a = round(w + b * 0.5)
w_snapped = a - b * 0.5  # x-component of Eisenstein point
```
This is ~3 FLOPs per weight element. For 3.4K params/room, negligible cost.

**B. CUDA warp-level snap for tournament.** The PTX kernel already processes 2 points per thread. If we treat each room's latent (16-dim) as 8 pairs of (x,y) coordinates, the existing kernel can snap all 16 latent dimensions across 250 rooms in a single kernel launch. This replaces numpy `forward_einsum` with a GPU Eisenstein-snapped forward pass.

**C. Delta as fitness signal.** The `delta` output (snap error) is a natural measure of how "lattice-aligned" a room's weights are. Rooms with low aggregate delta have weights near Eisenstein integers — they're "well-formed". Rooms with high delta are chaotic. Use `delta` as a breeding fitness metric alongside `activity`.

### New behaviors that emerge

- **Lattice attractors**: After many breed cycles, all weights drift toward Eisenstein lattice points. The grid self-organizes into a discrete structure.
- **Snapping as regularization**: breed() mutations that push weights off-lattice get snapped back. This prevents weight explosion without explicit clamping.
- **Hexagonal latent topology**: Eisenstein integers have 6-fold symmetry. Rooms near the same lattice region share structural properties, creating natural "neighborhoods" in weight space — a form of implicit clustering without k-means.

---

## 2. Tucker Decomposition (`tucker_decompose.f90`)

### What it does that RoomGrid doesn't

The FORTRAN module implements **Tucker-decomposed score computation** — factorizing a large score tensor into a shared core × factor matrices. The key routine is `compute_scores` (lines 24–69):

- Input: query vector `q`, row keys `(n_r, n_rows)`, col keys `(n_r, n_cols)`, core `(n_r, n_r)`
- Step 1: Project query → `q_row = row_keys * q` (O(n_r × n_rows))
- Step 2: Apply core → `q_core = core * q_row` (O(n_r²))
- Step 3: Column projection → `col_proj = col_keys^T * q_core` (O(n_r × n_cols))
- Step 4: Outer product → `scores = row_proj ⊗ col_proj` (O(n_rows × n_cols))

Total: O(n_r² + n_rows + n_cols) instead of O(n_rows × n_cols).

The reconstruction routine `tucker_reconstruct` (lines 80–96) shows the inverse: `X ≈ A × G × B^T`.

### How to adapt it

**A. Tucker-decomposed room weights.** Currently each room stores full weight matrices: `w1(64,32)`, `w2(32,16)`, `w3(16,16)` = 3,424 params/room. For 250 rooms = 856K params.

Tucker approach:
- **Shared core**: One `core_w1(r, r)`, `core_w2(r, r)`, `core_w3(r, r)` across all rooms (where r << min dimensions, say r=4)
- **Per-room factors**: Each room stores `A_i(64, r)`, `B_i(32, r)` for w1, etc.
- Per-room params: 64×4 + 32×4 + 32×4 + 16×4 + 16×4 + 16×4 = 704 params/room (vs 3,424)
- Shared core: ~200 params total
- **Memory drops 4×**, enabling 1M+ rooms in the same footprint

**B. breed() on factors only.** Currently breed() clones and mutates all 3.4K weights (room_grid.py line 136–144). With Tucker: breed only mutates the per-room factors `A_i, B_i`. The shared core stays fixed. This means:
- Mutation is smaller and more targeted (704 params vs 3,424)
- Core provides structural stability; factors provide room diversity
- Breed cost drops ~5×

**C. Tucker forward pass.** Replace `forward_einsum` with Tucker reconstruction:
```python
# Current: h = x @ w1[i] + b1  (64×32 matmul per room)
# Tucker:  h = (x @ A_i) @ core_w1 @ B_i^T + b1  (64×4 → 4×4 → 4×32)
```
The intermediate dimension r=4 means the expensive matmul is 4×4 instead of 64×32. Forward cost drops ~8× per room.

### New behaviors that emerge

- **Shared structure, individual expression**: All rooms share the same "skeleton" (core) but have unique "musculature" (factors). This is closer to biological systems where genes share a body plan but express differently.
- **Core evolution**: The core could be slowly updated (e.g., PCA of all room factors every N ticks). This gives the grid a "collective learning" mechanism without backprop.
- **Room count explosion**: At 704 params/room instead of 3,424, we can fit ~600K rooms in the same memory. At that scale, tournament selection and Penrose positioning become critical — exactly the infrastructure already in `swarm/penrose.py`.
- **Cross-room analysis**: With shared factors, you can compare rooms by their factor matrices directly (cosine distance on factor vectors) instead of comparing latent outputs. This is faster and more interpretable.

### Risk

The OPEN-QUESTIONS.md (#1) already flagged this: **does Tucker decomposition preserve latent diversity?** The architecture shootout showed JEPA kills diversity. Tucker is different (it's a decomposition, not an objective), but the reduced rank (r=4 vs full rank 32) might compress away the very diversity we need. **Must test experimentally.**

---

## 3. AVX-512 HDC XOR Judge (`flux_hdc_avx512.h` + `flux_avx512.c`)

### What it does that RoomGrid doesn't

Two key capabilities:

**HDC similarity** (`flux_hdc_avx512.h`):
- `hdc_hamming_128(a, b)`: XOR two 128-bit vectors, popcount the result → Hamming distance in ~1ns (line 30–33)
- `hdc_similarity_128(a, b)`: Convert Hamming distance to [0,1] similarity (line 35–38)
- `hdc_match_128(a, b, threshold)`: Binary pass/fail with configurable threshold (line 40–42)
- `hdc_batch_match_128(query, stored, n)`: Scan N stored vectors for best match (line 116–130)
- 512-bit and 1024-bit variants using VPOPCNTDQ (lines 52–80)

**AVX-512 batch ops** (`flux_avx512.c`):
- `avx512_range_check_batch`: 16 int32 range checks per cycle (lines 28–52)
- `avx512_domain_popcount`: Hardware popcount for 8 uint64 domains simultaneously (lines 58–80)
- `avx512_domain_intersect`: Bitmask AND for domain intersection (lines 86–96)
- `avx512_vnni_constraint_eval`: INT8 dot products via VNNI (lines 104–125)

### How to adapt it

**A. Binary latents + Hamming novelty.** This is OPEN-QUESTIONS.md #3 and it's the most impactful change. Currently `novelty()` (room_grid.py line 78–84) uses cosine distance on float32(16) latents:

```python
zn = z / (np.linalg.norm(z) + 1e-8)
rn = recent / (np.linalg.norm(recent, axis=-1, keepdims=True) + 1e-8)
return float(1.0 - (zn * rn).sum(axis=-1).mean())
```

This requires normalization + dot product = expensive. The HDC approach:
1. After the forward pass, binarize latents: `z_bin[i] = 1 if z[i] > 0 else 0` → 16-bit per room
2. Pack 16 rooms into one uint16 (or 4 rooms into uint64 as 16-bit fields)
3. Novelty = `16 - popcount(xor(z_bin, recent_bin))` → 1 instruction

The `hdc_hamming_128` function (flux_hdc_avx512.h line 30) already does this for 128-bit vectors. Our 16-dim latents pack into 16 bits — even cheaper.

**B. Range-check firing as AVX-512 batch.** The `tick()` method (room_grid.py line 100–113) fires rooms where `nv > 0.5` OR random < chaos. This is a batch range check. The `avx512_range_check_batch` (flux_avx512.c line 28) does exactly this: 16 range checks per cycle. Replace the Python loop with a vectorized check:
- Pack all novelties into an int32 array
- One AVX-512 call checks 16 rooms at once
- 250 rooms = 16 AVX-512 calls instead of 250 Python iterations

**C. VNNI for forward pass.** The `avx512_vnni_constraint_eval` (flux_avx512.c line 104) computes INT8 dot products via hardware. If we quantize room weights to INT8 (OPEN-QUESTIONS.md #4), the forward pass becomes:
- `h = int8_dot(w1_quantized[i], x_quantized)` via VNNI
- 64 products per cycle on Zen 5
- Would replace both the numpy einsum and Rust FFI paths

**D. Batch match for tournament.** The `hdc_batch_match_128` (flux_hdc_avx512.h line 116) finds the best match among N stored vectors. Use this for tournament selection: given a "target" latent, find which room matches it best. Currently there's no tournament matching in RoomGrid — this enables it at ~1ns per comparison.

### New behaviors that emerge

- **1000× faster novelty computation**: Cosine distance on float32(16) ≈ 1μs/room. Hamming on 16-bit ≈ 1ns/room. For 10K rooms × 20 history = 200K comparisons per tick: 200ms → 0.2ms.
- **Deterministic firing**: Binary novelty is a clean threshold — no floating-point edge cases. A room either fired or didn't, no ambiguity.
- **Bit-level breeding**: Instead of float32 mutation, breed() could flip bits in the binary latent. XOR with a random mask = mutation. This is HDC-style binding and is O(1) per room.
- **Room topology from Hamming distances**: All-pairs Hamming distances between room fingerprints give a natural graph structure. Rooms close in Hamming space are "similar" — this graph can drive tournament brackets, breeding selection, and Penrose position refinement.

### Concrete implementation path

1. Add `z_bin` field to RoomGrid: pack each latent into uint16 after forward pass
2. Replace `novelty()` with `16 - popcount(xor(z_bin, history_bin))`
3. Compare diversity scores (from architecture shootout) — should be identical or better since sign(x) preserves most information
4. If diversity holds, remove float32 novelty entirely

---

## Cross-Cutting Insight: The Batch-Parallel Convergence

All three systems share one message: **individual operations should be trivial; batch is the unit of work.**

| System | Individual Op | Batch Size | Key Insight |
|--------|--------------|------------|-------------|
| PTX Eisenstein | 3 FLOPs snap | 2 per thread, 128 threads/block | ILP doubles throughput |
| Tucker Decomp | r² core matmul | N rooms share core | Shared structure amortizes cost |
| AVX-512 HDC | 1ns XOR+popcount | 16 per cycle | Binary ops are essentially free |

RoomGrid is already batch-parallel in forward pass. But **novelty, breed, and tick** are still per-room Python loops. The convergence path:

1. **Tick → AVX-512 batch range check** (16 rooms/cycle)
2. **Novelty → Hamming popcount** (1ns/room vs 1μs/room)
3. **Breed → Eisenstein snap** (deterministic lattice mutation)
4. **Forward → Tucker reconstruction** (8× fewer ops per room)
5. **Memory → Tucker factors** (5× more rooms in same space)

Steps 1–2 are independent and can be tested immediately. Step 3 is a breed() modification. Step 4–5 are a full rewrite of the weight system and need the Tucker diversity experiment first.

---

## Recommended Experiment Priority

1. **Binary novelty** (AVX-512 HDC adaptation) — easiest to test, biggest speed win, minimal risk
2. **Eisenstein breed snapping** — small change to breed(), testable in one session
3. **Tucker weight decomposition** — requires architectural experiment first (does diversity survive rank reduction?)
4. **VNNI INT8 forward** — depends on quantization experiment from OPEN-QUESTIONS #4
