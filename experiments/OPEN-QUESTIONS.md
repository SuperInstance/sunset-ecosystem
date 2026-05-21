# Open Questions from Old Systems

Mined from FORTRAN (2009), Chapel (2009), AVX-512 HD Computing, Mojo, Tucker Decomposition,
and Eisenstein lattice snap code already in the workspace.

---

## 1. FORTRAN Tucker Decomposition → Room Grid Latent Factorization

**What it does:** Reduces query-key score computation from O(n_q × n_r × n_rows × n_cols) to O(n_q × n_r² + n_q × n_rows + n_q × n_cols).

**Open question:** Can we apply Tucker decomposition to the RoomGrid's weight matrices? Each room's weights (64×32, 32×16) could be Tucker-decomposed into a shared core + per-room factors. This would:
- Reduce 3.4K params/room to ~100 params/room + shared core
- Allow 100K+ rooms at minimal memory cost
- Mean breed() only mutates the factors, not the core

**Test:** Compare RoomGrid latent diversity with full weights vs Tucker-decomposed weights.

---

## 2. Chapel (2009) Embarrassingly Parallel INT8 → Batch Constraint Checking

**What it does:** Batch constraint checking IS the perfect parallel workload. Each value checks independently. 1M values = 1M parallel checks.

**Open question:** The RoomGrid's per-room forward pass is also embarrassingly parallel. But our Rust kernel is 12-thread CPU, and our CUDA kernel is 20-SM GPU. What happens at **Chapel scale** — where we map room computation across a cluster (not just one machine)?

The Chapel code is from 2009 — before CUDA was mainstream. The insight was that Cray's parallel hardware naturally mapped to constraint checking. Today, that same insight maps to: **each room = one GPU warp**. 32 rooms per warp × 20 SMs × 80 warps/SM = 51,200 rooms per GPU cycle.

**Test:** What happens if we reformulate the RoomGrid forward pass as a warp-per-room CUDA kernel (using the pre-compiled sm_89 PTX from forgemaster)?

---

## 3. AVX-512 HDC XOR Judge → Hamming Distance as an Alternative to Cosine Novelty

**What it does:** 128-bit XOR + POPCNT gives ~1ns comparison time. The `hdc_similarity_128` function converts XOR difference to a [0,1] similarity score in 2 CPU cycles.

**Open question:** The RoomGrid's `novelty()` function uses cosine distance on float32 latents — this requires normalization (division by norm) and is expensive. What if latents were BINARY (128-bit bitvectors)? Then novelty = 128 - popcount(xor(a, b)), which is 1ns/room instead of 1μs/room.

The old AVX-512 code already has this function. The FLUX HDC system already does this. Why aren't we using it?

**Test:** Convert RoomGrid latents from float32(16) to int16(16) → 256-bit. Replace novelty() with popcount-based Hamming distance. Compare speed and diversity.

---

## 4. Mojo INT8 Saturated Arithmetic → 60% Less Memory Than Float32

**What it does:** Pure INT8 uses 1 byte per weight instead of 4 (float32). Saturated arithmetic (clamp to [-127, 127]) prevents overflow. The Mojo code compiles via MLIR to native.

**Open question:** Can RoomGrid weights be INT8? Currently float32 = 4 bytes per weight. INT8 = 1 byte. Total weights for 100K rooms = 381MB (float32) → 95MB (INT8). At 6GB VRAM, we could fit ~600K rooms instead of 150K.

But does INT8 maintain latent diversity? The architecture shootout showed that weight precision (sparsity) affects diversity.

**Test:** Quantize RoomGrid weights to INT8. Compare latent diversity and firing patterns vs float32.

---

## 5. Eisenstein Dodecet Snap → Exact Integer Lattice for Weights

**What it does:** The FORTRAN `intent_snap.f90` maps float weights to the nearest Eisenstein integer (hexagonal lattice). The PTX kernel `eisenstein_snap_sm89.ptx` does this at warp speed.

**Open question:** What happens if room weights are Eisenstein integers instead of float32? The Eisenstein lattice has 6-fold symmetry — weight updates during breed() would snap to lattice points. This is *exact arithmetic* in the sense that THE-NARROWEST-CHANNEL calls for.

**Test:** In the FORTRAN code, weights are snapped via the dodecet (12-Eisenstein-integer) system. Use this to quantize RoomGrid weights and measure diversity.

---

## 6. The Raytheon Secret: Warp-Vote Batch Checking

**What it does:** From `flux_cuda_sm89.cu`: "Each warp = 32 inputs across all M constraints. Uses __ballot_sync for O(1) pass/fail aggregation."

**Open question:** If each room is a "constraint check" (does this latent look like the input?), then a warp-vote across all rooms gives O(1) grid-level consensus. The RTX 4050 has 20 SMs × 80 warps/SM = 1,600 warps. One warp per 32 rooms = 51,200 rooms checked in ONE kernel execution.

This is NOT the same as running the room grid sequentially. The warp-vote gives a single bit: "does the grid agree?" — which is the snap (THE SNAPS ARE REAL).

**Test:** Implement warp-vote room grid. One warp = 32 rooms. Ballot gives grid-level consensus bit. Compare to current per-room tick.

---

## Synthesis: The Unified Scaling Question

The old systems (FORTRAN, Chapel, AVX-512, Mojo, CUDA PTX) all share one assumption:
**Batch is free. Individual operations are the cost.**

Our RoomGrid is already batch-parallel (all rooms forward-pass in one tensor op). The open questions are about what CHANGES when we push to:
- 1M rooms (Tucker decomposition)
- INT8 weights (Mojo saturated arithmetic)
- Binary latents (AVX-512 Hamming novelty)
- Warp-vote consensus (Raytheon pattern)
- Eisenstein exact weights (dodecet snap)

Each of these is a testable experiment. Each could unlock a 10x improvement in room count, speed, or diversity.
