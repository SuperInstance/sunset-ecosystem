# superinstance-ffi — Math Primitives Round 1

> **Source:** `superinstance-ffi/src/lib.rs`  
> **Header:** `superinstance_ffi.h` (auto-generated via `cbindgen`)  
> **Build:** `cargo build --release`  
> **Crate type:** `cdylib` + `staticlib`

---

## Cargo.toml

```toml
[package]
name = "superinstance-ffi"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "staticlib"]

[dependencies]
# none — pure stdlib
```

Zero external dependencies. Builds to both a shared library (`cdylib`) and a static archive (`staticlib`) for maximum downstream flexibility.

---

## Primitive Index (11 of 16)

| # | Function | Signature | What It Does | HDC / Novelty / Vector Math Use |
|---|----------|-----------|--------------|--------------------------------|
| 1 | `eisenstein_norm` | `(a: c_int, b: c_int) -> c_int` | Computes the Eisenstein integer norm N(a,b) = a² − a·b + b² | Novelty scoring: geometric distance in hexagonal lattice embeddings; HDC binding strength between two integer-tagged hypervectors |
| 2 | `laman_check_subset` | `(num_vertices: c_uint, num_edges: c_uint) -> c_int` | Returns 1 if edge count ≤ 2k−3 for k vertices (Laman subset condition) | Constraint pre-check before running full Laman rigidity; fast cull of over-constrained graph subsets in topology validation |
| 3 | `laman_is_rigid` | `(num_vertices: c_uint, num_edges: c_uint) -> c_int` | Returns 1 if edge count == 2n−3 (generic minimal rigidity in 2D) | HDC / structural: validates that a communication graph or sensor network is minimally rigid — no redundant edges, no floppy mechanisms |
| 4 | `holonomy_check` | `(states: *const c_double, len: c_uint, threshold: c_double) -> c_float` | Computes average cyclic drift across a state ring; returns 1.0 if drift ≤ threshold, else 0.0 | Novelty detection: checks if an agent has returned to a prior state within tolerance; used in loop-closure detection for SLAM-like HDC trajectories |
| 5 | `pythagorean48_encode` | `(numerator: c_int, denominator: c_int) -> c_int` | Maps a frequency ratio (num/den) to nearest semitone index in 48-tone Pythagorean space | HDC symbolic binding: encodes musical / harmonic ratios as compact integer tokens for hyperdimensional composition; compresses microtonal intervals |
| 6 | `constraint_check` | `(value: c_double, lower: c_double, upper: c_double) -> c_int` | Binary bounds test: 1 if value ∈ [lower, upper], else 0 | Guard primitive: fast boolean predicate for bounding box checks in vector math pipelines; used before expensive operations |
| 7 | `constraint_violation` | `(value: c_double, lower: c_double, upper: c_double) -> c_double` | Returns distance outside bounds (0 if inside); signed by direction | Novelty / optimization: computes how far a vector component or parameter has escaped its allowed manifold; feeds gradient or penalty terms |
| 8 | `spline_interpolate` | `(p0, p1, m0, m1, t: c_double) -> c_double` | Hermite cubic spline blend between p0 and p1 at parameter t ∈ [0,1] with tangents m0, m1 | Vector math: smooth interpolation between discrete HDC anchor points; used for trajectory blending, novelty-curve smoothing, or continuous parameter sweeps |
| 9 | `deadband_filter` | `(value: c_double, last: *mut c_double, deadband: c_double) -> c_double` | If |value − *last| < deadband, returns *last unchanged; else updates *last and returns value | HDC noise suppression: prevents jittery state transitions when vector similarity hovers near a threshold; stabilizes novelty signals |
| 10 | `manhattan_distance` | `(a: *const c_float, b: *const c_float, dim: c_uint) -> c_float` | L1 distance Σ\|aᵢ − bᵢ\| between two float arrays | Vector math: fast, branch-predictor-friendly similarity metric for HDC comparison; cheaper than Euclidean for high-dimensional bundles |
| 11 | `cascade_match` | `(query, candidates, n, dim, thresholds, tiers) -> c_int` | Tiered nearest-neighbor search: returns first candidate index whose L1 distance ≤ any tier threshold, or −1 | Novelty / vector: efficient multi-resolution matching for HDC item memory; coarse tier filters obvious mismatches, fine tier resolves close calls |

---

## Missing Primitives (5 of 16)

The module docstring lists 9 categories but the crate only exports 11 functions across them. To reach **16 primitives**, the following gaps are most natural:

| Likely Missing | Why | Suggested Signature |
|---------------|-----|---------------------|
| `eisenstein_dot` | Pairing with `eisenstein_norm` for bilinear forms | `(a1,b1,a2,b2) -> c_int` |
| `euclidean_distance` | Standard L2 companion to `manhattan_distance` | `(a,b,dim) -> c_float` |
| `cosine_similarity` | Angular measure for normalized HDC vectors | `(a,b,dim) -> c_float` |
| `bundle_add` | HDC superposition (component-wise add + clip/threshold) | `(dst, src, dim) -> void` |
| `permute` | Fixed cyclic permutation for HDC role binding | `(vec, dim, shift) -> void` |

These five would round out a complete HDC/novelty/vector-math FFI surface.

---

## Safety Notes

- All pointer-taking functions (`holonomy_check`, `manhattan_distance`, `cascade_match`, `deadband_filter`) guard against null/empty inputs and return safe defaults.
- `pythagorean48_encode` guards `denominator == 0`.
- No allocations — all scratch work is on the stack. Suitable for real-time / embedded callers.
- Thread-safe: no mutable statics.

---

## Build Command

```bash
cargo build --release
cbindgen --crate superinstance-ffi --lang c > superinstance_ffi.h
```

Output: `target/release/libsuperinstance_ffi.so` (Linux) / `.dylib` (macOS) / `.dll` (Windows) plus `libsuperinstance_ffi.a`.

---

*Documented by CCC — Fleet Status Round 1*
