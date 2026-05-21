# SPEC-JEPA-GRID-OPTIMIZATION.md — Optimize the JEPA Room Grid

## Problem

The JEPAGrid (`sunset-ecosystem/nerve/room_grid.py`) runs 10K rooms in ~5ms via numpy einsum. Analysis of the code shows this is ~95% Python/numpy dispatch overhead, ~5% actual FLOPs.

**The math:**
- 10K rooms × 3 matmuls (64×32, 32×16, 16×16) = 60K matmuls
- Total FLOPs: 10K × (64×32 + 32×16 + 16×16) × 2 = 10K × 3328 = 33.28M FLOPs
- At 4GHz × 2 FMA = 133μs theoretical minimum
- Current: 5ms = 37.6× overhead factor

**Why einsum is slow here:** The three einsum calls (`"d,ndh->nh"`, `"nh,nhl->nl"`, `"nl,nll->nl"`) each invoke numpy's general-purpose contraction, which:
1. Parses the subscript string every call
2. Allocates temporaries for each intermediate
3. Cannot fuse the ReLU into the matmul
4. Has poor cache behavior because weights are stored as `(n, d, h)` — each room's weights are scattered across cache lines

## Ground-Level Code

### Current hot path (room_grid.py lines 100-112)

```python
def _batch_forward(self, x: np.ndarray) -> np.ndarray:
    h = np.einsum("d,ndh->nh", x, self.w["w1"]) + self.w["b1"][0]
    np.maximum(h, 0, out=h)
    h = np.einsum("nh,nhl->nl", h, self.w["w2"]) + self.w["b2"][0]
    np.maximum(h, 0, out=h)
    return np.einsum("nl,nll->nl", h, self.w["w3"]) + self.w["b3"][0]
```

### Memory layout problem

Current weight layout: `w1[n, d, h]` = `(10000, 64, 32)` = 81.92 MB for w1 alone.
All 3 weight tensors = ~150 MB. This exceeds L3 cache on most machines.

**Proposed packed layout** (room-major, cache-line aligned):

```c
// Each room's weights packed contiguously
typedef struct {
    float w1[64*32];    // 2048 floats = 8 KB
    float b1[32];       // 128 bytes
    float w2[32*16];    // 512 floats = 2 KB
    float b2[16];       // 64 bytes
    float w3[16*16];    // 256 floats = 1 KB
    float b3[16];       // 64 bytes
    // Total: ~11.3 KB per room
    // 10000 rooms = ~110 MB (fits in memory, 1 room = fits in L1)
} room_weights_t __attribute__((aligned(64)));
```

One room's weights = 11.3 KB — fits comfortably in L1 cache (32 KB typical). Sequential access means prefetcher can stay ahead.

### Proposed micro-kernel (C/AVX2)

Reference: `warp-room/src/warp-constraints.h` for INT8 saturated constraint patterns.

New file: `sunset-ecosystem/nerve/jepa_kernel.c`

```c
#include <immintrin.h>
#include <stdint.h>

// Room weights packed as above
typedef struct {
    float w1[2048]; float b1[32];
    float w2[512];  float b2[16];
    float w3[256];  float b3[16];
} room_weights_t __attribute__((aligned(64)));

// Fused matmul+ReLU for one room
// mat: (out_dim × in_dim) row-major, x: (in_dim,), bias: (out_dim,), out: (out_dim,)
static inline void fused_matmul_relu(
    const float *mat, int out_dim, int in_dim,
    const float *x, const float *bias, float *out
) {
    // Process 8 outputs at a time via AVX2
    for (int i = 0; i < out_dim; i += 8) {
        __m256 acc = _mm256_loadu_ps(&bias[i]);  // bias into accumulator
        for (int k = 0; k < in_dim; k++) {
            __m256 w = _mm256_loadu_ps(&mat[(i)*in_dim + k]); // row i, weight k
            // Wait — we need to broadcast x[k] and multiply
            __m256 xk = _mm256_set1_ps(x[k]);
            acc = _mm256_fmadd_ps(xk, w, acc);
        }
        // Fused ReLU: max(acc, 0)
        __m256 zero = _mm256_setzero_ps();
        acc = _mm256_max_ps(acc, zero);
        _mm256_storeu_ps(&out[i], acc);
    }
}

// Full 3-layer forward for N rooms, single input
void jepa_grid_forward(
    const room_weights_t *weights,  // (n,) packed
    const float *x,                 // (64,) input
    float *latents,                 // (n, 16) output
    int n                           // number of rooms
) {
    float h1[32] __attribute__((aligned(32)));
    float h2[16] __attribute__((aligned(32)));

    for (int i = 0; i < n; i++) {
        const room_weights_t *w = &weights[i];
        // Layer 1: (64→32) + ReLU
        fused_matmul_relu(w->w1, 32, 64, x, w->b1, h1);
        // Layer 2: (32→16) + ReLU
        fused_matmul_relu(w->w2, 16, 32, h1, w->b2, h2);
        // Layer 3: (16→16) linear (no ReLU)
        for (int j = 0; j < 16; j++) {
            float acc = w->b3[j];
            for (int k = 0; k < 16; k++) {
                acc += w->w3[j * 16 + k] * h2[k];
            }
            latents[i * 16 + j] = acc;
        }
    }
}
```

**Note on the matmul kernel:** The inner loop above processes 8 output rows simultaneously. For 64×32, this is 8 rows × 64 columns = 512 FMA ops. For 32×16, 8 rows × 32 cols = 256 FMA ops. The small dimensions mean loop overhead is minimal.

### Python binding

New file: `sunset-ecosystem/nerve/_jepa_kernel.pyx` (Cython) or use cffi:

```python
# _jepa_kernel.pyx
import numpy as np
cimport numpy as np

cdef extern from "jepa_kernel.c":
    void jepa_grid_forward(
        const room_weights_t *weights,
        const float *x, float *latents, int n)

def forward_batch(weights_packed, x_np):
    cdef float[64] x_buf
    cdef int n = len(weights_packed)
    out = np.empty((n, 16), dtype=np.float32)
    # ... pointer dance ...
    jepa_grid_forward(<room_weights_t*>weights_ptr, &x_buf[0], out_ptr, n)
    return out
```

### Weight packing utility

Add to `room_grid.py`:

```python
def pack_weights(self) -> np.ndarray:
    """Pack weights into cache-friendly room-major layout for C kernel."""
    n = self.n
    # 2048+32+512+16+256+16 = 2880 floats per room
    packed = np.empty((n, 2880), dtype=np.float32)
    for i in range(n):
        off = 0
        packed[i, off:off+2048] = self.w["w1"][i].ravel(); off += 2048
        packed[i, off:off+32] = self.w["b1"][0, i]; off += 32
        packed[i, off:off+512] = self.w["w2"][i].ravel(); off += 512
        packed[i, off:off+16] = self.w["b2"][0, i]; off += 16
        packed[i, off:off+256] = self.w["w3"][i].ravel(); off += 256
        packed[i, off:off+16] = self.w["b3"][0, i]; off += 16
    return np.ascontiguousarray(packed)
```

### Alternative: Rust+SIMD

If C/AVX2 is rejected for portability, use Rust with `std::simd` or `packed_simd2`:

Reference: `src/bma.rs` for tight loop patterns over arrays.

```rust
// jepa_kernel.rs
use std::arch::x86_64::*;

#[repr(C, align(64))]
pub struct RoomWeights {
    w1: [f32; 2048], b1: [f32; 32],
    w2: [f32; 512],  b2: [f32; 16],
    w3: [f32; 256],  b3: [f32; 16],
}

pub unsafe fn jepa_grid_forward(
    weights: &[RoomWeights],
    x: &[f32; 64],
    latents: &mut [f32],
) {
    for (i, w) in weights.iter().enumerate() {
        let mut h1 = [0.0f32; 32];
        let mut h2 = [0.0f32; 16];
        // Layer 1: matmul 64→32 + ReLU (same pattern as C version)
        for j in (0..32).step_by(8) {
            let mut acc = _mm256_loadu_ps(w.b1.as_ptr().add(j));
            for k in 0..64 {
                let xk = _mm256_set1_ps(x[k]);
                let wj = _mm256_loadu_ps(w.w1.as_ptr().add((j*64) + k*64));
                acc = _mm256_fmadd_ps(xk, wj, acc);
            }
            let zero = _mm256_setzero_ps();
            acc = _mm256_max_ps(acc, zero);
            _mm256_storeu_ps(h1.as_mut_ptr().add(j), acc);
        }
        // Layer 2: 32→16 + ReLU (similar)
        // Layer 3: 16→16 linear (similar)
        // ... copy to latents[i*16..(i+1)*16]
    }
}
```

### Benchmark spec

New file: `sunset-ecosystem/nerve/bench_grid.py`

```python
"""Benchmark: einsum vs C kernel."""
import time, numpy as np
from room_grid import JEPAGrid

def bench_einsum(n=10000, ticks=100):
    g = JEPAGrid(n)
    t0 = time.perf_counter()
    for _ in range(ticks):
        x = np.random.randn(64).astype(np.float32)
        g.tick(x)
    elapsed = time.perf_counter() - t0
    return elapsed / ticks  # ms per tick

def bench_kernel(n=10000, ticks=100):
    # After C kernel integration
    # g = JEPAGrid(n, kernel="c")
    pass

# Targets:
# einsum: 5ms/tick (current)
# C/AVX2: < 1ms/tick
# Theoretical: 133μs/tick
```

## Decision

**Use C/AVX2 with Python ctypes binding.** Not Rust — the kernel is 80 lines of C and the build complexity of adding Rust to a Python project isn't worth it. The C file compiles in 2 seconds and loads via `ctypes.CDLL`.

The packed weight layout is the key optimization. Each room's ~11 KB fits in L1. Sequential room iteration lets the hardware prefetcher work perfectly.

## Implementation Order

1. Write `jepa_kernel.c` with AVX2 fused matmul+ReLU
2. Add `Makefile` target: `gcc -O3 -mavx2 -mfma -shared -fPIC -o jepa_kernel.so jepa_kernel.c`
3. Write `bench_grid.py` — baseline einsum numbers
4. Add `pack_weights()` to `JEPAGrid` class
5. Add `_forward_c()` method that calls the shared library via ctypes
6. Benchmark: confirm < 1ms for 10K rooms
7. Make `_forward_c` the default, keep `_batch_forward` as fallback
8. Update `novelty()` computation to batch with the kernel output

## Success Criteria

- [ ] 10K rooms forward pass < 1ms (down from 5ms)
- [ ] Weight packing produces contiguous 11.3 KB per room
- [ ] ReLU fused into matmul (no separate max operation)
- [ ] Falls back to einsum if AVX2 unavailable
- [ ] `bench_grid.py` shows comparative numbers
- [ ] No numpy allocation in the hot path (pre-allocated output buffer)
- [ ] Thread-safe (same `threading.Lock` pattern as current `_lock`)
