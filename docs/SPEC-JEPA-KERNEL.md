# SPEC-JEPA-KERNEL.md
**Author:** CCC (Systems Architect)  
**Date:** 2026-05-21  
**Status:** ARCHITECTURE — Next steps for the Rust JEPA kernel

---

## 1. Current State

The Rust JEPA kernel in `nerve/src/lib.rs` achieves **2.35ms for 10K rooms** (235ns/room). This is the `jepa_forward_batch` function — a 3-layer MLP (64→32→16→16) with ReLU, multi-threaded via `std::thread::scope`.

The Python fallback in `nerve/room_grid.py` does the same with numpy einsum at ~3-5ms for 10K rooms.

### Architecture of one room
```
Input (64) → Linear(64,32) + ReLU → Linear(32,16) + ReLU → Linear(16,16) → latent (16)
```
Parameters per room: 64×32 + 32 + 32×16 + 16 + 16×16 + 16 = **2,560 params** (~10KB fp32)

### Current parallelism strategy
- Rooms are independent (same input, different weights)
- Split across `min(available_cores, 12)` threads
- Each thread processes a contiguous chunk of rooms
- Cache-friendly: weights accessed sequentially per room

## 2. The Question: What's Next?

Three paths, in order of impact and difficulty:

### Path A: Python Drop-in Replacement (EASIEST — do first)

Wire the Rust kernel into `room_grid.py` as a drop-in for `_batch_forward`.

**Mechanism:** `cdylib` + `ctypes`

The Rust crate already compiles to a C dynamic library (`#[no_mangle] pub extern "C"`). Python calls it via ctypes:

```python
# room_grid.py — Rust-accelerated path
import ctypes
import os
import numpy as np

_LIB = None


def _load_rust():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = os.path.join(
        os.path.dirname(__file__), "../../target/release/libnerve.so"
    )
    if not os.path.exists(lib_path):
        return None
    _LIB = ctypes.CDLL(lib_path)
    _LIB.jpa_forward_batch.restype = None
    _LIB.jpa_forward_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x_ptr
        ctypes.POINTER(ctypes.c_float),  # w1
        ctypes.POINTER(ctypes.c_float),  # w2
        ctypes.POINTER(ctypes.c_float),  # w3
        ctypes.POINTER(ctypes.c_float),  # b1
        ctypes.POINTER(ctypes.c_float),  # b2
        ctypes.POINTER(ctypes.c_float),  # b3
        ctypes.c_size_t,  # n
        ctypes.POINTER(ctypes.c_float),  # out_ptr
    ]
    return _LIB


class JEPAGrid:
    def _batch_forward(self, x: np.ndarray) -> np.ndarray:
        lib = _load_rust()
        if lib is None:
            return self._batch_forward_numpy(x)  # fallback

        n = self.n
        w = self.w
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros(n * 16, dtype=np.float32)

        lib.jpa_forward_batch(
            x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["w1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["w2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["w3"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["b1"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["b2"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w["b3"].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out.reshape(n, 16)
```

**Critical detail:** The Rust kernel expects weights in room-major layout: `w1[n * 64 * 32]` where room i's weights start at `i * 64 * 32`. The Python `make_weights` stores as `w1[n, 64, 32]` — which is contiguous in the same order. **Layouts match.** No transposition needed.

However: the Rust kernel flattens biases to `b1[n * 32]` but Python stores as `b1[1, n, 32]`. Need to reshape:

```python
b1_flat = w["b1"].reshape(-1)  # (1, n, 32) → (n * 32) — correct if contiguous
```

Python biases are initialized as `np.zeros((1, n, h))` — the `.reshape(-1)` produces `n * h` zeros in room-major order. **This matches Rust.**

**Expected speedup:** 2-3x over numpy einsum (2.35ms vs ~5ms for 10K rooms).

### Path B: GPU via plato_forge CUDA kernels (MEDIUM — do second)

The `plato_forge/forge_kernels.cu` infrastructure provides:
- Eisenstein weight snapping (lattice quantization)
- Deadband gradient throttle with HPDF dither
- BMA convergence detection
- Weight compression for tile export
- Pipeline double-buffer copy

**Adaptation for JEPA rooms:**

Each room's forward pass is `x @ W1 + b1 → ReLU → @W2 + b2 → ReLU → @W3 + b3`. This is a batched matrix multiply with per-room weights — essentially a grouped convolution.

```cuda
// Conceptual: one block per room, all rooms in parallel
__global__ void jepa_forward_kernel(
    const float* x,          // (64,) — shared input
    const float* w1,         // (n, 64, 32)
    const float* w2,         // (n, 32, 16)  
    const float* w3,         // (n, 16, 16)
    const float* b1,         // (n, 32)
    const float* b2,         // (n, 16)
    const float* b3,         // (n, 16)
    float* out,              // (n, 16)
    int n
) {
    int room = blockIdx.x;
    if (room >= n) return;
    
    // Each thread handles a subset of output neurons
    // Layer 1: 64→32
    extern __shared__ float h32[];
    for (int col = threadIdx.x; col < 32; col += blockDim.x) {
        float sum = b1[room * 32 + col];
        for (int row = 0; row < 64; row++) {
            sum += x[row] * w1[room * 64 * 32 + row * 32 + col];
        }
        h32[col] = fmaxf(0.0f, sum);
    }
    __syncthreads();
    
    // Layer 2: 32→16
    // ... similar pattern
    // Layer 3: 16→16
    // ... write to out[room * 16 + col]
}
```

**Expected performance:** With 10K rooms × 2.5K params each = 25M params, all fitting in ~100MB VRAM:
- RTX 4050 (8GB, sm_89): ~0.1-0.3ms for 10K rooms (10-30x over CPU Rust)
- Single wave of 10K blocks, each block has 32 threads — 320K threads total

**Integration path:**
1. Add `jepa_forward_kernel` to `forge_kernels.cu`
2. Expose C API: `jepa_forward_gpu(x, w1, w2, w3, b1, b2, b3, n, out)`
3. Python ctypes wrapper (same pattern as Rust path)
4. Auto-select GPU if CUDA available, else Rust, else numpy

### Path C: Chapel-level Multi-Node (HARD — future)

For 100K+ rooms across multiple machines. Chapel's locale model maps well:
- Each locale owns a slice of rooms
- Input `x` is broadcast to all locales
- Each locale runs its rooms independently
- Results gathered to locale 0

This is *not* needed until room counts exceed single-machine capacity. The current 65-agent thermal budget limits practical rooms to ~10K. Skip until needed.

## 3. Recommended Implementation Order

```
Week 1: Path A — ctypes Rust drop-in
         └── Modify room_grid.py to detect and load libnerve.so
         └── Add _batch_forward_numpy as fallback
         └── Benchmark: expect 2-3x over pure numpy

Week 2: Path B — CUDA kernel in plato_forge
         └── Add jepa_forward_kernel to forge_kernels.cu
         └── Build pipeline: compile → ctypes → auto-detect GPU
         └── Benchmark: expect 10-30x over CPU Rust

Later:   Path C — Chapel multi-node
         └── Only if room count exceeds single-machine limits
```

## 4. Weight Layout Contract

This is the critical interface between Rust/CUDA/Python:

```
w1: [n * 64 * 32]  — room-major, row-major within each room
w2: [n * 32 * 16]
w3: [n * 16 * 16]
b1: [n * 32]       — one bias vector per room
b2: [n * 16]
b3: [n * 16]
x:  [64]           — shared input signal
out: [n * 16]      — one latent per room
```

All arrays are `f32`, contiguous, no padding. This is the ABI that all three backends (numpy, Rust, CUDA) must agree on.

## 5. The Backprop Question

The Rust kernel only does forward passes. For learning (room adaptation), we need backward passes too. Options:

1. **No backprop** — rooms are initialized randomly, selected by tournament, and rebirth-ed when cold. This is the current `JEPAGrid.rebirth()` approach. No gradients needed.
2. **Hebbian update** — rooms that fire together strengthen their weights. Simple outer product update, no backward pass needed. Could be added to the Rust kernel.
3. **Full backprop** — if rooms need to learn specific patterns. Requires a `jepa_backward_batch` kernel.

**Recommendation:** Stick with option 1 (rebirth-only) for now. The ecosystem's COLLECT→SELECT→COMPILE grammar handles adaptation through selection (tournament) and compilation (sunset), not through gradient descent on individual rooms.
