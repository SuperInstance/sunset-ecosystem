# SPEC-JEPA-GRID-OPTIMIZATION.md — Optimize the JEPA Room Grid

## Status: PARTIALLY SHIPPED

The Rust kernel (`nerve/src/lib.rs`) already ships with multi-threaded batch forward, achieving ~2.35ms for 10K rooms. This spec tracks the remaining work: Python binding, weight packing, and the novelty computation bottleneck.

## What Shipped (nerve/src/lib.rs)

The Rust `jepa-kernel` crate (`nerve/src/Cargo.toml`, `cdylib` target):

- `forward_room()`: 3-layer MLP (64→32→ReLU→16→ReLU→16), row-outer loop with zero-skip
- `jepa_forward_batch()`: Multi-threaded via `std::thread::scope`, splits rooms across up to 12 cores
- Weight layout: flat arrays indexed by `ri * dim_product` (room-major, interleaved)
- Benchmark: 10K rooms × 50 passes, reports ns/room
- Exports as `extern "C" fn` for ctypes/FFI binding

**Remaining gap:** The Rust kernel only does forward. It doesn't compute novelty, doesn't update activity counters, doesn't manage the history ring buffer. Python still owns those. The binding layer needs to ship.

## Problem

The JEPAGrid (`nerve/room_grid.py`) runs 10K rooms in ~5ms via numpy einsum. The Rust kernel brings this to ~2.35ms. But the Python `_batch_forward()` is still the default path — no FFI binding exists yet.

**Remaining bottlenecks after Rust forward:**
1. **Python binding missing** — `_batch_forward()` still uses einsum
2. **Novelty computation is O(n × history)** — cosine distance against last 3 latents per room, done in a Python loop
3. **History ring buffer** — `dict[int, list[ndarray]]` with manual `.pop(0)` is O(n) shift

## Ground-Level Code

### Shipped: Rust kernel (`nerve/src/lib.rs`)

The Rust kernel already exists and benchmarks at ~2.35ms for 10K rooms.

**Weight layout** (room-major flat arrays):
```
w1: [n × 64 × 32]  — room i at offset i*2048
w2: [n × 32 × 16]  — room i at offset i*512
w3: [n × 16 × 16]  — room i at offset i*256
b1: [n × 32]       — room i at offset i*32
b2: [n × 16]       — room i at offset i*16
b3: [n × 16]       — room i at offset i*16
```

This IS the room-major packed layout — each room's weights are contiguous.
One room: 2048+32+512+16+256+16 = 2880 floats = 11.25 KB. Fits in L1.

**Threading**: `std::thread::scope` with `available_parallelism().min(12)`. Falls back to single-thread for n < 100.

**Zero-skip optimization**: `if xr == 0.0 { continue; }` — skips zero inputs after ReLU kills negatives. For ReLU with ~50% sparsity, this saves ~50% of multiply ops in layers 2-3.

### Remaining: Python ctypes binding

New file: `sunset-ecosystem/nerve/rust_kernel.py`

```python
"""Python binding to the Rust jepa-kernel shared library.

Build: cd nerve/src && cargo build --release
       → target/release/libjepa_kernel.so
"""
import ctypes
import numpy as np
from pathlib import Path

_LIB_PATH = Path(__file__).parent / "src" / "target" / "release" / "libjepa_kernel.so"
_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.jepa_forward_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x_ptr
        ctypes.POINTER(ctypes.c_float),  # w1
        ctypes.POINTER(ctypes.c_float),  # w2
        ctypes.POINTER(ctypes.c_float),  # w3
        ctypes.POINTER(ctypes.c_float),  # b1
        ctypes.POINTER(ctypes.c_float),  # b2
        ctypes.POINTER(ctypes.c_float),  # b3
        ctypes.c_size_t,                 # n
        ctypes.POINTER(ctypes.c_float),  # out_ptr
    ]
    _lib.jepa_forward_batch.restype = None
    return _lib

def forward_batch(w: dict, x: np.ndarray, n: int) -> np.ndarray:
    """Call Rust kernel for all rooms.

    Args:
        w: weight dict from make_weights (arrays must be contiguous float32)
        x: (64,) input signal
        n: number of rooms

    Returns:
        (n, 16) latent array
    """
    lib = _load()
    x = np.ascontiguousarray(x.ravel(), dtype=np.float32)
    out = np.empty((n, 16), dtype=np.float32)

    # Flatten weight arrays to match Rust's expected layout
    w1 = np.ascontiguousarray(w["w1"].reshape(n, -1).ravel(), dtype=np.float32)
    w2 = np.ascontiguousarray(w["w2"].reshape(n, -1).ravel(), dtype=np.float32)
    w3 = np.ascontiguousarray(w["w3"].reshape(n, -1).ravel(), dtype=np.float32)
    b1 = np.ascontiguousarray(w["b1"].reshape(n, -1).ravel(), dtype=np.float32)
    b2 = np.ascontiguousarray(w["b2"].reshape(n, -1).ravel(), dtype=np.float32)
    b3 = np.ascontiguousarray(w["b3"].reshape(n, -1).ravel(), dtype=np.float32)

    lib.jepa_forward_batch(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out
```

### Remaining: Batch novelty computation

Current `novelty()` in `room_grid.py` is per-room, called in a Python loop. This is the remaining bottleneck after Rust forward.

Replace with vectorized batch novelty:

```python
def batch_novelty(latents: np.ndarray, history: dict[int, list[np.ndarray]]) -> np.ndarray:
    """Vectorized novelty for all rooms.

    Args:
        latents: (n, 16) current latents
        history: room_id → list of past latents (up to 3)

    Returns:
        (n,) novelty scores [0, 1]
    """
    n = latents.shape[0]
    norms = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    zn = latents / norms  # (n, 16) normalized

    # Build padded history tensor: (n, 3, 16)
    hist_tensor = np.zeros((n, 3, 16), dtype=np.float32)
    hist_mask = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        h = history.get(i, [])
        for j, z in enumerate(h[-3:]):
            hist_tensor[i, j] = z
            hist_mask[i, j] = 1.0

    # Cosine similarity: (n, 3) = sum(zn[h,i] * hist[n,i], axis=-1)
    h_norms = np.linalg.norm(hist_tensor, axis=-1, keepdims=True) + 1e-8  # (n, 3, 1)
    hn = hist_tensor / h_norms  # (n, 3, 16)
    sims = (zn[:, np.newaxis, :] * hn).sum(axis=-1)  # (n, 3)

    # Masked mean
    mask_sum = hist_mask.sum(axis=1, keepdims=True) + 1e-8
    mean_sim = (sims * hist_mask).sum(axis=1, keepdims=True) / mask_sum  # (n, 1)
    novelty = 1.0 - mean_sim.ravel()  # (n,)

    # Where no history, return 0.5
    no_hist = hist_mask.sum(axis=1) < 2
    novelty[no_hist] = 0.5

    return novelty
```

### Remaining: History ring buffer replacement

Current: `dict[int, list[ndarray]]` with `pop(0)` — O(n) shift per tick.
Replace with `collections.deque(maxlen=20)` or pre-allocated numpy ring.

```python
from collections import deque

# In JEPAGrid.__init__:
self.history: dict[int, deque[np.ndarray]] = {
    i: deque(maxlen=20) for i in range(n)
}

# In tick():
self.history[i].append(z.copy())  # O(1), auto-evicts oldest
```

### Integration into JEPAGrid

Add to `room_grid.py`:

```python
class JEPAGrid:
    def __init__(self, n=250, ..., kernel: str = "auto"):
        ...
        self._kernel = kernel
        self._rust_available = False
        if kernel in ("auto", "rust"):
            try:
                from nerve.rust_kernel import forward_batch as _rust_forward
                self._rust_forward = _rust_forward
                self._rust_available = True
            except (ImportError, OSError):
                self._rust_available = False

    def _batch_forward(self, x):
        if self._rust_available:
            return self._rust_forward(self.w, x, self.n)
        # Fallback: numpy einsum
        h = np.einsum("d,ndh->nh", x, self.w["w1"]) + self.w["b1"][0]
        np.maximum(h, 0, out=h)
        h = np.einsum("nh,nhl->nl", h, self.w["w2"]) + self.w["b2"][0]
        np.maximum(h, 0, out=h)
        return np.einsum("nl,nll->nl", h, self.w["w3"]) + self.w["b3"][0]
```

## Decision

**Rust kernel IS the forward path.** It already shipped at `nerve/src/lib.rs`. The remaining work is:

1. Python ctypes binding (`nerve/rust_kernel.py`)
2. Wire into `JEPAGrid._batch_forward()` with auto-detection + einsum fallback
3. Vectorize the novelty computation (currently Python-loop bottleneck)
4. Replace `list.pop(0)` with `deque(maxlen=20)` for O(1) history
5. Pre-allocate the output buffer to avoid per-tick allocation

No C/AVX2 alternative needed — the Rust kernel already does row-outer loops with zero-skip and multi-threading. Adding explicit SIMD intrinsics is future work after the binding ships.

## Implementation Order

1. `cd nerve/src && cargo build --release` — verify .so builds
2. Write `nerve/rust_kernel.py` — ctypes binding
3. Add `kernel="auto"` param to `JEPAGrid.__init__`, wire `_rust_forward`
4. Write `nerve/bench_grid.py` — compare einsum vs Rust
5. Replace `history[i].pop(0)` with `deque(maxlen=20)` — O(1) fix
6. Write vectorized `batch_novelty()` — eliminate Python loop
7. Pre-allocate output buffer in `JEPAGrid.__init__` — zero allocation in tick
8. Benchmark end-to-end tick: target < 3ms for 10K rooms (forward + novelty + history)

## Success Criteria

- [ ] `cargo build --release` produces `libjepa_kernel.so`
- [ ] `rust_kernel.py` loads and calls the Rust kernel via ctypes
- [ ] `JEPAGrid(kernel="auto")` uses Rust when available, einsum otherwise
- [ ] 10K rooms forward pass < 2.5ms (Rust kernel, matching current bench)
- [ ] Full tick (forward + novelty + history) < 3ms for 10K rooms
- [ ] History ring buffer is O(1) append (deque, no list.pop(0))
- [ ] No numpy allocation in the hot path after warmup
- [ ] `bench_grid.py` prints comparative einsum vs Rust timings
- [ ] Falls back gracefully if .so not found (no crash, just slower)
