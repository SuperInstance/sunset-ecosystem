import numpy as np
from numba import njit, prange
from nerve.room_grid import forward_einsum
import time

@njit(cache=True, fastmath=True, parallel=True)
def _forward_einsum_numba_transposed(
    x, w1_t, b1, w2_t, b2, w3_t, b3, n,
):
    """w1_t is (n, 32, 64) so inner loop over k is contiguous."""
    out = np.empty((n, 16), dtype=np.float32)
    for i in prange(n):
        h = np.empty(32, dtype=np.float32)
        for j in range(32):
            s = 0.0
            for k in range(64):
                s += x[k] * w1_t[i, j, k]
            h[j] = s + b1[i, j]
        for j in range(32):
            if h[j] < 0.0:
                h[j] = 0.0
        h2 = np.empty(16, dtype=np.float32)
        for j in range(16):
            s = 0.0
            for k in range(32):
                s += h[k] * w2_t[i, j, k]
            h2[j] = s + b2[i, j]
        for j in range(16):
            if h2[j] < 0.0:
                h2[j] = 0.0
        for j in range(16):
            s = 0.0
            for k in range(16):
                s += h2[k] * w3_t[i, j, k]
            out[i, j] = s + b3[i, j]
    return out

def _compiled_forward_transposed(w, x):
    xflat = x.ravel().astype(np.float32)
    n = w["w1"].shape[0]
    b1 = w["b1"][0] if w["b1"].ndim == 3 else w["b1"]
    b2 = w["b2"][0] if w["b2"].ndim == 3 else w["b2"]
    b3 = w["b3"][0] if w["b3"].ndim == 3 else w["b3"]
    # Pre-transpose weights ONCE
    w1_t = np.ascontiguousarray(w["w1"].transpose(0, 2, 1))
    w2_t = np.ascontiguousarray(w["w2"].transpose(0, 2, 1))
    w3_t = np.ascontiguousarray(w["w3"].transpose(0, 2, 1))
    return _forward_einsum_numba_transposed(xflat, w1_t, b1, w2_t, b2, w3_t, b3, n)

for n in [100, 250, 500, 1000]:
    np.random.seed(42)
    w = {
        "w1": np.random.randn(n, 64, 32).astype(np.float32) * 0.01,
        "b1": np.zeros((1, n, 32), dtype=np.float32),
        "w2": np.random.randn(n, 32, 16).astype(np.float32) * 0.01,
        "b2": np.zeros((1, n, 16), dtype=np.float32),
        "w3": np.broadcast_to(np.eye(16, dtype=np.float32) * 0.99, (n, 16, 16)).copy(),
        "b3": np.zeros((1, n, 16), dtype=np.float32),
    }
    x = np.random.randn(64).astype(np.float32)
    _compiled_forward_transposed(w, x)  # warmup
    forward_einsum(w, x)  # warmup
    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        _compiled_forward_transposed(w, x)
    t_numba = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in range(trials):
        forward_einsum(w, x)
    t_numpy = (time.perf_counter() - t0) * 1000
    print(f"n={n:4d}  numba_transposed={t_numba/trials:.3f}ms  numpy={t_numpy/trials:.3f}ms  speedup={t_numpy/t_numba:.2f}x")
    expected = forward_einsum(w, x)
    actual = _compiled_forward_transposed(w, x)
    diff = np.max(np.abs(expected - actual))
    print(f"  diff={diff:.2e}")
