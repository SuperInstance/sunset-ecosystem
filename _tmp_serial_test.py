import numpy as np
from numba import njit
from nerve.room_grid import forward_einsum
import time

@njit(cache=True, fastmath=True)
def _forward_einsum_numba_core_serial(
    x, w1, b1, w2, b2, w3, b3, n,
):
    out = np.empty((n, 16), dtype=np.float32)
    for i in range(n):
        h = np.empty(32, dtype=np.float32)
        for j in range(32):
            s = 0.0
            for k in range(64):
                s += x[k] * w1[i, k, j]
            h[j] = s + b1[i, j]
        for j in range(32):
            if h[j] < 0.0:
                h[j] = 0.0
        h2 = np.empty(16, dtype=np.float32)
        for j in range(16):
            s = 0.0
            for k in range(32):
                s += h[k] * w2[i, k, j]
            h2[j] = s + b2[i, j]
        for j in range(16):
            if h2[j] < 0.0:
                h2[j] = 0.0
        for j in range(16):
            s = 0.0
            for k in range(16):
                s += h2[k] * w3[i, k, j]
            out[i, j] = s + b3[i, j]
    return out

def _compiled_forward(w, x):
    xflat = x.ravel().astype(np.float32)
    n = w["w1"].shape[0]
    b1 = w["b1"][0] if w["b1"].ndim == 3 else w["b1"]
    b2 = w["b2"][0] if w["b2"].ndim == 3 else w["b2"]
    b3 = w["b3"][0] if w["b3"].ndim == 3 else w["b3"]
    return _forward_einsum_numba_core_serial(xflat, w["w1"], b1, w["w2"], b2, w["w3"], b3, n)

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
    _compiled_forward(w, x)  # warmup
    forward_einsum(w, x)  # warmup
    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        _compiled_forward(w, x)
    t_numba = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in range(trials):
        forward_einsum(w, x)
    t_numpy = (time.perf_counter() - t0) * 1000
    print(f"n={n:4d}  numba_serial={t_numba/trials:.3f}ms  numpy={t_numpy/trials:.3f}ms  speedup={t_numpy/t_numba:.2f}x")
