import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True, parallel=True)
def _forward_einsum_numba_core_parallel(
    x, w1, b1, w2, b2, w3, b3, n,
):
    out = np.empty((n, 16), dtype=np.float32)
    for i in prange(n):
        # --- layer 1: x @ w1[i] + b1[i]  → h(32) ---
        h = np.empty(32, dtype=np.float32)
        for j in range(32):
            s = 0.0
            for k in range(64):
                s += x[k] * w1[i, k, j]
            h[j] = s + b1[i, j]
        # relu
        for j in range(32):
            if h[j] < 0.0:
                h[j] = 0.0
        # --- layer 2: h @ w2[i] + b2[i] → h2(16) ---
        h2 = np.empty(16, dtype=np.float32)
        for j in range(16):
            s = 0.0
            for k in range(32):
                s += h[k] * w2[i, k, j]
            h2[j] = s + b2[i, j]
        # relu
        for j in range(16):
            if h2[j] < 0.0:
                h2[j] = 0.0
        # --- layer 3: h2 @ w3[i] + b3[i] → out[i](16) ---
        for j in range(16):
            s = 0.0
            for k in range(16):
                s += h2[k] * w3[i, k, j]
            out[i, j] = s + b3[i, j]
    return out

if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    w1 = np.random.randn(n, 64, 32).astype(np.float32) * 0.01
    b1 = np.zeros((n, 32), dtype=np.float32)
    w2 = np.random.randn(n, 32, 16).astype(np.float32) * 0.01
    b2 = np.zeros((n, 16), dtype=np.float32)
    w3 = np.broadcast_to(np.eye(16, dtype=np.float32) * 0.99, (n, 16, 16)).copy()
    b3 = np.zeros((n, 16), dtype=np.float32)
    x = np.random.randn(64).astype(np.float32)

    # Warmup
    r = _forward_einsum_numba_core_parallel(x, w1, b1, w2, b2, w3, b3, n)
    print("warmup shape:", r.shape)

    import time
    trials = 50
    t0 = time.perf_counter()
    for _ in range(trials):
        _forward_einsum_numba_core_parallel(x, w1, b1, w2, b2, w3, b3, n)
    t_numba = (time.perf_counter() - t0) * 1000

    # numpy einsum baseline
    from nerve.room_grid import forward_einsum
    w = {"w1": w1, "b1": b1[np.newaxis], "w2": w2, "b2": b2[np.newaxis],
         "w3": w3, "b3": b3[np.newaxis]}
    t0 = time.perf_counter()
    for _ in range(trials):
        forward_einsum(w, x)
    t_numpy = (time.perf_counter() - t0) * 1000

    print(f"numba parallel: {t_numba/trials:.3f} ms/tick")
    print(f"numpy einsum:   {t_numpy/trials:.3f} ms/tick")
    print(f"speedup:        {t_numpy/t_numba:.2f}x")
    # correctness
    expected = forward_einsum(w, x)
    actual = _forward_einsum_numba_core_parallel(x, w1, b1, w2, b2, w3, b3, n)
    diff = np.max(np.abs(expected - actual))
    print(f"max diff: {diff:.2e}")
