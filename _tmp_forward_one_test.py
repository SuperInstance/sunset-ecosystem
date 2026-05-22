import numpy as np
from numba import njit
from nerve.room_grid import forward_one
import time

@njit(cache=True, fastmath=True)
def _forward_one_numba_core(x, w1, b1, w2, b2, w3, b3):
    # layer 1
    h = np.empty(32, dtype=np.float32)
    for j in range(32):
        s = 0.0
        for k in range(64):
            s += x[k] * w1[k, j]
        h[j] = s + b1[j]
    for j in range(32):
        if h[j] < 0.0:
            h[j] = 0.0
    # layer 2
    h2 = np.empty(16, dtype=np.float32)
    for j in range(16):
        s = 0.0
        for k in range(32):
            s += h[k] * w2[k, j]
        h2[j] = s + b2[j]
    for j in range(16):
        if h2[j] < 0.0:
            h2[j] = 0.0
    # layer 3
    out = np.empty(16, dtype=np.float32)
    for j in range(16):
        s = 0.0
        for k in range(16):
            s += h2[k] * w3[k, j]
        out[j] = s + b3[j]
    return out

def _compiled_forward_one(w, x):
    xflat = x.ravel().astype(np.float32)
    return _forward_one_numba_core(xflat, w["w1"], w["b1"], w["w2"], w["b2"], w["w3"], w["b3"])

# Test single-room forward
n = 1
np.random.seed(42)
w = {
    "w1": np.random.randn(64, 32).astype(np.float32) * 0.01,
    "b1": np.zeros(32, dtype=np.float32),
    "w2": np.random.randn(32, 16).astype(np.float32) * 0.01,
    "b2": np.zeros(16, dtype=np.float32),
    "w3": np.eye(16, dtype=np.float32) * 0.99,
    "b3": np.zeros(16, dtype=np.float32),
}
x = np.random.randn(64).astype(np.float32)

_compiled_forward_one(w, x)  # warmup
forward_one(w, x)  # warmup

trials = 50
t0 = time.perf_counter()
for _ in range(trials):
    _compiled_forward_one(w, x)
t_numba = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
for _ in range(trials):
    forward_one(w, x)
t_numpy = (time.perf_counter() - t0) * 1000

print(f"numba={t_numba/trials:.3f}ms  numpy={t_numpy/trials:.3f}ms  speedup={t_numpy/t_numba:.2f}x")
expected = forward_one(w, x)
actual = _compiled_forward_one(w, x)
diff = np.max(np.abs(expected - actual))
print(f"diff={diff:.2e}")
