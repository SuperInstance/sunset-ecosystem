import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def test_dot():
    a = np.ones((100, 64), dtype=np.float32)
    b = np.ones(64, dtype=np.float32)
    return np.dot(b, a.T)  # (64,) dot (64, 100) -> (100,)

@njit(cache=True, fastmath=True)
def test_maximum():
    a = np.ones((10, 16), dtype=np.float32) * (-2)
    np.maximum(a, 0, out=a)
    return a

@njit(cache=True, fastmath=True)
def test_manual_loop(n=100):
    out = np.empty((n, 16), dtype=np.float32)
    w1 = np.ones((n, 64, 32), dtype=np.float32) * 0.01
    b1 = np.zeros((n, 32), dtype=np.float32)
    w2 = np.ones((n, 32, 16), dtype=np.float32) * 0.01
    b2 = np.zeros((n, 16), dtype=np.float32)
    w3 = np.ones((n, 16, 16), dtype=np.float32) * 0.99
    b3 = np.zeros((n, 16), dtype=np.float32)
    x = np.ones(64, dtype=np.float32)

    for i in range(n):
        # layer 1: x(64) @ w1[i](64,32) -> h(32)
        h = np.empty(32, dtype=np.float32)
        for j in range(32):
            s = 0.0
            for k in range(64):
                s += x[k] * w1[i, k, j]
            h[j] = s + b1[i, j]
        # relu
        for j in range(32):
            if h[j] < 0:
                h[j] = 0.0
        # layer 2: h(32) @ w2[i](32,16) -> h2(16)
        h2 = np.empty(16, dtype=np.float32)
        for j in range(16):
            s = 0.0
            for k in range(32):
                s += h[k] * w2[i, k, j]
            h2[j] = s + b2[i, j]
        # relu
        for j in range(16):
            if h2[j] < 0:
                h2[j] = 0.0
        # layer 3: h2(16) @ w3[i](16,16) -> out[i](16)
        for j in range(16):
            s = 0.0
            for k in range(16):
                s += h2[k] * w3[i, k, j]
            out[i, j] = s + b3[i, j]
    return out

if __name__ == "__main__":
    try:
        r = test_dot()
        print("dot works:", r.shape)
    except Exception as e:
        print("dot failed:", type(e).__name__, e)
    try:
        r = test_maximum()
        print("maximum works:", r.sum())
    except Exception as e:
        print("maximum failed:", type(e).__name__, e)
    try:
        r = test_manual_loop(100)
        print("manual_loop works:", r.shape, r.sum())
    except Exception as e:
        print("manual_loop failed:", type(e).__name__, e)
