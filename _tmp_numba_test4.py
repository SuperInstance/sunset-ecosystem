import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def test_norm():
    a = np.ones((10, 16), dtype=np.float32)
    return np.linalg.norm(a, axis=1)

@njit(cache=True, fastmath=True)
def test_where():
    a = np.ones(10, dtype=np.float32)
    return np.where(a > 0.5, 1.0, 0.0)

@njit(cache=True, fastmath=True)
def test_random():
    return np.random.random(10)

if __name__ == "__main__":
    for fn in [test_norm, test_where, test_random]:
        try:
            r = fn()
            print(f"{fn.__name__} works: shape={r.shape if hasattr(r, 'shape') else 'scalar'}")
        except Exception as e:
            print(f"{fn.__name__} failed: {type(e).__name__} {e}")
