import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def test_einsum():
    a = np.ones((10, 64, 32), dtype=np.float32)
    b = np.ones(64, dtype=np.float32)
    return np.einsum('d,ndh->nh', b, a)

@njit(cache=True, fastmath=True)
def test_matmul():
    a = np.ones((100, 64), dtype=np.float32)
    b = np.ones((64, 32), dtype=np.float32)
    return a @ b

if __name__ == "__main__":
    try:
        r = test_einsum()
        print("einsum works:", r.shape)
    except Exception as e:
        print("einsum failed:", type(e).__name__, e)
    try:
        r = test_matmul()
        print("matmul works:", r.shape)
    except Exception as e:
        print("matmul failed:", type(e).__name__, e)
