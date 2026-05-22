import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def test_maximum2():
    a = np.ones((10, 16), dtype=np.float32) * (-2)
    b = np.zeros((10, 16), dtype=np.float32)
    return np.maximum(a, b)

@njit(cache=True, fastmath=True)
def test_relu_inplace():
    a = np.ones((10, 16), dtype=np.float32) * (-2)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] < 0:
                a[i, j] = 0.0
    return a

if __name__ == "__main__":
    try:
        r = test_maximum2()
        print("maximum2 works:", r.sum())
    except Exception as e:
        print("maximum2 failed:", type(e).__name__, e)
    try:
        r = test_relu_inplace()
        print("relu_inplace works:", r.sum())
    except Exception as e:
        print("relu_inplace failed:", type(e).__name__, e)
