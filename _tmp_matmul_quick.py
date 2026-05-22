import numpy as np
n = 10
w1 = np.random.randn(n, 64, 32).astype(np.float32)
x = np.random.randn(64).astype(np.float32)
try:
    result = np.matmul(x, w1)
    print("matmul worked, shape:", result.shape)
except Exception as e:
    print("matmul failed:", e)
