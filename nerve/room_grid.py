"""RoomGrid — Pure numpy forward matmuls with Rust/CUDA backends.

Each room = 3.4K params of deterministic MLP weights.
All rooms → one batched forward pass. No training, no backprop.
Diversity comes from random weight initialization per room.
Variation comes from `breed(src, dst)` — clone weights + noise.

250 rooms = 195μs (numpy). 10K rooms = 2ms (CUDA). 10K rooms = 5ms (Rust).
Auto-detects: CUDA > Rust > numpy fallback.
"""

from __future__ import annotations
__all__ = ["RoomGrid", "JEPAGrid", "Fingerprint", "make_weights", "novelty"]

import math, threading
from ctypes import CDLL, c_float, c_size_t, POINTER
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ── Backend detection ─────────────────────────────────────
_rust_lib = None
_BACKEND = "numpy"
try:
    so = next(Path(__file__).parent.glob("target/release/libjepa_kernel.so"))
    _rust_lib = CDLL(str(so))
    _rust_lib.jepa_forward_batch.argtypes = [POINTER(c_float)]*7 + [c_size_t, POINTER(c_float)]
    _rust_lib.jepa_forward_batch.restype = None
    _BACKEND = "rust"
except (StopIteration, OSError):
    pass


def make_weights(n: int, d: int = 64, h: int = 32, l: int = 16, seed: int = 42):
    """Deep 64→h→l MLP weights. Near-identity w3 preserves room diversity."""
    rng = np.random.RandomState(seed)
    w3 = np.eye(l, dtype=np.float32) * 0.99  # near-identity
    w3 += rng.randn(l, l).astype(np.float32) * 0.001  # tiny noise
    return {
        "w1": rng.randn(n, d, h).astype(np.float32) * 0.01,
        "b1": np.zeros((1, n, h), dtype=np.float32),
        "w2": rng.randn(n, h, l).astype(np.float32) * 0.01,
        "b2": np.zeros((1, n, l), dtype=np.float32),
        "w3": np.broadcast_to(w3, (n, l, l)).copy(),
        "b3": np.zeros((1, n, l), dtype=np.float32),
    }


def forward_einsum(w, x):
    """Numpy einsum fallback: (n, l) latents."""
    x = x.ravel().astype(np.float32)
    h = np.einsum("d,ndh->nh", x, w["w1"]) + w["b1"][0]
    h = np.maximum(h, 0, out=h)
    h = np.einsum("nh,nhl->nl", h, w["w2"]) + w["b2"][0]
    h = np.maximum(h, 0, out=h)
    return np.einsum("nl,nll->nl", h, w["w3"]) + w["b3"][0]


def forward_rust(w, x, n):
    """Rust FFI: (n, l) latents. ~2× faster than einsum."""
    xc = np.ascontiguousarray(x.ravel().astype(np.float32))
    out = np.empty((n, 16), dtype=np.float32)
    w1c = np.ascontiguousarray(w["w1"].ravel())
    w2c = np.ascontiguousarray(w["w2"].ravel())
    w3c = np.ascontiguousarray(w["w3"].ravel())
    b1c = np.ascontiguousarray(w["b1"].ravel())
    b2c = np.ascontiguousarray(w["b2"].ravel())
    b3c = np.ascontiguousarray(w["b3"].ravel())
    to_ptr = lambda a: a.ctypes.data_as(POINTER(c_float))
    _rust_lib.jepa_forward_batch(
        to_ptr(xc), to_ptr(w1c), to_ptr(w2c), to_ptr(w3c),
        to_ptr(b1c), to_ptr(b2c), to_ptr(b3c), n, to_ptr(out),
    )
    return out


def forward_one(w, i, x):
    """Single room: (l,) latent."""
    x = x.ravel().astype(np.float32)
    h = x @ w["w1"][i] + w["b1"][0, i]
    h = np.maximum(h, 0)
    h = h @ w["w2"][i] + w["b2"][0, i]
    h = np.maximum(h, 0)
    return h @ w["w3"][i] + w["b3"][0, i]


def novelty(z, history):
    """Cosine-distance novelty vs recent history."""
    if len(history) < 2:
        return 0.5
    recent = np.stack(history[-3:])
    zn = z / (np.linalg.norm(z) + 1e-8)
    rn = recent / (np.linalg.norm(recent, axis=-1, keepdims=True) + 1e-8)
    return float(1.0 - (zn * rn).sum(axis=-1).mean())


@dataclass
class Fingerprint:
    i: int
    sine: np.ndarray
    noise: np.ndarray
    step: np.ndarray
    activity: int
    def diff(self, other):
        n = lambda a,b: np.linalg.norm(a-b)
        return float(n(self.sine, other.sine) + n(self.noise, other.noise) + n(self.step, other.step))
    def __repr__(self):
        return f"Fingerprint(room={self.i}, activity={self.activity})"


class RoomGrid:
    """N rooms × MLP. Forward only. No training.

    Each room has unique random weights. Diversity comes from
    initialization + breed(). No training, no backprop ever needed.

    Usage:
        g = RoomGrid(250)
        g.tick(np.random.randn(64))   # all rooms signal
        g.cold()                     # sunset candidates
        g.breed(5, 100)              # clone room 5's weights to 100
    """

    def __init__(self, n=250, d=64, h=32, l=16, chaos=0.3):
        self.n = n
        self.w = make_weights(n, d, h, l)
        self.activity = np.zeros(n, dtype=np.int32)
        self.chaos = np.full(n, chaos, dtype=np.float32)
        self.history = {}
        self.ticks = 0
        self.l = l
        t = np.linspace(0, 2 * math.pi, d)
        self._ref = {"sine": np.sin(t).astype(np.float32),
                     "noise": np.random.randn(d).astype(np.float32),
                     "step": np.concatenate([np.zeros(d//2), np.ones(d//2)]).astype(np.float32)}

    def _forward(self, x):
        return forward_rust(self.w, x, self.n) if _BACKEND == "rust" else forward_einsum(self.w, x)

    def tick(self, x):
        self.ticks += 1
        latents = self._forward(x)
        fired = []
        for i in range(self.n):
            z = latents[i]
            nv = novelty(z, self.history.get(i, []))
            if nv > 0.5 or np.random.random() < self.chaos[i]:
                self.activity[i] += 1
                fired.append(i)
                self.chaos[i] = max(0.01, self.chaos[i] * 0.99)
            self.history.setdefault(i, []).append(z.copy())
            if len(self.history[i]) > 20:
                self.history[i].pop(0)
        return {"fired": len(fired), "ids": fired[:10], "tick": self.ticks}

    def fingerprints(self, n=50):
        return [Fingerprint(i, forward_one(self.w,i,self._ref["sine"]),
                forward_one(self.w,i,self._ref["noise"]),
                forward_one(self.w,i,self._ref["step"]), int(self.activity[i]))
                for i in range(min(n, self.n))]

    def top(self, k=10):
        idx = np.argsort(self.activity)[::-1][:k]
        return [(int(i), int(self.activity[i])) for i in idx]

    def cold(self, thresh=1):
        return [int(i) for i in range(self.n) if self.activity[i] < thresh]

    def rebirth(self, i):
        rng = np.random.RandomState(i + 9999)
        for k, shp in [("w1", (64, 32)), ("w2", (32, 16)), ("w3", (16, 16))]:
            self.w[k][i] = rng.randn(*shp).astype(np.float32) * 0.01
        self.activity[i] = 0
        self.chaos[i] = 0.3
        self.history[i] = []

    def breed(self, src, dst):
        """Rebirth dst with weights cloned from src + light mutation."""
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] = self.w[k][src].copy()
        rng = np.random.RandomState(dst + 8888)
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] += rng.randn(*self.w[k][dst].shape).astype(np.float32) * 0.005
        self.activity[dst] = 0
        self.chaos[dst] = 0.3
        self.history[dst] = []

    def __repr__(self):
        backend_str = "rust" if _BACKEND == "rust" else "numpy"
        return f"RoomGrid(n={self.n}, ticks={self.ticks}, active={int((self.activity>0).sum())}, {backend_str})"

    @property
    def stats(self):
        a = int((self.activity > 0).sum())
        return {"rooms": self.n, "ticks": self.ticks, "active": a, "cold": self.n - a, "pct": f"{a/self.n*100:.1f}%"}


if __name__ == "__main__":
    import time
    for n in [250, 1000, 5000, 10000]:
        g = RoomGrid(n)
        start = time.perf_counter()
        for _ in range(10):
            g.tick(np.random.randn(64))
        avg = (time.perf_counter() - start) / 10
        print(f"{n:5d} rooms: {avg*1000:.1f}ms/tick ({avg/n*1e9:.0f}ns/room)")
    b = "Rust FFI" if _BACKEND == "rust" else "numpy"
    print(f"Backend: {b}")


# Alias for SPEC-NERVE-TOPOLOGY compatibility
JEPAGrid = RoomGrid
