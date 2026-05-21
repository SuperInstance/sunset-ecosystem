"""JEPAGrid — Rust subprocess wrapper with API parity to room_grid.py.

Spawns the Rust `jepa-cli` binary as a subprocess for the forward pass.
Falls back to pure numpy (room_grid.forward_einsum) if the binary is unavailable.

All other methods (tick, fingerprints, rebirth, breed, etc.) use the same
implementation as room_grid.py; only the heavy _forward() path is swapped.
"""

from __future__ import annotations
__all__ = ["JEPAGrid", "Fingerprint", "make_weights", "novelty"]

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from room_grid import (
    make_weights,
    forward_einsum,
    forward_one,
    novelty,
    Fingerprint,
)

# ── Discover Rust binary ────────────────────────────────────
_RUST_EXE = None
for candidate in [
    Path(__file__).parent / "target" / "release" / "jepa-cli",
    Path(__file__).parent / "target" / "debug" / "jepa-cli",
]:
    if candidate.exists():
        _RUST_EXE = str(candidate)
        break


def forward_rust_subprocess(w, x, n):
    """Call the Rust CLI via subprocess; return (n, 16) latents."""
    if _RUST_EXE is None:
        raise RuntimeError("Rust binary jepa-cli not found")

    payload = {
        "n": int(n),
        "x": x.ravel().astype(np.float32).tolist(),
        "w1": np.ascontiguousarray(w["w1"].ravel()).tolist(),
        "w2": np.ascontiguousarray(w["w2"].ravel()).tolist(),
        "w3": np.ascontiguousarray(w["w3"].ravel()).tolist(),
        "b1": np.ascontiguousarray(w["b1"].ravel()).tolist(),
        "b2": np.ascontiguousarray(w["b2"].ravel()).tolist(),
        "b3": np.ascontiguousarray(w["b3"].ravel()).tolist(),
    }

    # Pass via stdin to avoid temp-file races
    proc = subprocess.run(
        [_RUST_EXE],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"jepa-cli failed: {proc.stderr}")

    result = json.loads(proc.stdout)
    latents = np.array(result["latents"], dtype=np.float32).reshape(n, 16)
    return latents


class JEPAGrid:
    """N rooms × JEPA. Subprocess Rust or numpy einsum fallback."""

    def __init__(self, n=250, d=64, h=32, l=16, chaos=0.3):
        self.n = n
        self.w = make_weights(n, d, h, l)
        self.activity = np.zeros(n, dtype=np.int32)
        self.chaos = np.full(n, chaos, dtype=np.float32)
        self.history = {}
        self.ticks = 0
        self.l = l
        t = np.linspace(0, 2 * math.pi, d)
        self._ref = {
            "sine": np.sin(t).astype(np.float32),
            "noise": np.random.randn(d).astype(np.float32),
            "step": np.concatenate([np.zeros(d // 2), np.ones(d // 2)]).astype(np.float32),
        }

    def _forward(self, x):
        try:
            return forward_rust_subprocess(self.w, x, self.n)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            return forward_einsum(self.w, x)

    def tick(self, x):
        self.ticks += 1
        latents = self._forward(x)
        fired = []
        for i in range(self.n):
            z = latents[i]
            nv = novelty(z, self.history.get(i, []))
            if nv > 0.3 or np.random.random() < self.chaos[i]:
                self.activity[i] += 1
                fired.append(i)
                self.chaos[i] = max(0.01, self.chaos[i] * 0.99)
            self.history.setdefault(i, []).append(z.copy())
            if len(self.history[i]) > 20:
                self.history[i].pop(0)
        return {"fired": len(fired), "ids": fired[:10], "tick": self.ticks}

    def fingerprints(self, n=50):
        return [
            Fingerprint(
                i,
                forward_one(self.w, i, self._ref["sine"]),
                forward_one(self.w, i, self._ref["noise"]),
                forward_one(self.w, i, self._ref["step"]),
                int(self.activity[i]),
            )
            for i in range(min(n, self.n))
        ]

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
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] = self.w[k][src].copy()
        rng = np.random.RandomState(dst + 8888)
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] += rng.randn(*self.w[k][dst].shape).astype(np.float32) * 0.005
        self.activity[dst] = 0
        self.chaos[dst] = 0.3
        self.history[dst] = []

    def __repr__(self):
        backend = "rust-subprocess" if _RUST_EXE else "numpy"
        return f"JEPAGrid(n={self.n}, ticks={self.ticks}, active={int((self.activity > 0).sum())}, {backend})"

    @property
    def stats(self):
        a = int((self.activity > 0).sum())
        return {"rooms": self.n, "ticks": self.ticks, "active": a, "cold": self.n - a, "pct": f"{a / self.n * 100:.1f}%"}


if __name__ == "__main__":
    import time
    for n in [250, 1000, 5000, 10000]:
        g = JEPAGrid(n)
        start = time.perf_counter()
        for _ in range(10):
            g.tick(np.random.randn(64))
        avg = (time.perf_counter() - start) / 10
        print(f"{n:5d} rooms: {avg * 1000:.1f}ms/tick ({avg / n * 1e9:.0f}ns/room)")
    print(f"Backend: {'Rust subprocess' if _RUST_EXE else 'numpy fallback'}")
