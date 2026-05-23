"""RoomGrid — Hardware-aware adaptive forward engine.

Auto-detects at import time and loads the fastest kernel:
  1. CUDA      — GPU kernel (requires nvcc + CUDA runtime)
  2. Rust Persistent — weights in Rust memory, zero-copy per tick
  3. Rust Oneshot    — legacy ctypes (overhead kills small arrays)
  4. numpy     — pure einsum fallback (always works)

Each room = 3.4K params. No training, no backprop."""

from __future__ import annotations
__all__ = ["RoomGrid", "JEPAGrid", "Fingerprint", "make_weights", "novelty", "batch_novelty"]

import math, threading, logging, sys
from collections import deque
from ctypes import CDLL, c_float, c_size_t, POINTER, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np

log = logging.getLogger(__name__)

# ── Hardware detection — gets to the metal ────────────────
_RUST_LIB = None
_CUDA_LIB = None
_BACKEND = "numpy"          # default fallback

# Thresholds for backend switching (rooms)
_RUST_ONESHOT_THRESHOLD = 500   # below: numpy wins (ctypes overhead)
_RUST_PERSIST_THRESHOLD = 50  # below: numpy wins (setup cost)
_CUDA_THRESHOLD = 1000       # above: CUDA dominates

# Try CUDA first (fastest)
try:
    import ctypes as _ctypes_cuda
    _CUDA_LIB = _ctypes_cuda.CDLL("libcudart.so")
    _BACKEND = "cuda"
except OSError:
    _CUDA_LIB = None

# Try Rust persistent FFI (fastest CPU path)
if _BACKEND == "numpy":
    try:
        _so = next(Path(__file__).parent.glob("target/release/libjepa_kernel.so"))
        _RUST_LIB = CDLL(str(_so))
        # New persistent API: weights stay in Rust
        _RUST_LIB.jepa_grid_create.argtypes = [
            c_size_t,
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
            POINTER(c_float), POINTER(c_float), POINTER(c_float),
        ]
        _RUST_LIB.jepa_grid_create.restype = c_void_p
        _RUST_LIB.jepa_grid_tick.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
        _RUST_LIB.jepa_grid_tick.restype = None
        _RUST_LIB.jepa_grid_tick_batch.argtypes = [
            c_void_p, POINTER(c_float), c_size_t, POINTER(c_float)
        ]
        _RUST_LIB.jepa_grid_tick_batch.restype = None
        _RUST_LIB.jepa_grid_destroy.argtypes = [c_void_p]
        _RUST_LIB.jepa_grid_destroy.restype = None
        # Legacy oneshot API
        _RUST_LIB.jepa_forward_batch.argtypes = [POINTER(c_float)]*7 + [c_size_t, POINTER(c_float)]
        _RUST_LIB.jepa_forward_batch.restype = None
        _BACKEND = "rust_persistent"
    except (StopIteration, OSError, AttributeError):
        _RUST_LIB = None

# If persistent API missing, try oneshot-only (FM's v1 .so has forward_batch only)
if _BACKEND == "numpy":
    try:
        _so = next(Path(__file__).parent.glob("target/release/libjepa_kernel.so"))
        _RUST_LIB = CDLL(str(_so))
        _RUST_LIB.jepa_forward_batch.argtypes = [POINTER(c_float)]*7 + [c_size_t, POINTER(c_float)]
        _RUST_LIB.jepa_forward_batch.restype = None
        _BACKEND = "rust_oneshot"
    except (StopIteration, OSError, AttributeError):
        _RUST_LIB = None


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
    """Numpy einsum fallback: (n, l) latents.

    PERFORMANCE: optimize=False skips expensive path optimization
    for small arrays where overhead exceeds computation time.
    """
    x = x.ravel().astype(np.float32)
    h = np.einsum("d,ndh->nh", x, w["w1"], optimize=False) + w["b1"][0]
    h = np.maximum(h, 0, out=h)
    h = np.einsum("nh,nhl->nl", h, w["w2"], optimize=False) + w["b2"][0]
    h = np.maximum(h, 0, out=h)
    return np.einsum("nl,nll->nl", h, w["w3"], optimize=False) + w["b3"][0]


def _to_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(POINTER(c_float))


def forward_rust_oneshot(w, x, n):
    """Legacy Rust FFI — 7× ascontiguousarray overhead per tick.
    Only used for medium arrays where numpy overhead is still high.
    """
    if _RUST_LIB is None:
        return forward_einsum(w, x)
    xc = np.ascontiguousarray(x.ravel().astype(np.float32))
    out = np.empty((n, 16), dtype=np.float32)
    _RUST_LIB.jepa_forward_batch(
        _to_ptr(xc),
        _to_ptr(np.ascontiguousarray(w["w1"].ravel())),
        _to_ptr(np.ascontiguousarray(w["w2"].ravel())),
        _to_ptr(np.ascontiguousarray(w["w3"].ravel())),
        _to_ptr(np.ascontiguousarray(w["b1"].ravel())),
        _to_ptr(np.ascontiguousarray(w["b2"].ravel())),
        _to_ptr(np.ascontiguousarray(w["b3"].ravel())),
        n, _to_ptr(out),
    )
    return out


class PersistentRustGrid:
    """Weights live in Rust memory. Python only sends signals.
    Eliminates 7× ascontiguousarray() overhead per tick.
    """
    __slots__ = ("n", "_handle", "_out")

    def __init__(self, n: int, weights: dict):
        self.n = n
        self._handle = None
        self._out = np.empty((n, 16), dtype=np.float32)

        w1 = np.ascontiguousarray(weights["w1"].ravel(), dtype=np.float32)
        w2 = np.ascontiguousarray(weights["w2"].ravel(), dtype=np.float32)
        w3 = np.ascontiguousarray(weights["w3"].ravel(), dtype=np.float32)
        b1 = np.ascontiguousarray(weights["b1"].ravel(), dtype=np.float32)
        b2 = np.ascontiguousarray(weights["b2"].ravel(), dtype=np.float32)
        b3 = np.ascontiguousarray(weights["b3"].ravel(), dtype=np.float32)

        self._handle = _RUST_LIB.jepa_grid_create(
            n, _to_ptr(w1), _to_ptr(w2), _to_ptr(w3),
            _to_ptr(b1), _to_ptr(b2), _to_ptr(b3),
        )
        if not self._handle:
            raise RuntimeError("jepa_grid_create failed")

    def tick(self, signal: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(signal.ravel()[:64], dtype=np.float32)
        _RUST_LIB.jepa_grid_tick(
            self._handle, _to_ptr(x), _to_ptr(self._out),
        )
        return self._out

    def tick_batch(self, signals: np.ndarray) -> np.ndarray:
        batch = signals.shape[0]
        sigs = np.ascontiguousarray(signals.reshape(batch, 64).astype(np.float32))
        out = np.empty((batch, self.n, 16), dtype=np.float32)
        _RUST_LIB.jepa_grid_tick_batch(
            self._handle, _to_ptr(sigs), batch, _to_ptr(out),
        )
        return out

    def __del__(self):
        if self._handle:
            _RUST_LIB.jepa_grid_destroy(self._handle)
            self._handle = None

    def __repr__(self):
        return f"PersistentRustGrid(n={self.n}, alive={self._handle is not None})"


# ── Auto-dispatch table ───────────────────────────────────
# Maps (backend, n_rooms) → forward function or grid instance
_dispatch = {}


def _select_backend(n: int, d: int = 64, h: int = 32, l: int = 16):
    """Return the fastest backend for this room count and dimensions.

    Rust/CUDA backends are compiled with fixed dimensions (64→32→16).
    If dimensions differ, fall back to numpy to avoid segfaults.
    """
    if d != 64 or h != 32 or l != 16:
        return "numpy"
    if _BACKEND == "cuda" and n >= _CUDA_THRESHOLD:
        return "cuda"
    if _BACKEND == "rust_persistent":
        if n >= _RUST_ONESHOT_THRESHOLD:
            return "rust_persistent"
        elif n >= _RUST_PERSIST_THRESHOLD:
            return "rust_oneshot"
    if _BACKEND == "rust_oneshot" and n >= _RUST_ONESHOT_THRESHOLD:
        return "rust_oneshot"
    return "numpy"


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


def batch_novelty(latents: np.ndarray, hist: np.ndarray, hist_count: np.ndarray,
                 hist_idx: int, hist_max: int) -> np.ndarray:
    """Vectorized novelty for all rooms — ring buffer edition.

    Auto-selects Numba JIT if available (~7× faster after warmup),
    falls back to numpy otherwise.
    """
    if _HAS_NUMBA:
        return _batch_novelty_numba(latents, hist, hist_count, hist_idx, hist_max)
    return _batch_novelty_numpy(latents, hist, hist_count, hist_idx, hist_max)


def _batch_novelty_numpy(latents, hist, hist_count, hist_idx, hist_max):
    """Pure numpy implementation (fallback)."""
    n = latents.shape[0]
    norms = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-8
    zn = latents / norms  # (n, 16)

    # Extract last 3 entries from ring buffer — vectorized
    offsets = [(hist_idx - 1) % hist_max,
               (hist_idx - 2) % hist_max,
               (hist_idx - 3) % hist_max]
    hist_tensor = hist[offsets].transpose(1, 0, 2)  # (n, 3, 16)

    hist_mask = np.zeros((n, 3), dtype=np.float32)
    for j in range(3):
        hist_mask[:, j] = (hist_count >= j + 1).astype(np.float32)

    h_norms = np.linalg.norm(hist_tensor, axis=-1, keepdims=True) + 1e-8
    hn = hist_tensor / h_norms
    sims = (zn[:, np.newaxis, :] * hn).sum(axis=-1)  # (n, 3)

    mask_sum = hist_mask.sum(axis=1, keepdims=True) + 1e-8
    mean_sim = (sims * hist_mask).sum(axis=1, keepdims=True) / mask_sum
    novelty = 1.0 - mean_sim.ravel()

    no_hist = hist_mask.sum(axis=1) < 2
    novelty[no_hist] = 0.5
    return novelty


# ── Numba JIT version (auto-compiled at import if numba available) ──
_HAS_NUMBA = False
_batch_novelty_numba = None

try:
    from numba import njit
    import numba

    @njit(cache=True, fastmath=True)
    def _batch_novelty_numba_inner(latents, h1, h2, h3, hist_count):
        """Numba-compiled novelty kernel.
        
        Pre-extracted history slices (h1,h2,h3) to avoid dynamic indexing
        inside Numba. Computes cosine similarity per room.
        """
        n = latents.shape[0]
        l = latents.shape[1]
        
        # Normalize latents: zn = latents / ||latents||
        zn = np.empty((n, l), dtype=np.float32)
        for i in range(n):
            norm_sq = 0.0
            for j in range(l):
                v = latents[i, j]
                norm_sq += v * v
            norm = np.sqrt(norm_sq) + 1e-8
            for j in range(l):
                zn[i, j] = latents[i, j] / norm
        
        # Compute similarities with each history slice
        sims = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            for k, h in enumerate((h1, h2, h3)):
                # Normalize history slice
                norm_sq = 0.0
                for j in range(l):
                    v = h[i, j]
                    norm_sq += v * v
                norm = np.sqrt(norm_sq) + 1e-8
                
                # Cosine similarity
                sim = 0.0
                for j in range(l):
                    sim += zn[i, j] * (h[i, j] / norm)
                sims[i, k] = sim
        
        # Mask by hist_count and compute mean
        novelty = np.empty(n, dtype=np.float32)
        for i in range(n):
            count = hist_count[i]
            if count < 2:
                novelty[i] = 0.5
            else:
                total = 0.0
                valid = 0
                for k in range(3):
                    if k < count:
                        total += sims[i, k]
                        valid += 1
                if valid > 0:
                    mean_sim = total / valid
                    novelty[i] = 1.0 - mean_sim
                else:
                    novelty[i] = 0.5
        
        return novelty

    def _batch_novelty_numba(latents, hist, hist_count, hist_idx, hist_max):
        """Python wrapper: extracts ring buffer slices, calls Numba kernel."""
        # Extract 3 history slices (cheap — no new allocation of full tensor)
        h1 = hist[(hist_idx - 1) % hist_max]
        h2 = hist[(hist_idx - 2) % hist_max]
        h3 = hist[(hist_idx - 3) % hist_max]
        return _batch_novelty_numba_inner(latents, h1, h2, h3, hist_count)

    _HAS_NUMBA = True

except ImportError:
    pass  # _HAS_NUMBA stays False, _batch_novelty_numba stays None


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

    def __init__(self, n=250, d=64, h=32, l=16, chaos=0.3, compiler=None,
                 agent_config=None):
        self.n = n
        self.w = make_weights(n, d, h, l)
        self.activity = np.zeros(n, dtype=np.int32)
        self.chaos = np.full(n, chaos, dtype=np.float32)
        self.ticks = 0
        self.l = l
        self._out = np.empty((n, 16), dtype=np.float32)  # pre-allocated output buffer
        self.latents = np.zeros((n, 16), dtype=np.float32)  # last tick outputs
        # Ring buffer history: (maxlen, n, l) — vectorized, no per-room deques
        self._hist_max = 20
        self._hist = np.zeros((self._hist_max, n, l), dtype=np.float32)
        self._hist_idx = 0  # write pointer
        self._hist_count = np.zeros(n, dtype=np.int32)  # how many entries per room
        t = np.linspace(0, 2 * math.pi, d)
        self._ref = {"sine": np.sin(t).astype(np.float32),
                     "noise": np.random.randn(d).astype(np.float32),
                     "step": np.concatenate([np.zeros(d//2), np.ones(d//2)]).astype(np.float32)}
        self._flux_checker = None  # Optional FluxConstraintChecker
        self._compiler = None     # Optional RoomGridCompiler
        self._cognition_loop = None  # Optional CognitionLoop
        self._plato_observer = None  # Optional RoomGridPlatoObserver
        self._last_fired_ids: list[int] = []  # stored for cognition observer
        # ── Cognition integration ────────────────────────────
        if agent_config is not None:
            from perception.cognition_loop import AgentConfig, CognitionLoop
            if isinstance(agent_config, AgentConfig):
                self._agent_config = agent_config
                if agent_config.enable_cognition:
                    self._cognition_loop = CognitionLoop(agent_config)
                    log.info("CognitionLoop attached (interval=%d)",
                             agent_config.cognition_interval)
            elif hasattr(agent_config, "enable_cognition"):
                # Duck-typed config
                self._agent_config = agent_config
                if agent_config.enable_cognition:
                    self._cognition_loop = CognitionLoop(agent_config)
            else:
                raise TypeError("agent_config must be an AgentConfig or duck-typed equivalent")
        else:
            self._agent_config = None
        # ── Auto-compile integration ─────────────────────────
        if compiler is not None:
            if compiler == "auto":
                try:
                    from sunset.compiler_integration import RoomGridCompiler
                    self._compiler = RoomGridCompiler(self)
                    self._compiler.auto_compile(ticks=100, ab_trials=50)
                except Exception as e:
                    log.warning("RoomGrid auto-compile failed: %s", e)
            elif hasattr(compiler, "auto_compile"):
                self._compiler = compiler
                compiler.grid = self
            else:
                raise TypeError("compiler must be 'auto' or a RoomGridCompiler instance")

    def attach_plato_observer(self, observer) -> None:
        """Attach a PLATO tile observer for tick/lifecycle events."""
        if hasattr(observer, "on_tick") and hasattr(observer, "on_agent_spawn"):
            self._plato_observer = observer
            log.info("PlatoObserver attached to RoomGrid(n=%d)", self.n)
        else:
            raise TypeError("Expected RoomGridPlatoObserver-like object")

    def attach_flux_checker(self, checker) -> None:
        """Attach a FLUX constraint checker for self-correcting behavior."""
        # Duck-type check — avoids import path issues in tests
        if hasattr(checker, "check_batch") and hasattr(checker, "get_violations"):
            self._flux_checker = checker
            log.info("FLUX constraint checker attached to RoomGrid(n=%d)", self.n)
        else:
            raise TypeError("Expected FluxConstraintChecker-like object (needs check_batch, get_violations)")

    def _forward(self, x):
        """Auto-dispatch to fastest backend for this room count."""
        # Infer dimensions from weights
        d = self.w["w1"].shape[1]  # (n, d, h)
        h = self.w["w1"].shape[2]
        l = self.w["w3"].shape[1]  # (n, l, l)
        backend = _select_backend(self.n, d, h, l)

        if backend == "cuda":
            if not hasattr(self, "_cuda_grid"):
                from nerve.cuda_bridge import PersistentCUDAGrid
                self._cuda_grid = PersistentCUDAGrid(self.n, self.w)
            return self._cuda_grid.tick(x)

        if backend == "rust_persistent":
            if not hasattr(self, "_rust_grid"):
                self._rust_grid = PersistentRustGrid(self.n, self.w)
                # Warm-up tick to amortize Rust/Numba JIT cold-start cost
                self._rust_grid.tick(np.zeros(d, dtype=np.float32))
            return self._rust_grid.tick(x)

        if backend == "rust_oneshot":
            return forward_rust_oneshot(self.w, x, self.n)

        return forward_einsum(self.w, x)

    def tick(self, x):
        self.ticks += 1
        latents = self._forward(x)
        self.latents = latents  # Store for constraint checking
        # Vectorized ring-buffer append — no Python loop, no per-room .copy()
        self._hist[self._hist_idx] = latents
        self._hist_idx = (self._hist_idx + 1) % self._hist_max
        self._hist_count = np.minimum(self._hist_count + 1, self._hist_max)
        # ── Compiled routing fast-path ───────────────────────
        _tick_routing_compiled = getattr(sys.modules.get("nerve.room_grid"), "_tick_routing_compiled", None)
        if _tick_routing_compiled is not None:
            nv = batch_novelty(latents, self._hist, self._hist_count, self._hist_idx, self._hist_max)
            fired_mask, new_chaos, fired_count = _tick_routing_compiled(
                latents, self.chaos, self.n,
                self._hist, self._hist_count, self._hist_idx, self._hist_max,
            )
            self.chaos = new_chaos
            fired = np.where(fired_mask)[0].tolist()[:10]
            self._last_fired_ids = fired
            self.activity[fired_mask] += 1
            # FLUX constraint feedback
            if self._flux_checker is not None:
                from sunset.flux_integration import apply_constraint_feedback
                apply_constraint_feedback(self, self._flux_checker)
            # ── Cognition loop (compiled path) ─────────────
            if self._cognition_loop is not None:
                self._cognition_loop.loop(self)
            # ── PLATO observer ─────────────────────────────────
            if self._plato_observer is not None:
                self._plato_observer.on_tick(self, self.ticks, 0.0)
            return {"fired": fired_count, "ids": fired, "tick": self.ticks}
        # ── Fallback vectorised novelty + chaos gating ───────
        nv = batch_novelty(latents, self._hist, self._hist_count, self._hist_idx, self._hist_max)
        chaos_fire = np.random.random(self.n) < self.chaos
        fired_mask = (nv > 0.5) | chaos_fire
        fired = np.where(fired_mask)[0].tolist()[:10]
        self._last_fired_ids = fired
        self.activity[fired_mask] += 1
        self.chaos = np.where(fired_mask, np.maximum(0.01, self.chaos * 0.99), self.chaos)
        # FLUX constraint feedback
        if self._flux_checker is not None:
            from sunset.flux_integration import apply_constraint_feedback
            apply_constraint_feedback(self, self._flux_checker)
        # ── Cognition loop ───────────────────────────────────
        if self._cognition_loop is not None:
            self._cognition_loop.loop(self)
        # ── PLATO observer ─────────────────────────────────
        if self._plato_observer is not None:
            self._plato_observer.on_tick(self, self.ticks, 0.0)
        return {"fired": int(fired_mask.sum()), "ids": fired, "tick": self.ticks}

    def tick_batch(self, signals):
        """Batch tick: signals (batch, 64) -> list of tick dicts.

        Amortizes kernel launch overhead across multiple ticks.
        Falls back to numpy einsum if no fast backend is available.
        """
        backend = _select_backend(self.n)
        if backend == "cuda" and hasattr(self, "_cuda_grid"):
            latents = self._cuda_grid.tick_batch(signals)
        elif backend == "rust_persistent" and hasattr(self, "_rust_grid"):
            latents = self._rust_grid.tick_batch(signals)
        else:
            # Fallback: numpy loop
            results = []
            for sig in signals:
                results.append(self.tick(sig))
            return results

        # Process batch results through novelty/chaos/feedback
        results = []
        for b in range(latents.shape[0]):
            self.ticks += 1
            latent = latents[b]
            self.latents = latent
            self._hist[self._hist_idx] = latent
            self._hist_idx = (self._hist_idx + 1) % self._hist_max
            self._hist_count = np.minimum(self._hist_count + 1, self._hist_max)
            nv = batch_novelty(latent, self._hist, self._hist_count, self._hist_idx, self._hist_max)
            chaos_fire = np.random.random(self.n) < self.chaos
            fired_mask = (nv > 0.5) | chaos_fire
            fired = np.where(fired_mask)[0].tolist()[:10]
            self._last_fired_ids = fired
            self.activity[fired_mask] += 1
            self.chaos = np.where(fired_mask, np.maximum(0.01, self.chaos * 0.99), self.chaos)
            if self._flux_checker is not None:
                from sunset.flux_integration import apply_constraint_feedback
                apply_constraint_feedback(self, self._flux_checker)
            # ── Cognition loop (batch) ─────────────────────
            if self._cognition_loop is not None:
                self._cognition_loop.loop(self)
            # ── PLATO observer (batch) ─────────────────────
            if self._plato_observer is not None:
                self._plato_observer.on_tick(self, self.ticks, 0.0)
            results.append({"fired": int(fired_mask.sum()), "ids": fired, "tick": self.ticks})
        return results

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
        """Reset room `i` to random weights. Invalidates Rust cache."""
        rng = np.random.RandomState(i + 9999)
        for k, shp in [("w1", (64, 32)), ("w2", (32, 16)), ("w3", (16, 16))]:
            self.w[k][i] = rng.randn(*shp).astype(np.float32) * 0.01
        self.activity[i] = 0
        self.chaos[i] = 0.3
        self._hist[:, i, :] = 0.0
        self._hist_count[i] = 0
        # Invalidate persistent grid — weights changed
        if hasattr(self, "_rust_grid"):
            del self._rust_grid
        # Invalidate CUDA grid — weights changed
        if hasattr(self, "_cuda_grid"):
            del self._cuda_grid

    def breed(self, src, dst):
        """Clone src weights to dst + mutation. Invalidates Rust cache."""
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] = self.w[k][src].copy()
        rng = np.random.RandomState(dst + 8888)
        for k in ("w1", "w2", "w3"):
            self.w[k][dst] += rng.randn(*self.w[k][dst].shape).astype(np.float32) * 0.005
        self.activity[dst] = 0
        self.chaos[dst] = 0.3
        self._hist[:, dst, :] = 0.0
        self._hist_count[dst] = 0
        # Invalidate persistent grid — weights changed
        if hasattr(self, "_rust_grid"):
            del self._rust_grid
        # Invalidate CUDA grid — weights changed
        if hasattr(self, "_cuda_grid"):
            del self._cuda_grid

    def __repr__(self):
        backend = _select_backend(self.n)
        return f"RoomGrid(n={self.n}, ticks={self.ticks}, active={int((self.activity>0).sum())}, backend={backend})"

    def agent_count(self) -> int:
        """Return the number of rooms that have fired at least once."""
        return int((self.activity > 0).sum())

    @property
    def stats(self):
        a = int((self.activity > 0).sum())
        return {"rooms": self.n, "ticks": self.ticks, "active": a, "cold": self.n - a, "pct": f"{a/self.n*100:.1f}%", "diversity": self.diversity()}

    def diversity(self, use_hdc: bool = True) -> float:
        """Population diversity: mean pairwise distance between active rooms.

        Computes the average Hamming (HDC) or cosine distance between all
        pairs of rooms that have fired at least once.  Returns 0.0 when
        fewer than 2 rooms are active.

        Args:
            use_hdc: When True, use HDC (XOR+POPCNT) Hamming distance.
                When False, use cosine distance.  HDC is ~100-1000x
                faster on AVX-512 hardware.

        Returns:
            Mean pairwise distance in [0, 1].  Higher = more diverse.
        """
        active = [i for i in range(self.n) if self.activity[i] > 0]
        m = len(active)
        if m < 2:
            return 0.0

        # Flatten each room's weights into a single vector
        vectors: list[np.ndarray] = []
        for i in active:
            vec = np.concatenate([
                self.w["w1"][i].ravel(),
                self.w["w2"][i].ravel(),
                self.w["w3"][i].ravel(),
            ]).astype(np.float32)
            vectors.append(vec)

        if use_hdc:
            try:
                from swarm.hdc_novelty import hdc_novelty_score
                total = 0.0
                count = 0
                for i in range(m):
                    for j in range(i + 1, m):
                        total += hdc_novelty_score(vectors[i], vectors[j])
                        count += 1
                return total / count if count else 0.0
            except ImportError:
                pass  # fall through to cosine

        # Cosine fallback
        vecs = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vecs / norms
        sims = normalized @ normalized.T
        sims = np.clip(sims, -1.0, 1.0)
        dists = 1.0 - sims
        # Upper triangle mean (excluding diagonal)
        triu = np.triu_indices(m, k=1)
        return float(dists[triu].mean())


if __name__ == "__main__":
    import time
    print("=== Backend Detection ===")
    print(f"  CUDA:     {'✅' if _CUDA_LIB else '❌'} (libcudart.so)")
    print(f"  Rust:     {'✅' if _RUST_LIB else '❌'} (libjepa_kernel.so)")
    print(f"  Selected: {_BACKEND}")
    print()
    print("=== Adaptive Benchmark (auto-selects backend per size) ===")
    for n in [10, 100, 500, 1000, 5000, 10000]:
        g = RoomGrid(n)
        # Warmup + trigger backend init
        for _ in range(5):
            g.tick(np.random.randn(64))
        start = time.perf_counter()
        for _ in range(20):
            g.tick(np.random.randn(64))
        avg = (time.perf_counter() - start) / 20
        backend = _select_backend(n)
        print(f"{n:5d} rooms: {avg*1000:7.2f}ms/tick  backend={backend}  {g}")
    print()
    print("=== Correctness Check (persistent vs numpy) ===")
    g1 = RoomGrid(1000)
    g2 = RoomGrid(1000)
    x = np.random.randn(64)
    for _ in range(5):
        g1.tick(x)
        g2.tick(x)
    # Force numpy for g2 by temporarily clearing rust
    _orig_backend = _BACKEND
    # numpy path
    out_np = forward_einsum(g1.w, x)
    # persistent rust path (if available)
    if _RUST_LIB:
        grid = PersistentRustGrid(1000, g1.w)
        out_rust = grid.tick(x)
        max_diff = np.max(np.abs(out_np - out_rust))
        print(f"numpy vs rust persistent: max_diff={max_diff:.2e}  {'✅' if max_diff < 1e-3 else '❌'}")
    else:
        print("Rust not available — skipping correctness check")


# Alias for SPEC-NERVE-TOPOLOGY compatibility
JEPAGrid = RoomGrid
