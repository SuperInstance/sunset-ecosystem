"""sunset/compiler_integration.py — Auto-compile RoomGrid hot paths via agentic-compiler.

Bridges the agentic-compiler standalone package with sunset-ecosystem's
RoomGrid, automatically optimizing diversity() and tick() internals.

Usage:
    from sunset.compiler_integration import RoomGridCompiler
    from nerve.room_grid import RoomGrid

    grid = RoomGrid(100, 512)
    compiler = RoomGridCompiler(grid)
    compiler.install()  # registers hot paths with agentic-compiler

    # Run your grid — the compiler profiles, compiles, and hot-swaps
    # automatically.  If compilation fails, Python fallback is preserved.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Optional agentic-compiler integration ────────────────────────
try:
    from agentic_compiler import Compiler, JitBackend, CompilationResult

    _HAS_COMPILER = True
except Exception:
    Compiler = None  # type: ignore[misc,assignment]
    JitBackend = None  # type: ignore[misc,assignment]
    CompilationResult = None  # type: ignore[misc,assignment]
    _HAS_COMPILER = False

# ── Optional nexus event bus ───────────────────────────────────
try:
    from nexus.fleet_event_bus import FleetEventBus

    _HAS_BUS = True
except Exception:
    FleetEventBus = None  # type: ignore[misc,assignment]
    _HAS_BUS = False

logger = logging.getLogger(__name__)


class RoomGridCompiler:
    """Auto-compiles RoomGrid hot paths using agentic-compiler.

    Registers the following functions as compilation targets:
      • diversity()  — numpy-heavy pairwise distance loop
      • tick()       — vectorized agent update loop

    The compiler profiles call frequency, generates a Numba kernel,
    A/B tests against the Python path, and hot-swaps if the compiled
    version is faster.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        grid: Any,
        compiler: Any | None = None,
        bus: Any | None = None,
        backend_preference: List[str] | None = None,
        ab_test_ticks: int = 50,
        min_speedup: float = 1.2,
    ) -> None:
        self.grid = grid
        self._compiler = compiler
        self._bus = bus
        self._ab_test_ticks = ab_test_ticks
        self._min_speedup = min_speedup
        self._backend_pref = backend_preference or ["numba", "rust", "cuda"]
        self._compiled: Dict[str, bool] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._original_methods: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------
    def install(self) -> bool:
        """Register hot paths with the compiler.  Return True if installed."""
        if not _HAS_COMPILER:
            logger.warning("agentic-compiler not installed — skipping RoomGridCompiler")
            return False

        if self._compiler is None:
            self._compiler = Compiler()
            self._compiler.install()

        # Snapshot original methods so we can restore them
        self._original_methods["diversity"] = self.grid.diversity
        self._original_methods["tick"] = self.grid.tick

        # Register with compiler profiler
        self._compiler.register(
            fn=self.grid.diversity,
            name="roomgrid.diversity",
            arg_shapes=[(self.grid.n, self.grid.d)],
        )
        self._compiler.register(
            fn=self.grid.tick,
            name="roomgrid.tick",
            arg_shapes=[],
        )

        logger.info("RoomGridCompiler installed — diversity() and tick() registered")
        self._maybe_emit(
            "compiler_installed",
            {
                "methods": ["diversity", "tick"],
                "backends": self._backend_pref,
            },
        )
        return True

    def uninstall(self) -> None:
        """Restore original Python methods."""
        for name, fn in self._original_methods.items():
            setattr(self.grid, name, fn)
        self._compiled.clear()
        self._maybe_emit(
            "compiler_uninstalled", {"methods": list(self._original_methods)}
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "installed": bool(self._compiler),
            "compiled_methods": dict(self._compiled),
            "results": dict(self._results),
            "backend_pref": self._backend_pref,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _maybe_emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._bus and _HAS_BUS:
            self._bus.emit({"type": event_type, **payload})

    def _ab_test(
        self, name: str, compiled_fn: Callable, python_fn: Callable, ticks: int
    ) -> bool:
        """Run A/B test: compiled vs Python.  Return True if compiled wins."""
        compiled_us = []
        python_us = []

        for _ in range(ticks):
            # Compiled
            t0 = time.perf_counter_ns()
            compiled_fn()
            t1 = time.perf_counter_ns()
            compiled_us.append((t1 - t0) / 1000)

            # Python
            t0 = time.perf_counter_ns()
            python_fn()
            t1 = time.perf_counter_ns()
            python_us.append((t1 - t0) / 1000)

        median_compiled = float(np.median(compiled_us))
        median_python = float(np.median(python_us))
        speedup = median_python / median_compiled if median_compiled > 0 else 0.0

        self._results[name] = {
            "compiled_median_us": median_compiled,
            "python_median_us": median_python,
            "speedup": speedup,
            "ticks": ticks,
        }

        logger.info(
            "A/B %s: compiled=%.1fµs python=%.1fµs speedup=%.2fx",
            name,
            median_compiled,
            median_python,
            speedup,
        )
        return speedup >= self._min_speedup


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RoomGridCompiler loaded.  Run pytest for tests.")
