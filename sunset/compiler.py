"""Agentic Compiler — Runtime-adaptive compilation for the Sunset Ecosystem.

Watches function calls, identifies hot paths, and generates optimized
implementations using Numba, Rust, or CUDA. The system learns which
paths matter most and accelerates them without human intervention.

Architecture:
    Profiler  →  Analyzer  →  CodeGenerator  →  Validator  →  Deployer
    (watch)      (rank)       (compile)        (A/B test)    (hot-swap)

Usage:
    from sunset.compiler import Compiler
    compiler = Compiler()
    compiler.install()  # monkey-patches hot paths
    
    # Run your code...
    # The compiler profiles and recompiles automatically.

See docs/AGENTIC-COMPILER-RESEARCH.md for full research.
"""

from __future__ import annotations

__all__ = ["Compiler", "Profiler", "JitBackend", "CompilationResult", "GridBackendSelector"]

import functools
import inspect
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .codegen import CodeGenerator, GeneratedKernel, PythonAnalyzer


# ── Configuration ───────────────────────────────────────────

SAMPLE_RATE = 0.05          # Profile 5% of calls (reduce overhead)
COMPILE_THRESHOLD = 100     # Min calls before considering compilation
SPEEDUP_THRESHOLD = 2.0     # Min expected speedup to bother compiling
WARMUP_CALLS = 10           # Calls to ignore during profiling (JIT warmup)


# ── Data structures ─────────────────────────────────────────

@dataclass
class FunctionStats:
    """Runtime statistics for a single function."""
    name: str
    module: str
    calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    input_shapes: List[Tuple] = field(default_factory=list)
    last_compiled: float = 0.0
    compiled_version: Optional[Any] = None
    speedup: float = 1.0

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / max(self.calls, 1)

    @property
    def optimization_potential(self) -> float:
        """Higher = more worth optimizing."""
        return self.calls * self.avg_time_ms * self.avg_time_ms


@dataclass
class CompilationResult:
    """Result of compiling a function to a backend."""
    original: Callable
    compiled: Callable
    backend: str
    compile_time_ms: float
    speedup: float
    validated: bool
    error: Optional[str] = None


class JitBackend:
    """Base class for compilation backends."""
    name: str = "base"

    def can_compile(self, func: Callable, stats: FunctionStats) -> bool:
        raise NotImplementedError

    def compile(self, func: Callable, stats: FunctionStats) -> CompilationResult:
        raise NotImplementedError


# ── Hardware Detection (shared with nerve/room_grid.py) ─────

def detect_hardware():
    """Detect available compute backends."""
    hw = {"numpy": True, "numba": False, "rust_persistent": False,
          "rust_oneshot": False, "cuda": False}
    
    # Numba
    try:
        import numba
        hw["numba"] = True
    except ImportError:
        pass
    
    # Rust persistent (libjepa_kernel.so)
    try:
        from pathlib import Path
        so = next(Path(__file__).parent.parent.glob("nerve/target/release/libjepa_kernel.so"))
        from ctypes import CDLL
        CDLL(str(so))
        hw["rust_persistent"] = True
        hw["rust_oneshot"] = True
    except (StopIteration, OSError):
        pass
    
    # CUDA
    try:
        from ctypes import CDLL
        CDLL("libcudart.so")
        hw["cuda"] = True
    except OSError:
        pass
    
    return hw


HARDWARE = detect_hardware()


# ── Grid-Aware Backend Selector ───────────────────────────

class GridBackendSelector:
    """Selects the optimal backend for NerveTopology grid forward passes.
    
    Matches room_grid.py logic:
      - n < 50:    numpy (ctypes overhead dominates)
      - 50-500:    rust_oneshot (medium arrays)
      - 500+:      rust_persistent (zero-copy, weights in Rust)
      - 1000+ + GPU: cuda (if available)
    """
    
    THRESHOLDS = {
        "numpy": 0,
        "rust_oneshot": 50,
        "rust_persistent": 500,
        "cuda": 1000,
    }
    
    @classmethod
    def select(cls, n_rooms: int) -> str:
        """Return best backend name for `n_rooms`."""
        candidates = []
        if HARDWARE["cuda"] and n_rooms >= cls.THRESHOLDS["cuda"]:
            candidates.append("cuda")
        if HARDWARE["rust_persistent"] and n_rooms >= cls.THRESHOLDS["rust_persistent"]:
            candidates.append("rust_persistent")
        elif HARDWARE["rust_oneshot"] and n_rooms >= cls.THRESHOLDS["rust_oneshot"]:
            candidates.append("rust_oneshot")
        if not candidates:
            candidates.append("numpy")
        return candidates[0]  # first = highest priority
    
    @classmethod
    def report(cls) -> str:
        lines = ["=== Hardware Detection ==="]
        for name, available in HARDWARE.items():
            lines.append(f"  {name:<18} {'✅' if available else '❌'}")
        lines.append("")
        lines.append("=== Backend Thresholds ===")
        for backend, thresh in cls.THRESHOLDS.items():
            lines.append(f"  {backend:<18} n >= {thresh}")
        return "\n".join(lines)

class NumbaBackend(JitBackend):
    """Numba LLVM JIT backend — delegates to CodeGenerator."""
    name = "numba"

    def __init__(self) -> None:
        self._generator = CodeGenerator().numba
        self._available = self._generator._available

    def can_compile(self, func: Callable, stats: FunctionStats) -> bool:
        if not self._available:
            return False
        analyzer = PythonAnalyzer().analyze(func)
        return (
            stats.calls > COMPILE_THRESHOLD
            and stats.avg_time_ms > 0.1
            and self._generator.can_generate(func, analyzer)
        )

    def compile(self, func: Callable, stats: FunctionStats) -> CompilationResult:
        analyzer = PythonAnalyzer().analyze(func)
        test_args = self._generate_test_args(func)
        kernel = self._generator.generate(func, analyzer, test_args)
        return CompilationResult(
            original=func,
            compiled=kernel.compiled,
            backend="numba",
            compile_time_ms=kernel.compile_time_ms,
            speedup=1.0,
            validated=False,
            error=kernel.error,
        )

    def _generate_test_args(self, func: Callable) -> Tuple:
        """Generate synthetic arguments for warmup/validation."""
        sig = inspect.signature(func)
        args = []
        for param in sig.parameters.values():
            if param.default is not inspect.Parameter.empty:
                args.append(param.default)
            else:
                name = param.name.lower()
                if any(x in name for x in ("array", "x", "signal", "latent")):
                    args.append(np.random.randn(64).astype(np.float32))
                elif "n" in name or "count" in name:
                    args.append(100)
                elif "w" in name and "weight" in name:
                    args.append({"w1": np.random.randn(100, 64, 32).astype(np.float32)})
                else:
                    args.append(0.5)
        return tuple(args)


# ── Rust FFI Backend ────────────────────────────────────────

class RustBackend(JitBackend):
    """Rust FFI backend — delegates to CodeGenerator."""
    name = "rust"

    def __init__(self) -> None:
        self._generator = CodeGenerator().rust
        self._available = self._generator._available

    def can_compile(self, func: Callable, stats: FunctionStats) -> bool:
        if not self._available:
            return False
        analyzer = PythonAnalyzer().analyze(func)
        return (
            stats.calls > COMPILE_THRESHOLD * 2
            and stats.avg_time_ms > 1.0
            and self._generator.can_generate(func, analyzer)
        )

    def compile(self, func: Callable, stats: FunctionStats) -> CompilationResult:
        analyzer = PythonAnalyzer().analyze(func)
        test_args = self._generate_test_args(func)
        kernel = self._generator.generate(func, analyzer, test_args)
        return CompilationResult(
            original=func,
            compiled=kernel.compiled,
            backend="rust",
            compile_time_ms=kernel.compile_time_ms,
            speedup=1.0,
            validated=False,
            error=kernel.error,
        )

    def _generate_test_args(self, func: Callable) -> Tuple:
        sig = inspect.signature(func)
        args = []
        for param in sig.parameters.values():
            if param.default is not inspect.Parameter.empty:
                args.append(param.default)
            else:
                name = param.name.lower()
                if "dict" in name or "map" in name:
                    args.append({"key": "value"})
                elif "str" in name or "text" in name:
                    args.append("test string")
                else:
                    args.append(0.5)
        return tuple(args)


# ── Profiler ────────────────────────────────────────────────

class Profiler:
    """Watches function calls and builds a heat map."""

    def __init__(self, sample_rate: float = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.stats: Dict[str, FunctionStats] = {}
        self._enabled = True
        self._call_count = 0

    def watch(self, func: Callable) -> Callable:
        """Decorator: profile calls to this function."""
        name = f"{func.__module__}.{func.__qualname__}"
        if name not in self.stats:
            self.stats[name] = FunctionStats(name=name, module=func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._call_count += 1
            # Sampling: only profile some calls to reduce overhead
            should_profile = (
                self._enabled
                and self._call_count < 1000  # heavy sampling early
                or np.random.random() < self.sample_rate
            )
            if should_profile:
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                self._record(name, elapsed, args)
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper

    def _record(self, name: str, elapsed_ms: float, args: Tuple) -> None:
        stat = self.stats[name]
        stat.calls += 1
        stat.total_time_ms += elapsed_ms
        stat.min_time_ms = min(stat.min_time_ms, elapsed_ms)
        stat.max_time_ms = max(stat.max_time_ms, elapsed_ms)
        # Record input shapes for later compilation
        shapes = []
        for arg in args[:3]:  # first 3 args only
            if hasattr(arg, 'shape'):
                shapes.append(arg.shape)
            elif hasattr(arg, '__len__') and not isinstance(arg, str):
                try:
                    shapes.append((len(arg),))
                except TypeError:
                    pass
        if shapes:
            stat.input_shapes.append(tuple(shapes))
            # Keep last 20 shapes
            stat.input_shapes = stat.input_shapes[-20:]

    def get_hotspots(self, top_n: int = 10) -> List[FunctionStats]:
        """Return the top-N functions by optimization potential."""
        ranked = sorted(
            self.stats.values(),
            key=lambda s: s.optimization_potential,
            reverse=True,
        )
        return ranked[:top_n]

    def report(self) -> str:
        """Human-readable profiling report."""
        lines = ["=== Agentic Compiler — Profiling Report ===", ""]
        lines.append(f"{'Function':<40} {'Calls':>8} {'Avg(ms)':>10} {'Total(ms)':>12} {'Potential':>12}")
        lines.append("-" * 90)
        for stat in self.get_hotspots(15):
            lines.append(
                f"{stat.name:<40} {stat.calls:>8} "
                f"{stat.avg_time_ms:>10.3f} {stat.total_time_ms:>12.1f} "
                f"{stat.optimization_potential:>12.0f}"
            )
        return "\n".join(lines)


# ── Compiler ────────────────────────────────────────────────

class Compiler:
    """The agentic compiler daemon.

    Watches the ecosystem, identifies hot paths, and recompiles them.
    """

    def __init__(self) -> None:
        self.profiler = Profiler()
        self.backends: List[JitBackend] = [
            NumbaBackend(),
            RustBackend(),
        ]
        self.compiled: Dict[str, CompilationResult] = {}
        self._installed = False
        self._originals: Dict[str, Callable] = {}  # for rollback

    def install(self, module_name: Optional[str] = None) -> None:
        """Monkey-patch all functions in a module for profiling.

        Args:
            module_name: If provided, only profile functions in this module.
                        If None, profile all callable attributes in sys.modules.
        """
        if self._installed:
            return
        self._installed = True

        targets = []
        if module_name:
            mod = sys.modules.get(module_name)
            if mod:
                targets.append((module_name, mod))
        else:
            for name, mod in list(sys.modules.items()):
                if name.startswith("sunset") or name.startswith("nerve"):
                    targets.append((name, mod))

        for mod_name, mod in targets:
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(mod, attr_name)
                if callable(obj) and hasattr(obj, "__module__"):
                    if obj.__module__ == mod_name:
                        wrapped = self.profiler.watch(obj)
                        setattr(mod, attr_name, wrapped)
                        key = f"{mod_name}.{attr_name}"
                        self._originals[key] = obj

    def uninstall(self) -> None:
        """Restore original functions."""
        for key, original in self._originals.items():
            mod_name, attr_name = key.rsplit(".", 1)
            mod = sys.modules.get(mod_name)
            if mod:
                setattr(mod, attr_name, original)
        self._installed = False

    def compile_hotspots(self, top_n: int = 5) -> List[CompilationResult]:
        """Compile the top-N hot functions using CodeGenerator.

        Pipeline:
          1. Find best backend via CodeGenerator.analyze()
          2. Compile with warmup + validation
          3. A/B test for correctness
          4. Measure actual speedup
          5. Hot-swap if validated + speedup > threshold
        """
        results = []
        hotspots = self.profiler.get_hotspots(top_n)
        generator = CodeGenerator()

        for stat in hotspots:
            key = stat.name
            if key in self.compiled:
                continue

            mod_name, func_name = key.rsplit(".", 1)
            mod = sys.modules.get(mod_name)
            if not mod:
                continue
            func = getattr(mod, func_name, None)
            if not func:
                continue

            # Analyze + compile
            analyzer = generator.analyze(func)
            test_args = self._generate_test_args(func)
            kernel = generator.compile(func, test_args)

            result = CompilationResult(
                original=func,
                compiled=kernel.compiled,
                backend=kernel.backend,
                compile_time_ms=kernel.compile_time_ms,
                speedup=1.0,
                validated=False,
                error=kernel.error,
            )
            self.compiled[key] = result
            results.append(result)

            if not kernel.ready:
                continue

            # Validate
            validated = generator.validate(kernel, func, test_args)
            result.validated = validated

            if validated and kernel.source_language != "python":
                # Measure speedup
                speedup = generator.measure_speedup(kernel, func, test_args)
                result.speedup = speedup
                stat.speedup = speedup

                if speedup >= SPEEDUP_THRESHOLD:
                    # Hot-swap!
                    deployed = generator.deploy(kernel, mod, func_name)
                    if deployed:
                        stat.compiled_version = kernel.compiled
                        result.compiled = kernel.compiled
                        print(f"[Compiler] 🔥 Hot-swapped {key} — {speedup:.1f}× speedup ({kernel.backend})")
                else:
                    print(f"[Compiler] ⚠️  {key} compiled but speedup {speedup:.1f}× < {SPEEDUP_THRESHOLD}× threshold")

        return results

    def compile_function(
        self,
        func: Callable,
        module: Optional[types.ModuleType] = None,
        attr_name: Optional[str] = None,
    ) -> CompilationResult:
        """Manually compile a single function (for testing/development).

        Args:
            func: Function to compile
            module: Module to hot-swap into (optional)
            attr_name: Attribute name in module (optional)

        Returns:
            CompilationResult with status
        """
        generator = CodeGenerator()
        test_args = self._generate_test_args(func)
        kernel = generator.compile(func, test_args)

        result = CompilationResult(
            original=func,
            compiled=kernel.compiled,
            backend=kernel.backend,
            compile_time_ms=kernel.compile_time_ms,
            speedup=1.0,
            validated=False,
            error=kernel.error,
        )

        if not kernel.ready:
            return result

        validated = generator.validate(kernel, func, test_args)
        result.validated = validated

        if validated and kernel.source_language != "python":
            speedup = generator.measure_speedup(kernel, func, test_args)
            result.speedup = speedup

            if module and attr_name and speedup >= SPEEDUP_THRESHOLD:
                generator.deploy(kernel, module, attr_name)
                result.compiled = kernel.compiled
                print(f"[Compiler] 🔥 Hot-swapped {attr_name} — {speedup:.1f}× speedup")

        return result

    def _validate(self, original: Callable, compiled: Callable, trials: int = 5) -> bool:
        """A/B test: verify compiled function produces same output."""
        try:
            for _ in range(trials):
                args = self._generate_test_args(original)
                expected = original(*args)
                actual = compiled(*args)
                if not self._outputs_equal(expected, actual):
                    return False
            return True
        except Exception:
            return False

    def _generate_test_args(self, func: Callable) -> Tuple:
        """Generate synthetic arguments matching function signature."""
        sig = inspect.signature(func)
        args = []
        for param in sig.parameters.values():
            if param.default is not inspect.Parameter.empty:
                args.append(param.default)
            else:
                # Guess based on name
                if "array" in param.name or "x" in param.name:
                    args.append(np.random.randn(64).astype(np.float32))
                elif "n" in param.name or "count" in param.name:
                    args.append(100)
                else:
                    args.append(0.5)
        return tuple(args)

    def _outputs_equal(self, a: Any, b: Any) -> bool:
        """Check if two outputs are approximately equal."""
        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.allclose(a, b, rtol=1e-3, atol=1e-5)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(
                self._outputs_equal(x, y) for x, y in zip(a, b)
            )
        return a == b

    def _measure_speedup(self, original: Callable, compiled: Callable) -> float:
        """Measure actual speedup by timing both."""
        args = self._generate_test_args(original)
        # Time original
        t0 = time.perf_counter()
        for _ in range(10):
            original(*args)
        t_orig = (time.perf_counter() - t0) * 1000
        # Time compiled
        t0 = time.perf_counter()
        for _ in range(10):
            compiled(*args)
        t_comp = (time.perf_counter() - t0) * 1000
        return t_orig / max(t_comp, 0.001)

    def report(self) -> str:
        """Full compiler status report including hardware detection."""
        lines = [self.profiler.report(), ""]
        lines.append(GridBackendSelector.report())
        lines.append("")
        lines.append("=== Compiled Functions ===")
        if not self.compiled:
            lines.append("None yet.")
        else:
            lines.append(f"{'Function':<40} {'Backend':<10} {'Speedup':>8} {'Validated':>10}")
            for key, result in self.compiled.items():
                v = "✅" if result.validated else "❌"
                lines.append(
                    f"{key:<40} {result.backend:<10} "
                    f"{result.speedup:>7.1f}× {v:>10}"
                )
        return "\n".join(lines)
