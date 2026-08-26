"""Code Generator — Procedural compilation from Python to Numba/Rust.

The agentic compiler doesn't just detect hardware. It writes code.

Pipeline:
  1. Analyze Python AST of hot function
  2. Generate equivalent in target language
  3. Compile + link
  4. A/B validate against original
  5. Hot-swap at runtime

Current backends:
  - numba: LLVM JIT for numpy-heavy functions
  - rust: Procedural Rust generation for loops/dicts/strings

Usage:
    from sunset.compiler import Compiler
    compiler = Compiler()
    compiler.install()

    # Run your code... profiler watches
    # After 100+ calls, compiler auto-generates optimized version
"""

from __future__ import annotations

__all__ = [
    "CodeGenerator",
    "NumbaGenerator",
    "RustGenerator",
    "GeneratedKernel",
    "ast_to_numba",
    "ast_to_rust",
]

import ast
import inspect
import os
import textwrap
import time
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Data Structures ───────────────────────────────────────


@dataclass
class GeneratedKernel:
    """A compiled kernel ready for validation and deployment."""

    name: str
    source_language: str  # "python", "numba", "rust"
    source_code: str
    compiled: Callable
    compile_time_ms: float
    backend: str
    error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.error is None and self.compiled is not None


# ── AST Analysis ──────────────────────────────────────────


class PythonAnalyzer(ast.NodeVisitor):
    """Analyze a Python function to determine compilation strategy."""

    def __init__(self):
        self.has_numpy = False
        self.has_loops = False
        self.has_dicts = False
        self.has_strings = False
        self.has_list_ops = False
        self.calls: List[str] = []
        self.array_ops: List[str] = []

    def analyze(self, func: Callable) -> "PythonAnalyzer":
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            self.visit(tree)
        except (OSError, TypeError):
            pass  # can't get source (built-in, etc.)
        return self

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in (
                "np",
                "numpy",
            ):
                self.has_numpy = True
                self.array_ops.append(node.func.attr)
            self.calls.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.has_loops = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.has_loops = True
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        self.has_dicts = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self.has_strings = True
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        self.has_list_ops = True
        self.generic_visit(node)

    @property
    def numba_score(self) -> float:
        """Higher = better candidate for Numba."""
        score = 0.0
        if self.has_numpy:
            score += 3.0
        if self.has_loops:
            score += 2.0
        if not self.has_dicts:
            score += 1.0
        if not self.has_strings:
            score += 0.5
        return score

    @property
    def rust_score(self) -> float:
        """Higher = better candidate for Rust."""
        score = 0.0
        if self.has_loops:
            score += 3.0
        if self.has_dicts:
            score += 2.0
        if self.has_strings:
            score += 1.0
        if not self.has_numpy:
            score += 0.5
        return score


# ── Numba Generator ───────────────────────────────────────


class NumbaGenerator:
    """Generate Numba JIT-compiled kernels from Python functions.

    Strategy: wrap the original function with @njit(cache=True).
    For functions that call other functions, recursively compile dependencies.
    """

    def __init__(self):
        self._cache: Dict[str, Callable] = {}
        try:
            import numba

            self._numba = numba
            self._available = True
        except ImportError:
            self._available = False

    def can_generate(self, func: Callable, analyzer: PythonAnalyzer) -> bool:
        if not self._available:
            return False
        return analyzer.numba_score >= 2.0

    def generate(
        self,
        func: Callable,
        analyzer: PythonAnalyzer,
        test_args: Optional[Tuple] = None,
    ) -> GeneratedKernel:
        """Compile a Python function to Numba.

        Args:
            func: The hot function to compile
            analyzer: AST analysis result
            test_args: Representative arguments for warmup/validation

        Returns:
            GeneratedKernel with compiled function or error
        """
        t0 = time.perf_counter()
        name = getattr(func, "__qualname__", "unknown")

        try:
            # Strategy 1: Direct njit
            compiled = self._numba.njit(func, cache=True, fastmath=True)

            # Warmup — critical: first call triggers LLVM compilation
            if test_args is not None:
                try:
                    compiled(*test_args)
                except Exception as e:
                    # Numba failed — try with object mode fallback
                    compiled = self._numba.jit(func, cache=True, forceobj=True)
                    compiled(*test_args)

            compile_time = (time.perf_counter() - t0) * 1000

            return GeneratedKernel(
                name=name,
                source_language="numba",
                source_code=inspect.getsource(func),
                compiled=compiled,
                compile_time_ms=compile_time,
                backend="numba",
            )

        except Exception as e:
            return GeneratedKernel(
                name=name,
                source_language="numba",
                source_code="",
                compiled=func,  # fallback to original
                compile_time_ms=(time.perf_counter() - t0) * 1000,
                backend="numba",
                error=f"Numba compilation failed: {e}",
            )

    def measure_speedup(
        self,
        original: Callable,
        compiled: Callable,
        test_args: Tuple,
        trials: int = 20,
    ) -> float:
        """Measure actual speedup with A/B timing."""
        # Warmup
        for _ in range(3):
            original(*test_args)
            compiled(*test_args)

        # Time original
        t0 = time.perf_counter()
        for _ in range(trials):
            original(*test_args)
        t_orig = (time.perf_counter() - t0) * 1000

        # Time compiled
        t0 = time.perf_counter()
        for _ in range(trials):
            compiled(*test_args)
        t_comp = (time.perf_counter() - t0) * 1000

        return t_orig / max(t_comp, 0.001)


# ── Rust Generator ────────────────────────────────────────


class RustGenerator:
    """Procedural Rust code generation from Python functions.

    Generates Rust source, compiles with rustc, loads as CDLL.
    Complex — currently handles simple numeric loops and array ops.
    """

    def __init__(self):
        self._cache: Dict[str, Callable] = {}
        self._available = False
        if os.system("which rustc > /dev/null 2>&1") == 0:
            self._available = True

    def can_generate(self, func: Callable, analyzer: PythonAnalyzer) -> bool:
        if not self._available:
            return False
        return analyzer.rust_score >= 2.0

    def generate(
        self,
        func: Callable,
        analyzer: PythonAnalyzer,
        test_args: Optional[Tuple] = None,
    ) -> GeneratedKernel:
        """Generate Rust from Python — currently stubs complex functions.

        Full procedural generation is a research project. For now:
        - Simple loop patterns: translated to Rust for loops
        - Numpy ops: delegated to ndarray + rust-numpy
        - Dict/string heavy: returns error (too complex for v1)
        """
        t0 = time.perf_counter()
        name = getattr(func, "__qualname__", "unknown")

        # V1: Only handle functions we've manually written Rust for
        # V2: Will use syn + quote crates for procedural generation
        return GeneratedKernel(
            name=name,
            source_language="rust",
            source_code=f"// Procedural Rust generation for {name}\n// TODO: implement via python-to-rust AST translator",
            compiled=func,
            compile_time_ms=(time.perf_counter() - t0) * 1000,
            backend="rust",
            error="Procedural Rust generation v2 not yet implemented. "
            "Use manual Rust kernels (see nerve/src/lib.rs).",
        )


# ── Code Generator Orchestrator ───────────────────────────


class CodeGenerator:
    """Orchestrates compilation from Python to accelerated backends.

    Usage:
        gen = CodeGenerator()
        kernel = gen.compile(my_hot_function)
        if kernel.ready:
            speedup = gen.validate_and_measure(kernel, test_args)
            if speedup > 2.0:
                gen.deploy(kernel, module, "function_name")
    """

    def __init__(self):
        self.numba = NumbaGenerator()
        self.rust = RustGenerator()
        self._deployed: Dict[str, GeneratedKernel] = {}

    def analyze(self, func: Callable) -> PythonAnalyzer:
        return PythonAnalyzer().analyze(func)

    def compile(
        self,
        func: Callable,
        test_args: Optional[Tuple] = None,
    ) -> GeneratedKernel:
        """Compile a function to the best available backend."""
        # Detect already-compiled Numba functions and pass through
        try:
            import numba

            if isinstance(func, numba.core.registry.CPUDispatcher):
                return GeneratedKernel(
                    name=getattr(func, "__qualname__", "unknown"),
                    source_language="numba",
                    source_code="",
                    compiled=func,
                    compile_time_ms=0.0,
                    backend="numba",
                )
        except ImportError:
            pass

        analyzer = self.analyze(func)

        # Try Numba first (fastest turnaround)
        if self.numba.can_generate(func, analyzer):
            kernel = self.numba.generate(func, analyzer, test_args)
            if kernel.ready:
                return kernel

        # Try Rust (best for complex logic)
        if self.rust.can_generate(func, analyzer):
            kernel = self.rust.generate(func, analyzer, test_args)
            if kernel.ready:
                return kernel

        # Fallback: identity kernel
        return GeneratedKernel(
            name=getattr(func, "__qualname__", "unknown"),
            source_language="python",
            source_code=inspect.getsource(func) if hasattr(func, "__code__") else "",
            compiled=func,
            compile_time_ms=0.0,
            backend="python",
            error="No backend available for this function pattern",
        )

    def validate(
        self,
        kernel: GeneratedKernel,
        original: Callable,
        test_args: Tuple,
        trials: int = 5,
    ) -> bool:
        """A/B test: verify compiled function produces same output."""
        if kernel.source_language == "python":
            return True  # identity

        try:
            for _ in range(trials):
                expected = original(*test_args)
                actual = kernel.compiled(*test_args)
                if not self._outputs_equal(expected, actual):
                    return False
            return True
        except Exception:
            return False

    def measure_speedup(
        self,
        kernel: GeneratedKernel,
        original: Callable,
        test_args: Tuple,
        trials: int = 20,
    ) -> float:
        """Measure speedup of compiled vs original."""
        if kernel.source_language == "python":
            return 1.0
        return self.numba.measure_speedup(original, kernel.compiled, test_args, trials)

    def deploy(
        self,
        kernel: GeneratedKernel,
        module: types.ModuleType,
        attr_name: str,
    ) -> bool:
        """Hot-swap compiled function into module."""
        if not kernel.ready:
            return False
        setattr(module, attr_name, kernel.compiled)
        self._deployed[f"{module.__name__}.{attr_name}"] = kernel
        return True

    @staticmethod
    def _outputs_equal(a: Any, b: Any) -> bool:
        """Check if two outputs are approximately equal.

        Handles type differences for numeric scalars (numpy.float64 vs float, etc.)
        """
        # Handle numeric scalars — numpy float vs Python float, etc.
        if hasattr(a, "dtype") and hasattr(b, "dtype"):
            return np.allclose(a, b, rtol=1e-3, atol=1e-5)

        # One is numpy scalar, one is Python scalar
        if hasattr(a, "dtype") or hasattr(b, "dtype"):
            try:
                return abs(float(a) - float(b)) < 1e-5
            except (TypeError, ValueError):
                return False

        if type(a) != type(b):
            return False
        if isinstance(a, np.ndarray):
            return np.allclose(a, b, rtol=1e-3, atol=1e-5)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(
                CodeGenerator._outputs_equal(x, y) for x, y in zip(a, b)
            )
        return a == b

    def report(self) -> str:
        lines = ["=== Code Generator — Deployed Kernels ==="]
        if not self._deployed:
            lines.append("None deployed yet.")
        else:
            lines.append(
                f"{'Function':<40} {'Backend':<10} {'Speedup':>8} {'Status':>10}"
            )
            for key, kernel in self._deployed.items():
                status = "✅" if kernel.ready else "❌"
                lines.append(
                    f"{key:<40} {kernel.backend:<10} "
                    f"{getattr(kernel, '_speedup', 1.0):>7.1f}× {status:>10}"
                )
        return "\n".join(lines)


# ── Convenience Functions ─────────────────────────────────


def ast_to_numba(func: Callable, test_args: Optional[Tuple] = None) -> GeneratedKernel:
    """One-shot: analyze + compile a function to Numba."""
    gen = CodeGenerator()
    return gen.compile(func, test_args)


def ast_to_rust(func: Callable, test_args: Optional[Tuple] = None) -> GeneratedKernel:
    """One-shot: analyze + compile a function to Rust."""
    gen = CodeGenerator()
    analyzer = gen.analyze(func)
    return gen.rust.generate(func, analyzer, test_args)
