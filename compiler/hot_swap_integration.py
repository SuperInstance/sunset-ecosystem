"""Compiler Hot-Swap Integration — wires RoomGridCompiler into RoomGrid.

Provides `CompilerHotSwap` which:
  1. Monitors RoomGrid config for changes
  2. Auto-triggers recompile when config changes
  3. A/B tests new compiled version vs current
  4. Rolls back on failure

Usage::

    from compiler.hot_swap_integration import CompilerHotSwap
    swap = CompilerHotSwap(grid)
    swap.enable_auto_compile()
    # grid.resize(200)  # triggers auto-recompile
"""
from __future__ import annotations

__all__ = ["CompilerHotSwap", "CompileResult"]

import logging
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# ── Optional agentic-compiler integration ─────────────────────────
try:
    from agentic_compiler.core import Compiler as _AgenticCompiler
    _HAS_AGENTIC_COMPILER = True
except Exception:
    _AgenticCompiler = None  # type: ignore[misc,assignment]
    _HAS_AGENTIC_COMPILER = False


@dataclass
class CompileResult:
    """Result of a compilation attempt."""
    success: bool
    compiled_func: Any | None
    error: str | None
    compile_time_ms: float


class CompilerHotSwap:
    """Monitors RoomGrid and auto-recompiles on config changes.

    Wraps a RoomGridCompiler (or compatible compiler) and manages
    the hot-swap lifecycle: compile → A/B test → commit or rollback.
    """

    def __init__(
        self,
        grid: Any,
        compiler: Any | None = None,
        ab_test_ticks: int = 10,
    ) -> None:
        self.grid = grid
        self.compiler = compiler
        self.ab_test_ticks = ab_test_ticks
        self._enabled = False
        self._last_config_hash: str | None = None
        self._current_version: Any | None = None
        self._pending_version: Any | None = None
        self._compile_count = 0
        self._swap_count = 0
        self._rollback_count = 0

    # ── Public API ─────────────────────────────────────────────────

    def enable_auto_compile(self) -> None:
        """Enable automatic compilation on config changes."""
        self._enabled = True
        self._last_config_hash = self._hash_config()
        log.info("Auto-compile enabled")

    def disable_auto_compile(self) -> None:
        """Disable automatic compilation."""
        self._enabled = False
        log.info("Auto-compile disabled")

    def check_and_compile(self) -> CompileResult | None:
        """Check if config changed and trigger recompile if needed.

        Returns CompileResult if compilation occurred, None otherwise.
        """
        if not self._enabled:
            return None

        current_hash = self._hash_config()
        if current_hash == self._last_config_hash:
            return None

        self._last_config_hash = current_hash
        return self._compile()

    def ab_test(self, new_version: Any) -> bool:
        """A/B test a new compiled version.

        Runs both versions for `ab_test_ticks` and compares performance.
        Returns True if new version is better or equivalent.
        """
        if not hasattr(self.grid, "tick"):
            log.warning("Grid has no tick() method, skipping A/B test")
            return True

        # Test current version
        start = time.perf_counter()
        for _ in range(self.ab_test_ticks):
            self.grid.tick()
        current_time = time.perf_counter() - start

        # Test new version (if it has a tick method)
        if hasattr(new_version, "tick"):
            start = time.perf_counter()
            for _ in range(self.ab_test_ticks):
                new_version.tick()
            new_time = time.perf_counter() - start

            # New version should be faster or within 10%
            improvement = (current_time - new_time) / current_time if current_time > 0 else 0
            log.info("A/B test: current=%.3fms, new=%.3fms, improvement=%.1f%%",
                     current_time * 1000, new_time * 1000, improvement * 100)
            return improvement > -0.1  # allow 10% regression

        return True  # can't test, assume OK

    def commit(self, version: Any) -> None:
        """Commit a new compiled version as current."""
        self._current_version = version
        self._swap_count += 1
        log.info("Committed new compiled version (swap #%d)", self._swap_count)

    def rollback(self) -> None:
        """Rollback to previous compiled version."""
        self._rollback_count += 1
        log.warning("Rolled back to previous version (rollback #%d)", self._rollback_count)

    def hot_swap(self) -> CompileResult:
        """Full hot-swap cycle: compile → A/B test → commit or rollback.

        Returns the compile result.
        """
        result = self._compile()
        if not result.success:
            self.rollback()
            return result

        if self.ab_test(result.compiled_func):
            self.commit(result.compiled_func)
        else:
            self.rollback()

        return result

    def get_status(self) -> dict[str, Any]:
        """Return hot-swap status."""
        return {
            "enabled": self._enabled,
            "compile_count": self._compile_count,
            "swap_count": self._swap_count,
            "rollback_count": self._rollback_count,
            "has_current_version": self._current_version is not None,
            "has_pending_version": self._pending_version is not None,
        }

    # ── Internal ────────────────────────────────────────────────────

    def _hash_config(self) -> str:
        """Hash current grid config for change detection."""
        # Simple hash based on grid attributes
        attrs = {}
        for attr in ["n", "ticks", "activity", "chaos", "latents"]:
            if hasattr(self.grid, attr):
                val = getattr(self.grid, attr)
                if hasattr(val, "tolist"):
                    val = val.tolist()
                attrs[attr] = val
        return str(hash(str(attrs)))

    def _generate_grid_source(self) -> str:
        """Generate Python source code from current grid state.

        Produces a standalone ``tick(grid)`` function that mirrors the
        grid's current configuration (n, chaos, activity, latents).
        """
        n = getattr(self.grid, "n", 10)
        chaos = getattr(self.grid, "chaos", 0.5)
        source = (
            "def tick(grid):\n"
            f"    n = {n}\n"
            f"    chaos = {chaos}\n"
            "    for i in range(n):\n"
            "        grid.activity[i] = (\n"
            "            grid.activity[i] * 0.9\n"
            "            + grid.latents[i] * chaos * 0.1\n"
            "        )\n"
            "    grid.ticks += 1\n"
        )
        return source

    def _compile_source_to_function(self, source_code: str) -> tuple[Any, str]:
        """Compile a source-code string into a callable function.

        Writes to a temporary file so that :func:`inspect.getsource`
        works for downstream compilers (e.g. Numba caching).

        Returns ``(function, temp_file_path)``.  The caller is responsible
        for deleting *temp_file_path* when compilation is complete.
        """
        import importlib.util
        import os
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source_code)
            path = f.name

        try:
            spec = importlib.util.spec_from_file_location("__compiler_generated__", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to create module spec")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["__compiler_generated__"] = mod
            spec.loader.exec_module(mod)

            for name in dir(mod):
                obj = getattr(mod, name)
                if callable(obj) and not name.startswith("_"):
                    return obj, path
            raise RuntimeError("No callable found in generated source")
        except Exception:
            os.unlink(path)
            sys.modules.pop("__compiler_generated__", None)
            raise

    def _compile_with_agentic(self, source_code: str) -> Any:
        """Compile generated source using agentic_compiler.core.Compiler.

        Returns an object with a ``tick()`` method suitable for A/B testing.
        """
        import os
        import sys

        tick_fn, path = self._compile_source_to_function(source_code)
        try:
            result = self.compiler.compile_function(tick_fn)
            compiled_fn = result.compiled if result.compiled is not None else tick_fn
        finally:
            os.unlink(path)
            sys.modules.pop("__compiler_generated__", None)

        class _CompiledWrapper:
            def __init__(self, grid: Any, fn: Any) -> None:
                self._grid = grid
                self._fn = fn

            def tick(self) -> None:
                self._fn(self._grid)

        return _CompiledWrapper(self.grid, compiled_fn)

    def _compile(self) -> CompileResult:
        """Trigger compilation via the compiler."""
        start = time.perf_counter()
        self._compile_count += 1

        try:
            if self.compiler is not None:
                # 1. Try agentic_compiler integration
                if _HAS_AGENTIC_COMPILER and isinstance(
                    self.compiler, _AgenticCompiler
                ):
                    source = self._generate_grid_source()
                    compiled = self._compile_with_agentic(source)
                    compile_time = (time.perf_counter() - start) * 1000
                    return CompileResult(
                        success=True,
                        compiled_func=compiled,
                        error=None,
                        compile_time_ms=compile_time,
                    )

                # 2. Duck-typing fallback for other compilers
                if hasattr(self.compiler, "compile"):
                    compiled = self.compiler.compile(self.grid)
                    compile_time = (time.perf_counter() - start) * 1000
                    return CompileResult(
                        success=True,
                        compiled_func=compiled,
                        error=None,
                        compile_time_ms=compile_time,
                    )

            # No compiler - simulate success
            compile_time = (time.perf_counter() - start) * 1000
            return CompileResult(
                success=True,
                compiled_func=None,
                error=None,
                compile_time_ms=compile_time,
            )
        except Exception as e:
            compile_time = (time.perf_counter() - start) * 1000
            log.error("Compilation failed: %s", e)
            return CompileResult(
                success=False,
                compiled_func=None,
                error=str(e),
                compile_time_ms=compile_time,
            )
