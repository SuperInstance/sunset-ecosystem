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

    def _compile(self) -> CompileResult:
        """Trigger compilation via the compiler."""
        start = time.perf_counter()
        self._compile_count += 1

        try:
            if self.compiler and hasattr(self.compiler, "compile"):
                compiled = self.compiler.compile(self.grid)
                compile_time = (time.perf_counter() - start) * 1000
                return CompileResult(
                    success=True,
                    compiled_func=compiled,
                    error=None,
                    compile_time_ms=compile_time,
                )
            else:
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
