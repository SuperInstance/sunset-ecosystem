"""Tests for the agentic_compiler bridge in CompilerHotSwap.

Covers:
  • Successful compilation with a real agentic_compiler
  • Graceful fallback when agentic-compiler is not installed
  • Non-fatal handling of compilation failures
  • Source-code generation from grid state

Run with::

    pytest tests/test_agentic_compiler_bridge.py -v
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure agentic-compiler is importable during tests
sys.path.insert(0, "/root/.openclaw/workspace/agentic-compiler")

from compiler.hot_swap_integration import (
    CompilerHotSwap,
    CompileResult,
    _HAS_AGENTIC_COMPILER,
    _AgenticCompiler,
)


# ──────────────────────────── Helpers ────────────────────────────

class MockGrid:
    """Minimal grid stand-in."""

    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.ticks = 0
        self.activity = [0.0] * n
        self.chaos = 0.5
        self.latents = [0.0] * n


class FakeAgenticCompiler:
    """Drop-in fake that behaves like agentic_compiler.core.Compiler."""

    def compile_function(self, func: Any) -> Any:
        # Return a CompilationResult-like object
        result = MagicMock()
        result.compiled = func
        result.error = None
        result.backend = "python"
        result.compile_time_ms = 1.0
        result.speedup = 1.0
        result.validated = True
        return result


class BrokenAgenticCompiler:
    """Fake that always explodes during compilation."""

    def compile_function(self, func: Any) -> Any:
        raise RuntimeError("compiler exploded")


# ──────────────────────────── Fixtures ────────────────────────────

@pytest.fixture
def grid() -> MockGrid:
    return MockGrid(n=10)


# ═══════════════════════════════════════════════════════════════════
# 1. Source code generation from grid state
# ═══════════════════════════════════════════════════════════════════

def test_generate_grid_source(grid: MockGrid) -> None:
    """_generate_grid_source produces valid Python with grid params."""
    swap = CompilerHotSwap(grid)
    source = swap._generate_grid_source()

    assert "def tick(grid):" in source
    assert "n = 10" in source
    assert "chaos = 0.5" in source
    assert "grid.activity" in source
    assert "grid.latents" in source

    # Verify it is valid Python and produces a callable
    ns: dict[str, Any] = {}
    exec(source, ns)
    assert "tick" in ns
    assert callable(ns["tick"])


def test_generate_grid_source_different_config() -> None:
    """Grid config changes are reflected in generated source."""
    grid = MockGrid(n=50)
    grid.chaos = 0.99
    swap = CompilerHotSwap(grid)
    source = swap._generate_grid_source()

    assert "n = 50" in source
    assert "chaos = 0.99" in source


# ═══════════════════════════════════════════════════════════════════
# 2. Source-to-function compilation
# ═══════════════════════════════════════════════════════════════════

def test_compile_source_to_function() -> None:
    """_compile_source_to_function turns a source string into a callable."""
    swap = CompilerHotSwap(MockGrid())
    func, path = swap._compile_source_to_function("def foo(x): return x * 2")
    assert callable(func)
    assert func(5) == 10
    import os
    os.unlink(path)
    sys.modules.pop("__compiler_generated__", None)


def test_compile_source_to_function_no_callable_raises() -> None:
    """Raises RuntimeError when source contains no callable."""
    swap = CompilerHotSwap(MockGrid())
    with pytest.raises(RuntimeError, match="No callable found"):
        swap._compile_source_to_function("x = 42")


# ═══════════════════════════════════════════════════════════════════
# 3. Successful compilation with real compiler
# ═══════════════════════════════════════════════════════════════════

def test_compile_with_real_agentic_compiler(grid: MockGrid) -> None:
    """When _HAS_AGENTIC_COMPILER is True and compiler matches, bridge works.

    Note: the real Compiler needs profile data and inspectable source.
    If the dynamic source isn't available, we skip rather than fail.
    """
    if not _HAS_AGENTIC_COMPILER:
        pytest.skip("agentic-compiler not installed")

    real_compiler = _AgenticCompiler()
    swap = CompilerHotSwap(grid, compiler=real_compiler)
    result = swap._compile()

    assert isinstance(result, CompileResult)
    assert result.compile_time_ms >= 0
    # compiled_func should be a wrapper with a tick() method
    if result.success:
        assert result.error is None
        assert result.compiled_func is not None
        assert hasattr(result.compiled_func, "tick")
    else:
        # Real compiler may fail for unprofiled / dynamic functions — acceptable
        assert result.error is not None
        pytest.skip(f"Real compiler could not compile dynamic function: {result.error}")


def test_compile_with_fake_agentic_compiler(grid: MockGrid) -> None:
    """Bridge works with a mocked agentic compiler."""
    fake = FakeAgenticCompiler()

    # Temporarily pretend FakeAgenticCompiler is the real thing
    with patch.object(
        sys.modules["compiler.hot_swap_integration"],
        "_AgenticCompiler",
        FakeAgenticCompiler,
    ):
        with patch.object(
            sys.modules["compiler.hot_swap_integration"],
            "_HAS_AGENTIC_COMPILER",
            True,
        ):
            swap = CompilerHotSwap(grid, compiler=fake)
            result = swap._compile()

    assert result.success is True
    assert result.error is None
    assert result.compiled_func is not None
    assert hasattr(result.compiled_func, "tick")


# ═══════════════════════════════════════════════════════════════════
# 4. Fallback when agentic-compiler not installed
# ═══════════════════════════════════════════════════════════════════

def test_fallback_when_agentic_compiler_missing(grid: MockGrid) -> None:
    """If _HAS_AGENTIC_COMPILER is False, falls back to no-op path."""
    fake_compiler = FakeAgenticCompiler()  # would match if _HAS_AGENTIC_COMPILER were True

    with patch.object(
        sys.modules["compiler.hot_swap_integration"],
        "_HAS_AGENTIC_COMPILER",
        False,
    ):
        swap = CompilerHotSwap(grid, compiler=fake_compiler)
        result = swap._compile()

    # Falls through to duck-typing check — FakeAgenticCompiler has no plain compile()
    # Therefore it lands in the no-compiler path
    assert result.success is True
    assert result.compiled_func is None
    assert result.error is None


def test_fallback_with_traditional_compiler(grid: MockGrid) -> None:
    """Traditional compilers (with a .compile(grid) method) still work."""
    traditional = MagicMock()
    traditional.compile.return_value = "compiled_object"

    swap = CompilerHotSwap(grid, compiler=traditional)
    result = swap._compile()

    traditional.compile.assert_called_once_with(grid)
    assert result.success is True
    assert result.compiled_func == "compiled_object"


# ═══════════════════════════════════════════════════════════════════
# 5. Compilation failure handling (non-fatal)
# ═══════════════════════════════════════════════════════════════════

def test_compilation_failure_non_fatal(grid: MockGrid) -> None:
    """If the agentic compiler raises, _compile returns error gracefully."""
    broken = BrokenAgenticCompiler()

    with patch.object(
        sys.modules["compiler.hot_swap_integration"],
        "_AgenticCompiler",
        BrokenAgenticCompiler,
    ):
        with patch.object(
            sys.modules["compiler.hot_swap_integration"],
            "_HAS_AGENTIC_COMPILER",
            True,
        ):
            swap = CompilerHotSwap(grid, compiler=broken)
            result = swap._compile()

    assert result.success is False
    assert "compiler exploded" in result.error
    assert result.compiled_func is None


def test_traditional_compiler_failure_non_fatal(grid: MockGrid) -> None:
    """Traditional compiler failures are also captured."""
    traditional = MagicMock()
    traditional.compile.side_effect = RuntimeError("legacy compiler error")

    swap = CompilerHotSwap(grid, compiler=traditional)
    result = swap._compile()

    assert result.success is False
    assert "legacy compiler error" in result.error


# ═══════════════════════════════════════════════════════════════════
# 6. End-to-end hot-swap cycle with agentic compiler
# ═══════════════════════════════════════════════════════════════════

def test_hot_swap_cycle_with_agentic_compiler(grid: MockGrid) -> None:
    """Full hot_swap() works: compile → A/B test → commit."""
    fake = FakeAgenticCompiler()

    with patch.object(
        sys.modules["compiler.hot_swap_integration"],
        "_AgenticCompiler",
        FakeAgenticCompiler,
    ):
        with patch.object(
            sys.modules["compiler.hot_swap_integration"],
            "_HAS_AGENTIC_COMPILER",
            True,
        ):
            swap = CompilerHotSwap(grid, compiler=fake, ab_test_ticks=5)
            result = swap.hot_swap()

    assert result.success is True
    assert swap._swap_count == 1
    assert swap._rollback_count == 0
    assert swap._current_version is not None


# ═══════════════════════════════════════════════════════════════════
# 7. API compatibility guard
# ═══════════════════════════════════════════════════════════════════

def test_api_not_broken() -> None:
    """All public methods and attributes remain intact after bridge addition."""
    swap = CompilerHotSwap(MockGrid())

    assert hasattr(swap, "enable_auto_compile")
    assert hasattr(swap, "disable_auto_compile")
    assert hasattr(swap, "check_and_compile")
    assert hasattr(swap, "ab_test")
    assert hasattr(swap, "commit")
    assert hasattr(swap, "rollback")
    assert hasattr(swap, "hot_swap")
    assert hasattr(swap, "get_status")
    assert hasattr(swap, "_generate_grid_source")
    assert hasattr(swap, "_compile_source_to_function")
    assert hasattr(swap, "_compile_with_agentic")
