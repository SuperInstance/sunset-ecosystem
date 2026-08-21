"""Comprehensive pytest test suite for compiler/hot_swap_integration.py.

Tests the CompilerHotSwap lifecycle with mock grid objects and a mock compiler.
All tests are self-contained and runnable with::

    pytest tests/test_compiler_hot_swap.py -v
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from compiler.hot_swap_integration import CompilerHotSwap, CompileResult


# ──────────────────────────── Helpers ────────────────────────────


class MockGrid:
    """Minimal grid stand-in with the attributes _hash_config looks for."""

    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.ticks = 0
        self.activity = [0.0] * n
        self.chaos = 0.5
        self.latents = [0.0] * n
        self.tick_count = 0

    def tick(self) -> None:
        self.tick_count += 1
        self.ticks += 1


class FastCompiledGrid:
    """Compiled grid with negligible overhead."""

    def __init__(self, grid: MockGrid) -> None:
        self._grid = grid

    def tick(self) -> None:
        self._grid.tick()


class SlowCompiledGrid:
    """Compiled grid that artificially sleeps on each tick."""

    def __init__(self, grid: MockGrid, sleep_sec: float = 0.001) -> None:
        self._grid = grid
        self._sleep_sec = sleep_sec

    def tick(self) -> None:
        import time

        time.sleep(self._sleep_sec)
        self._grid.tick()


class FakeCompiler:
    """Compiler that returns a pre-configured compiled object or raises."""

    def __init__(
        self, compiled: Any | None = None, exc: Exception | None = None
    ) -> None:
        self.compiled = compiled
        self.exc = exc

    def compile(self, grid: Any) -> Any:
        if self.exc is not None:
            raise self.exc
        return self.compiled


# ──────────────────────────── Fixtures ────────────────────────────


@pytest.fixture
def grid() -> MockGrid:
    return MockGrid(n=10)


@pytest.fixture
def swap(grid: MockGrid) -> CompilerHotSwap:
    return CompilerHotSwap(grid)


@pytest.fixture
def mock_compiler() -> MagicMock:
    compiler = MagicMock()
    compiler.compile.return_value = None
    return compiler


# ═══════════════════════════════════════════════════════════════════
# 1. Auto-compile triggers on config change
# ═══════════════════════════════════════════════════════════════════


def test_auto_compile_triggers_on_config_change(grid: MockGrid) -> None:
    """Changing grid.n after enabling auto-compile triggers recompilation."""
    swap = CompilerHotSwap(grid)
    swap.enable_auto_compile()

    # Baseline check — no change yet
    assert swap.check_and_compile() is None
    assert swap._compile_count == 0

    # Mutate config
    grid.n = 20
    result = swap.check_and_compile()

    assert isinstance(result, CompileResult)
    assert result.success is True
    assert swap._compile_count == 1


# ═══════════════════════════════════════════════════════════════════
# 2. A/B test compares performance
# ═══════════════════════════════════════════════════════════════════


def test_ab_test_compares_performance(grid: MockGrid) -> None:
    """A/B test runs both versions and accepts fast, rejects slow."""
    # Fast version should pass
    fast = FastCompiledGrid(grid)
    swap = CompilerHotSwap(grid, ab_test_ticks=10)
    assert swap.ab_test(fast) is True

    # Slow version should fail (artificially inject timings)
    slow = SlowCompiledGrid(grid, sleep_sec=0.005)
    swap_slow = CompilerHotSwap(grid, ab_test_ticks=50)
    assert swap_slow.ab_test(slow) is False


# ═══════════════════════════════════════════════════════════════════
# 3. Rollback on compile failure
# ═══════════════════════════════════════════════════════════════════


def test_rollback_on_compile_failure(grid: MockGrid) -> None:
    """If compile raises, hot_swap calls rollback and does not commit."""
    compiler = FakeCompiler(exc=RuntimeError("compile failed"))
    swap = CompilerHotSwap(grid, compiler=compiler)

    result = swap.hot_swap()

    assert result.success is False
    assert result.error == "compile failed"
    assert swap._rollback_count == 1
    assert swap._swap_count == 0
    assert swap._current_version is None


# ═══════════════════════════════════════════════════════════════════
# 4. Status reporting
# ═══════════════════════════════════════════════════════════════════


def test_status_reporting(swap: CompilerHotSwap) -> None:
    """get_status returns correct fields and values."""
    status = swap.get_status()

    assert status["enabled"] is False
    assert status["compile_count"] == 0
    assert status["swap_count"] == 0
    assert status["rollback_count"] == 0
    assert status["has_current_version"] is False
    assert status["has_pending_version"] is False

    # After a successful commit
    swap.commit("v1")
    status = swap.get_status()
    assert status["swap_count"] == 1
    assert status["has_current_version"] is True

    # After a rollback
    swap.rollback()
    status = swap.get_status()
    assert status["rollback_count"] == 1


# ═══════════════════════════════════════════════════════════════════
# 5. Disable auto-compile
# ═══════════════════════════════════════════════════════════════════


def test_disable_auto_compile(grid: MockGrid) -> None:
    """When disabled, check_and_compile returns None even if config changes."""
    swap = CompilerHotSwap(grid)
    swap.enable_auto_compile()
    swap.check_and_compile()  # baseline

    # Disable
    swap.disable_auto_compile()
    assert swap._enabled is False

    # Mutate config
    grid.n = 99
    result = swap.check_and_compile()

    assert result is None
    assert swap._compile_count == 0


# ═══════════════════════════════════════════════════════════════════
# 6. Hash config changes
# ═══════════════════════════════════════════════════════════════════


def test_hash_config_changes(grid: MockGrid) -> None:
    """_hash_config returns different hashes for different configs."""
    swap = CompilerHotSwap(grid)

    h1 = swap._hash_config()

    # Change tracked attribute
    grid.n = 99
    h2 = swap._hash_config()
    assert h1 != h2

    # Change another tracked attribute
    grid.chaos = 0.99
    h3 = swap._hash_config()
    assert h2 != h3

    # Revert back — hash should not equal original because other attrs changed
    grid.n = 10
    h4 = swap._hash_config()
    assert h4 != h1

    # Untracked attribute does not affect hash
    grid.untracked_attr = "hello"  # type: ignore[attr-defined]
    h5 = swap._hash_config()
    assert h4 == h5


# ═══════════════════════════════════════════════════════════════════
# 7. Commit updates current version
# ═══════════════════════════════════════════════════════════════════


def test_commit_updates_current_version(grid: MockGrid) -> None:
    """commit stores the new version and increments swap_count."""
    swap = CompilerHotSwap(grid)

    swap.commit("version_a")
    assert swap._current_version == "version_a"
    assert swap._swap_count == 1

    swap.commit("version_b")
    assert swap._current_version == "version_b"
    assert swap._swap_count == 2


# ═══════════════════════════════════════════════════════════════════
# 8. Hot-swap full cycle with mock compiler
# ═══════════════════════════════════════════════════════════════════


def test_hot_swap_full_cycle(grid: MockGrid) -> None:
    """End-to-end hot swap with a mock compiler: compile → A/B → commit."""
    compiled_obj = FastCompiledGrid(grid)
    mock_compiler = MagicMock()
    mock_compiler.compile.return_value = compiled_obj

    swap = CompilerHotSwap(grid, compiler=mock_compiler, ab_test_ticks=10)

    # Patch perf_counter so the fast version registers as an improvement
    fast_times = [0.0, 0.100, 0.100, 0.050]  # current=0.100, new=0.050 → 50% faster
    perf_idx = 0

    def fake_perf_fast() -> float:
        nonlocal perf_idx
        val = fast_times[perf_idx % len(fast_times)]
        perf_idx += 1
        return val

    with patch("time.perf_counter", side_effect=fake_perf_fast):
        result = swap.hot_swap()

    # Compilation was invoked
    mock_compiler.compile.assert_called_once_with(grid)
    assert result.success is True
    assert result.compiled_func is compiled_obj

    # A/B passed → committed
    assert swap._swap_count == 1
    assert swap._rollback_count == 0
    assert swap._current_version is compiled_obj

    # --- Second cycle: slow version should rollback ---
    slow_obj = SlowCompiledGrid(grid, sleep_sec=0.005)
    mock_compiler.compile.return_value = slow_obj
    swap.ab_test_ticks = 50

    # Patch perf_counter so the slow version registers as >10% regression
    slow_times = [
        0.0,
        0.100,
        0.100,
        0.111,
    ]  # current=0.100, new=0.011*50 → 11% regression
    perf_idx = 0

    def fake_perf_slow() -> float:
        nonlocal perf_idx
        val = slow_times[perf_idx % len(slow_times)]
        perf_idx += 1
        return val

    with patch("time.perf_counter", side_effect=fake_perf_slow):
        result2 = swap.hot_swap()

    assert result2.success is True
    assert swap._rollback_count == 1
    # current_version stays the previous compiled_obj
    assert swap._current_version is compiled_obj

    # --- Third cycle: simulate compile failure ---
    mock_compiler.compile.side_effect = RuntimeError("compile error")
    result3 = swap.hot_swap()
    assert result3.success is False
    assert swap._rollback_count == 2
