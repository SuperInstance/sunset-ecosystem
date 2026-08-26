"""Tests for CompilerHotSwap integration."""

from __future__ import annotations

import time

import pytest

from compiler.hot_swap_integration import CompilerHotSwap, CompileResult


class MockGrid:
    """Mock RoomGrid for testing."""

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
        # Realistic workload so A/B timing isn't dominated by call overhead
        for i in range(self.n):
            self.activity[i] = (
                self.activity[i] * 0.9 + self.latents[i] * self.chaos * 0.1
            )


class MockCompiler:
    """Mock compiler that returns a faster tick function."""

    def __init__(self, succeed: bool = True, fast: bool = True) -> None:
        self.succeed = succeed
        self.fast = fast

    def compile(self, grid: Any) -> Any:
        if not self.succeed:
            raise RuntimeError("Compilation failed")
        return MockCompiledGrid(grid, fast=self.fast)


class MockCompiledGrid:
    """Mock compiled grid — may be faster or slower."""

    def __init__(self, grid: Any, fast: bool = True) -> None:
        self._grid = grid
        self.fast = fast

    def tick(self) -> None:
        if not self.fast:
            time.sleep(0.001)
        # Fast path: skip the per-element loop, just bump counters
        self._grid.tick_count += 1
        self._grid.ticks += 1


class TestEnableDisable:
    def test_enable_sets_flag(self):
        grid = MockGrid()
        swap = CompilerHotSwap(grid)
        swap.enable_auto_compile()
        assert swap._enabled

    def test_disable_clears_flag(self):
        grid = MockGrid()
        swap = CompilerHotSwap(grid)
        swap.enable_auto_compile()
        swap.disable_auto_compile()
        assert not swap._enabled


class TestConfigHash:
    def test_same_config_same_hash(self):
        grid = MockGrid(n=10)
        swap = CompilerHotSwap(grid)
        h1 = swap._hash_config()
        h2 = swap._hash_config()
        assert h1 == h2

    def test_different_config_different_hash(self):
        grid1 = MockGrid(n=10)
        grid2 = MockGrid(n=20)
        swap1 = CompilerHotSwap(grid1)
        swap2 = CompilerHotSwap(grid2)
        assert swap1._hash_config() != swap2._hash_config()


class TestCheckAndCompile:
    def test_no_compile_when_disabled(self):
        grid = MockGrid()
        swap = CompilerHotSwap(grid)
        result = swap.check_and_compile()
        assert result is None

    def test_compile_on_config_change(self):
        grid = MockGrid(n=10)
        swap = CompilerHotSwap(grid)
        swap.enable_auto_compile()
        # First check sets baseline
        result = swap.check_and_compile()
        assert result is None  # no change yet
        # Change config
        grid.n = 20
        result = swap.check_and_compile()
        assert result is not None
        assert result.success

    def test_no_compile_when_unchanged(self):
        grid = MockGrid(n=10)
        swap = CompilerHotSwap(grid)
        swap.enable_auto_compile()
        swap.check_and_compile()  # set baseline
        result = swap.check_and_compile()
        assert result is None


class TestCompile:
    def test_compile_with_compiler(self):
        grid = MockGrid()
        compiler = MockCompiler()
        swap = CompilerHotSwap(grid, compiler=compiler)
        result = swap._compile()
        assert result.success
        assert result.compiled_func is not None

    def test_compile_failure(self):
        grid = MockGrid()
        compiler = MockCompiler(succeed=False)
        swap = CompilerHotSwap(grid, compiler=compiler)
        result = swap._compile()
        assert not result.success
        assert result.error is not None

    def test_compile_without_compiler(self):
        grid = MockGrid()
        swap = CompilerHotSwap(grid)
        result = swap._compile()
        assert result.success  # simulates success
        assert result.compiled_func is None


class TestABTest:
    def test_ab_test_passes_for_fast_version(self):
        grid = MockGrid()
        compiler = MockCompiler(fast=True)
        swap = CompilerHotSwap(grid, compiler=compiler)
        result = swap._compile()
        assert swap.ab_test(result.compiled_func)

    def test_ab_test_fails_for_slow_version(self):
        grid = MockGrid()
        compiler = MockCompiler(fast=False)
        swap = CompilerHotSwap(grid, compiler=compiler, ab_test_ticks=50)
        result = swap._compile()
        # Slow version should fail A/B test (more than 10% slower)
        assert not swap.ab_test(result.compiled_func)


class TestHotSwap:
    def test_hot_swap_success(self):
        grid = MockGrid()
        compiler = MockCompiler(fast=True)
        swap = CompilerHotSwap(grid, compiler=compiler, ab_test_ticks=50)
        result = swap.hot_swap()
        assert result.success
        assert swap._swap_count == 1
        assert swap._rollback_count == 0

    def test_hot_swap_rollback_on_slow(self):
        grid = MockGrid()
        compiler = MockCompiler(fast=False)
        swap = CompilerHotSwap(grid, compiler=compiler, ab_test_ticks=50)
        result = swap.hot_swap()
        assert result.success  # compilation succeeded
        assert swap._swap_count == 0
        assert swap._rollback_count == 1

    def test_hot_swap_rollback_on_compile_fail(self):
        grid = MockGrid()
        compiler = MockCompiler(succeed=False)
        swap = CompilerHotSwap(grid, compiler=compiler)
        result = swap.hot_swap()
        assert not result.success
        assert swap._swap_count == 0
        assert swap._rollback_count == 1


class TestStatus:
    def test_status_fields(self):
        grid = MockGrid()
        swap = CompilerHotSwap(grid)
        status = swap.get_status()
        assert not status["enabled"]
        assert status["compile_count"] == 0
        assert status["swap_count"] == 0
        assert status["rollback_count"] == 0

    def test_status_after_swap(self):
        grid = MockGrid()
        compiler = MockCompiler(fast=True)
        swap = CompilerHotSwap(grid, compiler=compiler, ab_test_ticks=50)
        swap.hot_swap()
        status = swap.get_status()
        assert status["compile_count"] == 1
        assert status["swap_count"] == 1


# Need to import Any for type hints
from typing import Any
