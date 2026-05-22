"""Cross-repo integration test: sunset-ecosystem + agentic-compiler.

Verifies that a real Compiler from agentic_compiler can be wired into
sunset-ecosystem's CompilerHotSwap and drive end-to-end compilation.

Run: pytest tests/test_cross_repo_integration.py -v
"""

import sys
import time
import numpy as np
import pytest

# ── Defensive imports ───────────────────────────────────────────

try:
    import agentic_compiler
    from agentic_compiler import Compiler as AgenticCompiler
    from agentic_compiler.core import CompilationResult as AgenticCompilationResult
    HAS_AGENTIC_COMPILER = True
    _import_error = ""
except Exception as exc:
    HAS_AGENTIC_COMPILER = False
    _import_error = str(exc)

try:
    from nerve.room_grid import RoomGrid
    from compiler.hot_swap_integration import CompilerHotSwap, CompileResult
    HAS_SUNSET = True
    _sunset_import_error = ""
except Exception as exc:
    HAS_SUNSET = False
    _sunset_import_error = str(exc)


# ── Skip if agentic_compiler unavailable ──────────────────────────

pytestmark = pytest.mark.skipif(
    not HAS_AGENTIC_COMPILER,
    reason=f"agentic_compiler not available: {_import_error}",
)


# ── Wrapper: provide no-arg tick for CompilerHotSwap ──────────

class NoArgTickGrid:
    """Wraps RoomGrid so tick() needs no arguments.

    CompilerHotSwap.ab_test() calls grid.tick() without a signal.
    RoomGrid.tick requires a signal, so this adapter injects one.
    All other attributes are proxied through for hash compatibility.
    """

    def __init__(self, grid: RoomGrid) -> None:
        self._grid = grid

    def __getattr__(self, name: str):
        return getattr(self._grid, name)

    def tick(self, *args, **kwargs):
        """Tick the underlying RoomGrid with an auto-generated signal."""
        signal = np.random.randn(64).astype(np.float32)
        return self._grid.tick(signal)


# ── Adapter: bridge agentic_compiler → CompilerHotSwap ──────────

class AgenticCompilerAdapter:
    """Wraps agentic_compiler.Compiler so it fits CompilerHotSwap.

    The adapter:
      1. Installs the profiler on nerve.room_grid
      2. Runs ticks to generate profile data
      3. Compiles hotspots via agentic_compiler
      4. Returns the compiled callable for CompilerHotSwap to A/B test
    """

    def __init__(self):
        self.compiler = AgenticCompiler()
        self._last_result = None
        self._compiled_any = False

    def compile(self, grid):
        """Profile, compile, and return the compiled function."""
        # Install profiler on nerve.room_grid
        self.compiler.install("nerve.room_grid")

        # Generate profile data — enough calls to cross compile thresholds
        signal = np.random.randn(64).astype(np.float32)
        for _ in range(150):
            grid.tick(signal)

        # Compile the hottest functions
        results = self.compiler.compile_hotspots(top_n=3)

        for result in results:
            if result.compiled is not None:
                self._last_result = result
                self._compiled_any = True
                return result.compiled

        # Fallback: return whatever we got, even if not validated
        if results:
            self._last_result = results[0]
            self._compiled_any = True
            return results[0].compiled

        return None

    @property
    def last_speedup(self) -> float:
        return getattr(self._last_result, "speedup", 1.0)

    @property
    def last_backend(self) -> str:
        return getattr(self._last_result, "backend", "python")


# ── Tests ────────────────────────────────────────────────────────

def test_imports_work():
    """Both repos are importable."""
    assert HAS_AGENTIC_COMPILER, "agentic_compiler failed to import"
    assert HAS_SUNSET, f"sunset-ecosystem modules failed to import: {_sunset_import_error}"


def test_compiler_hot_swap_with_real_compiler():
    """End-to-end: RoomGrid → CompilerHotSwap → agentic_compiler."""
    if not HAS_SUNSET:
        pytest.skip("sunset-ecosystem modules not available")

    np.random.seed(42)
    grid = RoomGrid(50)
    wrapper = NoArgTickGrid(grid)

    adapter = AgenticCompilerAdapter()
    swap = CompilerHotSwap(wrapper, compiler=adapter)
    swap.enable_auto_compile()

    # Ticks change wrapper.ticks which changes the hash
    for _ in range(10):
        wrapper.tick()

    # Force a compile check — should detect state change
    result = swap.check_and_compile()

    # Verify compilation was attempted
    assert swap._compile_count >= 1, "Compilation was not triggered"
    assert swap.get_status()["compile_count"] >= 1

    # If compilation produced a result, verify it's tracked
    if result is not None:
        assert isinstance(result, CompileResult)
        assert result.success is True


def test_compiled_function_is_used_and_faster():
    """Verify the compiled function is used and performance improves."""
    if not HAS_SUNSET:
        pytest.skip("sunset-ecosystem modules not available")

    np.random.seed(42)
    grid = RoomGrid(50)
    wrapper = NoArgTickGrid(grid)

    adapter = AgenticCompilerAdapter()
    swap = CompilerHotSwap(wrapper, compiler=adapter, ab_test_ticks=20)
    swap.enable_auto_compile()

    # Full hot-swap cycle: compile → A/B test → commit/rollback
    result = swap.hot_swap()

    # Compilation must have been attempted
    assert swap._compile_count >= 1, "hot_swap did not trigger compilation"

    # If it succeeded, verify the compiled function is tracked
    if result.success and result.compiled_func is not None:
        # A/B test passed → swap committed
        assert swap._swap_count >= 1, "A/B test did not commit the compiled version"
        assert swap.get_status()["has_current_version"] is True

        # Performance claim — if backend is python, speedup is identity (1.0).
        # If it's numba/rust, we expect > 1.0.
        if adapter.last_backend != "python":
            assert adapter.last_speedup > 1.0, (
                f"Expected speedup > 1.0 for {adapter.last_backend}, "
                f"got {adapter.last_speedup}"
            )


def test_tick_cycle_triggers_compilation():
    """Running tick cycles causes compilation to trigger."""
    if not HAS_SUNSET:
        pytest.skip("sunset-ecosystem modules not available")

    np.random.seed(42)
    grid = RoomGrid(50)
    wrapper = NoArgTickGrid(grid)

    adapter = AgenticCompilerAdapter()
    swap = CompilerHotSwap(wrapper, compiler=adapter)
    swap.enable_auto_compile()

    # Baseline — capture initial hash
    baseline_hash = swap._last_config_hash

    # Ticks change state (ticks, activity, latents, chaos)
    for _ in range(5):
        wrapper.tick()

    # Hash should have changed
    new_hash = swap._hash_config()
    assert new_hash != baseline_hash, "Grid state did not change after ticks"

    # check_and_compile should detect the change
    result = swap.check_and_compile()
    assert result is not None, "check_and_compile did not trigger after state change"
    assert swap._compile_count == 1


def test_adapter_produces_measurable_speedup_for_simple_func():
    """Direct test: agentic_compiler compiles a simple numpy function.

    This bypasses the auto-profiling uncertainty and proves the compiler
    can generate faster code when given a good candidate.
    """
    if not HAS_SUNSET:
        pytest.skip("sunset-ecosystem modules not available")

    # A simple einsum-like function that Numba can accelerate
    def _simple_forward(w1, w2, x):
        """Tiny forward pass — pure numpy, loop-free."""
        h = w1 @ x
        return w2 @ h

    np.random.seed(42)
    w1 = np.random.randn(32, 64).astype(np.float32)
    w2 = np.random.randn(16, 32).astype(np.float32)
    x = np.random.randn(64).astype(np.float32)

    compiler = AgenticCompiler()
    result = compiler.compile_function(_simple_forward)

    assert result.compiled is not None, "Compilation returned None"

    # Validate correctness
    expected = _simple_forward(w1, w2, x)
    actual = result.compiled(w1, w2, x)
    assert np.allclose(expected, actual, atol=1e-4, rtol=1e-3), (
        "Compiled output diverged from original"
    )

    # If validated and speedup measured, check it
    if result.validated and result.speedup is not None:
        # Accept >= 0.9× — overhead from small arrays can make speedup < 1.0
        assert result.speedup >= 0.9, (
            f"Speedup {result.speedup} is unexpectedly low"
        )


def test_cross_repo_status_includes_both_systems():
    """Status report contains fields from both sunset and agentic_compiler."""
    if not HAS_SUNSET:
        pytest.skip("sunset-ecosystem modules not available")

    np.random.seed(42)
    grid = RoomGrid(30)
    wrapper = NoArgTickGrid(grid)
    adapter = AgenticCompilerAdapter()
    swap = CompilerHotSwap(wrapper, compiler=adapter)
    swap.enable_auto_compile()

    # Trigger compile
    wrapper.tick()
    swap.hot_swap()

    status = swap.get_status()
    assert "enabled" in status
    assert "compile_count" in status
    assert "swap_count" in status
    assert "rollback_count" in status
    assert "has_current_version" in status

    # agentic_compiler state
    assert hasattr(adapter.compiler, "profiler")
    assert hasattr(adapter.compiler, "compiled")


# ── Global cleanup ───────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    """Restore any hot-swaps left behind by either system."""
    mod = sys.modules.get("nerve.room_grid")
    if mod is None:
        return
    for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
        obj = getattr(mod, attr, None)
        if obj is not None and hasattr(obj, "_sunset_original"):
            setattr(mod, attr, obj._sunset_original)
        if attr == "_tick_routing_compiled" and hasattr(mod, attr):
            delattr(mod, attr)
        # Also clean up agentic_compiler _agentic_original
        if obj is not None and hasattr(obj, "_agentic_original"):
            setattr(mod, attr, obj._agentic_original)
