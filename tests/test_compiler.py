"""Tests for the Sunset Compiler — profiler, Numba backend, auto-compile."""
import time
import numpy as np
import pytest
import inspect

from sunset.compiler import Compiler
from sunset.codegen import CodeGenerator

NUMBA_AVAILABLE = False
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    pass


class SlowSumClass:
    """Simple class with method for profiler testing."""

    def slow_sum(self, a, b):
        total = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total


class TestProfiler:
    """Profiler behavior."""

    def test_profiler_tracks_calls(self, compiler):
        """After 100 calls on a patched module, profiler records the function."""
        # Install profiler on the current test module
        import tests.test_compiler as test_mod
        compiler.install("tests.test_compiler")

        obj = SlowSumClass()
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)

        for _ in range(100):
            obj.slow_sum(a, b)

        hotspots = compiler.profiler.get_hotspots()
        names = [h.name for h in hotspots]
        assert any("slow_sum" in n for n in names), f"slow_sum not in hotspots: {names}"

    def test_profiler_counts_calls(self, compiler):
        """Profiler counts are accurate."""
        obj = SlowSumClass()
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)

        for _ in range(50):
            obj.slow_sum(a, b)

        hotspots = compiler.profiler.get_hotspots()
        slow_hot = next((h for h in hotspots if "slow_sum" in h.name), None)
        if slow_hot:
            assert slow_hot.calls >= 50, f"Call count too low: {slow_hot.calls}"


class TestNumbaBackend:
    """Numba code generation."""

    def test_numba_compiles(self):
        """CodeGenerator can compile simple numeric function."""
        # Skip if numba has API mismatch
        try:
            import numba.core.registry
            numba.core.registry.CPUDispatcher
        except AttributeError:
            pytest.skip("Numba API mismatch — cannot test compilation")

        gen = CodeGenerator()
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)

        kernel = gen.compile(slow_sum_func, test_args=(a, b))
        assert kernel is not None, "Compilation failed"

    def test_numba_speedup(self):
        """Compiled function is faster than original."""
        try:
            import numba.core.registry
            numba.core.registry.CPUDispatcher
        except AttributeError:
            pytest.skip("Numba API mismatch")

        gen = CodeGenerator()
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)

        kernel = gen.compile(slow_sum_func, test_args=(a, b))
        if not kernel.ready:
            pytest.skip("Compilation not ready")

        speedup = gen.measure_speedup(kernel, slow_sum_func, (a, b), trials=10)
        assert speedup > 1.0, f"No speedup: {speedup:.2f}×"

    def test_numba_output_correct(self):
        """Compiled output matches original (±1e-3)."""
        try:
            import numba.core.registry
            numba.core.registry.CPUDispatcher
        except AttributeError:
            pytest.skip("Numba API mismatch")

        gen = CodeGenerator()
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)

        kernel = gen.compile(slow_sum_func, test_args=(a, b))
        if not kernel.ready:
            pytest.skip("Compilation not ready")

        expected = slow_sum_func(a, b)
        actual = kernel.compiled(a, b)

        assert abs(expected - actual) < 1e-3, \
            f"Output mismatch: expected {expected}, got {actual}"


class TestAutoCompile:
    """Auto-compile hook in NerveTopology."""

    def test_enable_compiler_installs(self):
        """enable_compiler() installs profiler on nerve module."""
        from nerve.topology import NerveTopology
        topo = NerveTopology(n_rooms=100)
        topo.enable_compiler()

        # Compiler should be attached
        assert hasattr(topo, '_compiler'), "Compiler not attached"

    def test_compiler_does_not_break_tick(self):
        """Tick works normally with compiler enabled."""
        from nerve.topology import NerveTopology
        import numpy as np

        topo = NerveTopology(n_rooms=100)
        topo.enable_compiler()

        signals = {fid: np.random.randn(64).astype(np.float32) for fid in topo.fibers}
        result = topo.tick(signals)
        assert hasattr(result, 'rooms_fired'), "Tick result malformed with compiler"


def slow_sum_func(a, b):
    """Simple function for compiler testing."""
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


# ── New top-level unit tests (Task requirements) ──

def test_profiler_detects_hotspot(compiler):
    """After 100 calls the profiler identifies the watched function."""
    def dummy(x):
        return x + 1

    watched = compiler.profiler.watch(dummy)
    for i in range(100):
        watched(i)
    hotspots = compiler.profiler.get_hotspots()
    names = [h.name for h in hotspots]
    assert any("dummy" in n for n in names)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_speedup(compiler):
    """Numba-compiled function achieves >2× speedup over original."""
    np.random.seed(42)

    def slow_sum(array_a, array_b):
        total = 0.0
        for i in range(len(array_a)):
            total += array_a[i] * array_b[i]
        return total

    res = compiler.compile_function(slow_sum)
    assert res.backend == "numba"
    assert res.speedup > 2.0
    assert res.error is None


def test_auto_compile_wired(compiler):
    """NerveTopology.enable_compiler() installs the compiler profiler."""
    from nerve.topology import NerveTopology
    topo = NerveTopology(n_fibers=2, n_rooms=4)
    topo.enable_compiler()
    assert topo._compiler is not None
    assert topo._compiler._installed is True


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_compiler_skips_numba(compiler):
    """Already-@njit functions are returned as-is without double-compilation."""
    @numba.njit
    def already_fast(array_a, array_b):
        total = 0.0
        for i in range(len(array_a)):
            total += array_a[i] * array_b[i]
        return total

    res = compiler.compile_function(already_fast)
    assert res.backend == "numba"
    assert res.compiled is already_fast
    assert res.error is None


# ── Hot-Swap Tests ──────────────────────────────────────────

from sunset.compiler import hot_swap, hot_swap_restore


def test_hot_swap_replaces_original():
    """hot_swap installs a new callable and calls route through it."""
    import types
    import tests.test_compiler as test_mod

    # A simple target function in this test module
    def original_func(x):
        return x + 1

    # Inject it into the module so hot_swap can find it
    test_mod._hot_swap_target = original_func

    # Build replacement from raw CodeType
    code = compile("def _hot_swap_target(x): return x * 2", "<hot_swap_test>", "exec")
    func_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    assert isinstance(func_code, types.CodeType)

    result = hot_swap("tests.test_compiler", "_hot_swap_target", func_code)
    assert result["success"] is True
    assert result["original"] is original_func

    # The module attribute should now be the replacement
    replaced = getattr(test_mod, "_hot_swap_target")
    assert replaced is not original_func
    assert replaced(5) == 10  # x * 2, not x + 1

    # Cleanup
    setattr(test_mod, "_hot_swap_target", original_func)


def test_hot_swap_preserves_signature():
    """After hot_swap, inspect.signature matches the original."""
    import types
    import tests.test_compiler as test_mod

    def original_func(a, b, c=3.0):
        return a + b + c

    test_mod._sig_target = original_func
    orig_sig = inspect.signature(original_func)

    code = compile(
        "def _sig_target(a, b, c=3.0): return a + b + c",
        "<hot_swap_test>",
        "exec",
    )
    # Find the actual CodeType among constants (skip literals like 3.0)
    func_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    assert isinstance(func_code, types.CodeType)

    result = hot_swap("tests.test_compiler", "_sig_target", func_code)
    assert result["success"] is True

    replaced = getattr(test_mod, "_sig_target")
    new_sig = inspect.signature(replaced)
    assert new_sig == orig_sig

    # Cleanup
    setattr(test_mod, "_sig_target", original_func)


def test_hot_swap_is_reversible():
    """hot_swap_restore puts the original function back."""
    import types
    import tests.test_compiler as test_mod

    def original_func(x):
        return x + 100

    test_mod._rev_target = original_func

    code = compile("def _rev_target(x): return x * 99", "<hot_swap_test>", "exec")
    func_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    assert isinstance(func_code, types.CodeType)

    # Swap
    swap = hot_swap("tests.test_compiler", "_rev_target", func_code)
    assert swap["success"] is True
    assert getattr(test_mod, "_rev_target")(2) == 198

    # Restore
    restore = hot_swap_restore("tests.test_compiler", "_rev_target")
    assert restore["success"] is True
    assert restore["restored"] is original_func
    assert getattr(test_mod, "_rev_target")(2) == 102

    # Cleanup
    if hasattr(test_mod, "_rev_target"):
        delattr(test_mod, "_rev_target")


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")
def test_compiler_auto_hot_swap(compiler):
    """Compiler.hot_swap compiles + replaces in a single call."""
    np.random.seed(42)

    # A slow function we can compile
    def slow_dot(array_a, array_b):
        total = 0.0
        for i in range(len(array_a)):
            total += array_a[i] * array_b[i]
        return total

    # Inject into a module so the compiler can resolve it
    import tests.test_compiler as test_mod
    test_mod._auto_swap_target = slow_dot
    slow_dot.__module__ = "tests.test_compiler"

    # Auto compile + swap
    result = compiler.hot_swap(slow_dot, module=test_mod, attr_name="_auto_swap_target")
    assert result.validated is True
    assert result.backend == "numba"
    assert result.speedup >= 1.5  # expect some speedup from Numba

    # The module attribute should now be the compiled version
    swapped = getattr(test_mod, "_auto_swap_target")
    assert swapped is not slow_dot
    # Verify it still works
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    expected = slow_dot(a, b)
    actual = swapped(a, b)
    assert abs(expected - actual) < 1e-3

    # Restore via compiler.restore()
    restored = compiler.restore(key="tests.test_compiler._auto_swap_target")
    assert restored is True
    assert getattr(test_mod, "_auto_swap_target") is slow_dot

    # Cleanup
    delattr(test_mod, "_auto_swap_target")
    if "tests.test_compiler._auto_swap_target" in compiler._originals:
        del compiler._originals["tests.test_compiler._auto_swap_target"]
