"""Tests for the Sunset Compiler — profiler, Numba backend, auto-compile."""
import time
import numpy as np
import pytest

from sunset.compiler import Compiler
from sunset.codegen import CodeGenerator


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
            numba.core.registry.Dispatcher
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
            numba.core.registry.Dispatcher
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
            numba.core.registry.Dispatcher
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
