"""Tests for RoomGridCompiler — auto-compile, hot-swap, A/B, restore.

Requirements (from task spec):
    1. compile returns speedup > 1.0
    2. A/B correctness
    3. hot-swap works
    4. restore reverts
"""
import sys
import numpy as np
import pytest

from nerve.room_grid import RoomGrid, forward_einsum, batch_novelty, _batch_novelty_numpy
from sunset.compiler_integration import RoomGridCompiler, _HAS_NUMBA


# ── Custom small grid fixture (numba wins for n≤10) ──

@pytest.fixture
def room_grid_10():
    np.random.seed(42)
    return RoomGrid(10)


class TestCompilerBasics:
    """Instantiation and profiling."""

    def test_compiler_instantiates(self, room_grid_100):
        compiler = RoomGridCompiler(room_grid_100)
        assert compiler.grid is room_grid_100
        assert compiler._has_numba == _HAS_NUMBA

    def test_profile_tick_returns_phases(self, room_grid_100):
        compiler = RoomGridCompiler(room_grid_100)
        profile = compiler.profile_tick(ticks=20)
        assert "forward" in profile
        assert "novelty" in profile
        assert "routing" in profile
        assert all(v >= 0 for v in profile.values())


class TestCompileEinsum:
    """forward_einsum → Numba compilation + hot-swap.

    Uses a *small* grid (n=10) because numpy einsum overhead dominates
    at small sizes, giving the numba serial kernel a clear win.
    """

    def test_compile_einsum_speedup_and_correctness(self, room_grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_10)
        result = compiler.compile_einsum(ab_trials=50)
        assert result.validated, f"A/B correctness failed: {result.error}"
        assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"

    def test_compile_einsum_hot_swap_active(self, room_grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_10)
        compiler.compile_einsum(ab_trials=50)
        current = sys.modules["nerve.room_grid"].forward_einsum
        assert hasattr(current, "_sunset_original"), "Hot-swap did not install _sunset_original"

    def test_compile_einsum_restore_reverts(self, room_grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_10)
        compiler.compile_einsum(ab_trials=50)
        original = sys.modules["nerve.room_grid"].forward_einsum._sunset_original
        restored_count = compiler.restore_original()
        assert restored_count > 0, "restore_original returned 0"
        current = sys.modules["nerve.room_grid"].forward_einsum
        assert current is original, "After restore, forward_einsum is not the original"

    def test_compile_einsum_outputs_match_after_swap(self, room_grid_10):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        compiler = RoomGridCompiler(room_grid_10)
        expected = forward_einsum(room_grid_10.w, x)
        compiler.compile_einsum(ab_trials=50)
        actual = forward_einsum(room_grid_10.w, x)
        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-3), "Outputs diverged after hot-swap"
        compiler.restore_original()


class TestCompileNovelty:
    """batch_novelty → Numba compilation + hot-swap."""

    def test_compile_novelty_speedup_and_correctness(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        result = compiler.compile_novelty_scoring(ab_trials=50)
        assert result.validated, f"A/B correctness failed: {result.error}"
        assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"

    def test_compile_novelty_hot_swap_active(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.compile_novelty_scoring(ab_trials=50)
        current = sys.modules["nerve.room_grid"].batch_novelty
        assert hasattr(current, "_sunset_original"), "Hot-swap did not install _sunset_original"

    def test_compile_novelty_restore_reverts(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.compile_novelty_scoring(ab_trials=50)
        original = sys.modules["nerve.room_grid"].batch_novelty._sunset_original
        compiler.restore_original()
        current = sys.modules["nerve.room_grid"].batch_novelty
        assert current is original, "After restore, batch_novelty is not the original"

    def test_compile_novelty_outputs_match_after_swap(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        np.random.seed(42)
        n = room_grid_1000.n
        latents = np.random.randn(n, 16).astype(np.float32)
        hist = room_grid_1000._hist
        compiler = RoomGridCompiler(room_grid_1000)
        expected = batch_novelty(latents, hist, room_grid_1000._hist_count,
                                 room_grid_1000._hist_idx, room_grid_1000._hist_max)
        compiler.compile_novelty_scoring(ab_trials=50)
        actual = batch_novelty(latents, hist, room_grid_1000._hist_count,
                               room_grid_1000._hist_idx, room_grid_1000._hist_max)
        assert np.allclose(expected, actual, atol=1e-4, rtol=1e-3), "Novelty diverged after hot-swap"
        compiler.restore_original()


class TestCompileRouting:
    """Routing kernel compilation."""

    def test_compile_routing_speedup_and_correctness(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        result = compiler.compile_routing(ab_trials=50)
        assert result.validated, f"A/B correctness failed: {result.error}"
        assert result.speedup > 1.0, f"Expected speedup > 1.0, got {result.speedup}"

    def test_compile_routing_hot_swap_active(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.compile_routing(ab_trials=50)
        current = getattr(sys.modules["nerve.room_grid"], "_tick_routing_compiled", None)
        assert current is not None, "_tick_routing_compiled not installed"

    def test_compile_routing_restore_reverts(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.compile_routing(ab_trials=50)
        compiler.restore_original()
        current = getattr(sys.modules["nerve.room_grid"], "_tick_routing_compiled", None)
        assert current is None, "_tick_routing_compiled should be removed after restore"

    def test_compile_routing_tick_still_works(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.compile_routing(ab_trials=50)
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        for _ in range(10):
            out = room_grid_1000.tick(x)
            assert "fired" in out
            assert "ids" in out
        compiler.restore_original()


class TestAutoCompile:
    """End-to-end auto-compile flow."""

    def test_auto_compile_runs(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        result = compiler.auto_compile(ticks=50, ab_trials=30)
        assert result.validated
        assert result.speedup > 1.0
        assert result.target in ("einsum", "novelty", "routing")

    def test_auto_compile_tick_still_works(self, room_grid_1000):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        compiler = RoomGridCompiler(room_grid_1000)
        compiler.auto_compile(ticks=50, ab_trials=30)
        np.random.seed(42)
        x = np.random.randn(64).astype(np.float32)
        for _ in range(10):
            out = room_grid_1000.tick(x)
            assert "fired" in out
            assert "ids" in out
        compiler.restore_original()


class TestRoomGridInitIntegration:
    """Optional ``compiler`` param in RoomGrid.__init__."""

    def test_init_with_compiler_auto(self):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        np.random.seed(42)
        grid = RoomGrid(200, compiler="auto")
        assert grid._compiler is not None
        assert isinstance(grid._compiler, RoomGridCompiler)

    def test_init_with_compiler_instance(self):
        if not _HAS_NUMBA:
            pytest.skip("Numba not available")
        np.random.seed(42)
        grid = RoomGrid(200)
        compiler = RoomGridCompiler(grid)
        grid2 = RoomGrid(200, compiler=compiler)
        assert grid2._compiler is compiler

    def test_init_compiler_invalid_raises(self):
        with pytest.raises(TypeError):
            RoomGrid(100, compiler=12345)

    def test_init_without_compiler(self):
        grid = RoomGrid(100)
        assert grid._compiler is None


# ── Global cleanup: ensure we never leave hot-swaps behind ──

def pytest_sessionfinish(session, exitstatus):
    """Restore any lingering hot-swaps so other test files are not tainted."""
    mod = sys.modules.get("nerve.room_grid")
    if mod is None:
        return
    for attr in ("forward_einsum", "batch_novelty", "_tick_routing_compiled"):
        obj = getattr(mod, attr, None)
        if obj is not None and hasattr(obj, "_sunset_original"):
            setattr(mod, attr, obj._sunset_original)
        if attr == "_tick_routing_compiled" and hasattr(mod, attr):
            delattr(mod, attr)
