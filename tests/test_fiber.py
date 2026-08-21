"""Tests for NerveFiber — sensory pathway lifecycle (PERCEIVING → ADAPTING → COMPILED → NOVELTY_ALERT).

Covers FiberState, SensoryTile, NerveFiber init, state transitions,
feature extraction, hashing, and statistics.
"""

import numpy as np
import pytest

from nerve.fiber import (
    ECOSYSTEM,
    FiberState,
    NerveFiber,
    SensoryTile,
    _get_device_router,
)


# ---------------------------------------------------------------------------
# Ecosystem
# ---------------------------------------------------------------------------


class TestEcosystem:
    def test_keys_present(self):
        assert "eisenstein_embed" in ECOSYSTEM
        assert "device_router" in ECOSYSTEM
        assert "tensor_spline" in ECOSYSTEM
        assert "triplet_miner" in ECOSYSTEM


# ---------------------------------------------------------------------------
# FiberState
# ---------------------------------------------------------------------------


class TestFiberState:
    def test_values(self):
        assert FiberState.PERCEIVING.value == "perceiving"
        assert FiberState.ADAPTING.value == "adapting"
        assert FiberState.COMPILED.value == "compiled"
        assert FiberState.NOVELTY_ALERT.value == "novelty"


# ---------------------------------------------------------------------------
# SensoryTile
# ---------------------------------------------------------------------------


class TestSensoryTile:
    def test_defaults(self):
        t = SensoryTile(pattern_id="abc123")
        assert t.confidence == 0.0
        assert t.source_fiber == ""
        assert t.state == FiberState.PERCEIVING
        assert t.timestamp > 0

    def test_repr(self):
        t = SensoryTile(
            pattern_id="abc123456789", confidence=0.75, state=FiberState.COMPILED
        )
        r = repr(t)
        assert "abc12345" in r
        assert "conf=0.75" in r
        assert "compiled" in r


# ---------------------------------------------------------------------------
# NerveFiber init
# ---------------------------------------------------------------------------


class TestNerveFiberInit:
    def test_defaults(self):
        f = NerveFiber("f1")
        assert f.fiber_id == "f1"
        assert f.model_type == "generic"
        assert f.state == FiberState.PERCEIVING
        assert f.confidence == 0.0

    def test_custom_params(self):
        f = NerveFiber(
            "f2",
            model_type="jepa",
            adapt_threshold=0.8,
            novelty_threshold=0.5,
            epsilon=0.1,
        )
        assert f.model_type == "jepa"
        assert f.adapt_threshold == 0.8
        assert f.novelty_threshold == 0.5
        assert f.epsilon == 0.1

    def test_repr(self):
        f = NerveFiber("f1")
        r = repr(f)
        assert "f1" in r
        assert "perceiving" in r


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashSignal:
    def test_numpy_fast_path(self):
        arr = np.array([1.0, 2.0, 3.0])
        h1 = NerveFiber._hash_signal(arr)
        h2 = NerveFiber._hash_signal(arr)
        assert isinstance(h1, str)
        assert h1 == h2  # deterministic

    def test_string_fallback(self):
        h1 = NerveFiber._hash_signal("hello")
        h2 = NerveFiber._hash_signal("hello")
        assert isinstance(h1, str)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_fast_hash(self):
        arr = np.array([1.0, 2.0])
        h = NerveFiber._hash_signal_fast(arr)
        assert isinstance(h, str)

    def test_fingerprint(self):
        fp = NerveFiber._fingerprint_signal("test")
        assert isinstance(fp, int)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        f = NerveFiber("f1")
        feats = f._extract_features(arr)
        assert "shape" in feats
        assert "mean" in feats
        assert "std" in feats
        assert "nonzero" in feats

    def test_string(self):
        f = NerveFiber("f1")
        feats = f._extract_features("hello world 123")
        assert feats["length"] == 15
        assert feats["type"] == "str"
        assert feats["contains_digits"] is True
        assert feats["contains_alpha"] is True

    def test_cache(self):
        arr = np.array([5.0, 6.0])
        f = NerveFiber("f1")
        feats1 = f._extract_features(arr)
        feats2 = f._extract_features(arr)
        assert feats1 == feats2
        # cache exists
        assert hasattr(f, "_feature_cache")


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_perceiving_to_adapting(self):
        f = NerveFiber("f1", epsilon=0.1)
        tile = f.perceive("signal1")
        # First call: PERCEIVING branch, confidence -> 0.1, state -> ADAPTING
        assert tile.state == FiberState.ADAPTING
        assert f.state == FiberState.ADAPTING
        assert f.confidence == pytest.approx(0.1)

    def test_adapting_to_compiled(self):
        f = NerveFiber("f1", adapt_threshold=0.15, epsilon=0.1)
        f.perceive("signal1")  # 0 -> 0.1, ADAPTING
        tile = f.perceive("signal1")  # 0.1 -> 0.2, crosses threshold -> COMPILED
        assert f.confidence >= 0.15
        assert f.state == FiberState.COMPILED
        assert tile.state == FiberState.COMPILED

    def test_compiled_automatic(self):
        f = NerveFiber("f1", adapt_threshold=0.05, epsilon=1.0)
        f.perceive("signal1")  # PERCEIVING -> ADAPTING
        f.perceive("signal1")  # ADAPTING -> COMPILED
        assert f.state == FiberState.COMPILED
        tile = f.perceive("signal1")
        assert tile.state == FiberState.COMPILED
        assert tile.confidence == 1.0

    def test_novelty_alert(self):
        f = NerveFiber("f1", adapt_threshold=0.05, epsilon=1.0, novelty_threshold=0.3)
        f.perceive("signal1")  # ADAPTING
        f.perceive("signal1")  # COMPILED
        assert f.state == FiberState.COMPILED
        # new signal triggers novelty
        tile = f.perceive("signal2")
        # COMPILED + new signal -> NOVELTY_ALERT, then PERCEIVING/NOVELTY branch -> ADAPTING
        assert tile.state == FiberState.ADAPTING
        assert f.state == FiberState.ADAPTING

    def test_novelty_to_perceiving(self):
        f = NerveFiber("f1", adapt_threshold=0.05, epsilon=1.0)
        f.perceive("signal1")  # ADAPTING
        f.perceive("signal1")  # COMPILED
        f.perceive("signal2")  # novelty -> ADAPTING
        assert f.state == FiberState.ADAPTING
        # Next call continues from ADAPTING -> COMPILED again
        f.perceive("signal2")
        assert f.state == FiberState.COMPILED

    def test_total_signals(self):
        f = NerveFiber("f1")
        f.perceive("a")
        f.perceive("b")
        f.perceive("c")
        assert f.stats["total_signals"] == 3

    def test_compiled_signals(self):
        f = NerveFiber("f1", adapt_threshold=0.05, epsilon=1.0)
        f.perceive("a")  # ADAPTING
        f.perceive("a")  # COMPILED (counts as 1 compiled signal)
        f.perceive("a")  # COMPILED automatic (counts as 1 compiled signal)
        assert f.stats["compiled_signals"] == 2
        assert f.stats["compile_rate"] == pytest.approx(2 / 3)

    def test_compile_rate_zero(self):
        f = NerveFiber("f1")
        assert f.stats["compile_rate"] == 0.0

    def test_reset(self):
        f = NerveFiber("f1", adapt_threshold=0.05, epsilon=1.0)
        f.perceive("a")  # ADAPTING
        f.perceive("a")  # COMPILED
        assert f.state == FiberState.COMPILED
        f.reset()
        assert f.state == FiberState.PERCEIVING
        assert f.confidence == 0.0
        assert f.stats["compiled_patterns"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_perceive_concurrent(self):
        import threading

        f = NerveFiber("f1", epsilon=0.01)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    f.perceive("sig")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert f.stats["total_signals"] == 200


# ---------------------------------------------------------------------------
# _get_device_router
# ---------------------------------------------------------------------------


class TestDeviceRouter:
    def test_returns_none_when_unavailable(self):
        # In test environment, device_router is likely not installed
        router = _get_device_router()
        # Should not raise
        assert router is None or hasattr(router, "detect")
