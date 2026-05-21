"""Tests for the nerve fiber architecture."""

import pytest

from nerve.fiber import NerveFiber, FiberState, SensoryTile, ECOSYSTEM
from nerve.routing import RoutingLayer, Route, HebbianChannel
from nerve.adaptation import AdaptationEngine, ShoeTracker, ShoeState


# ── Ecosystem Integration Tests ─────────────────────────────


class TestEcosystemDetection:
    """Tests for optional ecosystem package detection."""

    def test_ecosystem_dict_exists(self):
        assert isinstance(ECOSYSTEM, dict)
        assert "eisenstein_embed" in ECOSYSTEM
        assert "device_router" in ECOSYSTEM
        assert "tensor_spline" in ECOSYSTEM
        assert "triplet_miner" in ECOSYSTEM

    def test_ecosystem_values_are_bool(self):
        for key, val in ECOSYSTEM.items():
            assert isinstance(val, bool), f"{key} should be bool, got {type(val)}"


@pytest.mark.skipif(
    not ECOSYSTEM.get("eisenstein_embed"),
    reason="eisenstein-embed not installed",
)
class TestEisensteinIntegration:
    """Tests using real eisenstein-embed bitvectors."""

    def test_fingerprint_signal_returns_int(self):
        f = NerveFiber("eis-1")
        fp = NerveFiber._fingerprint_signal("hello world")
        assert isinstance(fp, int)
        assert fp > 0

    def test_fingerprint_deterministic(self):
        fp1 = NerveFiber._fingerprint_signal("test signal")
        fp2 = NerveFiber._fingerprint_signal("test signal")
        assert fp1 == fp2

    def test_fingerprint_different_signals_differ(self):
        fp1 = NerveFiber._fingerprint_signal("signal alpha")
        fp2 = NerveFiber._fingerprint_signal("signal beta")
        assert fp1 != fp2

    def test_features_include_bitvector(self):
        f = NerveFiber("eis-2")
        tile = f.perceive("test bitvector")
        assert "bitvector_fingerprint" in tile.features
        assert "bitvector_hex" in tile.features
        assert isinstance(tile.features["bitvector_fingerprint"], int)

    def test_stem_similarity(self):
        """Related forms (deploy/deployment) should have similar fingerprints."""
        from eisenstein_embed.bitvector import (
            text_fingerprint,
            bitvector_similarity,
        )
        fp1 = text_fingerprint("deploy", use_stemming=True)
        fp2 = text_fingerprint("deployment", use_stemming=True)
        sim = bitvector_similarity(fp1, fp2)
        assert sim > 0.5, f"Expected high similarity, got {sim}"


@pytest.mark.skipif(
    not ECOSYSTEM.get("device_router"),
    reason="device-router not installed",
)
class TestDeviceRouterIntegration:
    """Tests using real device-router."""

    def test_device_detection_in_features(self):
        f = NerveFiber("dr-1")
        tile = f.perceive("test signal")
        assert "device_cuda" in tile.features
        assert "device_igpu" in tile.features

    def test_router_singleton(self):
        from nerve.fiber import _get_device_router
        r1 = _get_device_router()
        r2 = _get_device_router()
        assert r1 is r2


@pytest.mark.skipif(
    not ECOSYSTEM.get("triplet_miner"),
    reason="triplet-miner not installed",
)
class TestTripletMinerIntegration:
    """Tests using real triplet-miner."""

    def test_triplet_miner_import(self):
        from triplet_miner.git_miner import TripletMiner
        miner = TripletMiner()
        assert "TripletMiner" in repr(miner)


@pytest.mark.skipif(
    not ECOSYSTEM.get("tensor_spline"),
    reason="tensor-spline not installed",
)
class TestTensorSplineIntegration:
    """Tests using real tensor-spline."""

    def test_spline_linear_import(self):
        from tensor_spline.spline import SplineLinear
        # Just verify the import works
        assert SplineLinear is not None


# ── Fiber Tests ──────────────────────────────────────────────


class TestNerveFiber:
    def test_initial_state(self):
        f = NerveFiber("test-1")
        assert f.state == FiberState.PERCEIVING
        assert f.confidence == 0.0
        assert f.fiber_id == "test-1"

    def test_perceive_starts_perceiving(self):
        f = NerveFiber("test-2")
        tile = f.perceive("hello world")
        assert isinstance(tile, SensoryTile)
        assert tile.source_fiber == "test-2"
        assert tile.confidence > 0

    def test_adaptation_accumulates(self):
        f = NerveFiber("test-3", epsilon=0.2)
        # Repeated signals build confidence
        for i in range(5):
            f.perceive("repeated signal")
        assert f.confidence > 0.5
        assert f.state in (FiberState.ADAPTING, FiberState.COMPILED)

    def test_compilation_at_threshold(self):
        f = NerveFiber("test-4", epsilon=0.15, adapt_threshold=0.8)
        # Keep sending until compiled
        for i in range(20):
            f.perceive("compile me")
        assert f.state == FiberState.COMPILED
        assert f.confidence >= 0.8

    def test_compiled_automatic_processing(self):
        f = NerveFiber("test-5", epsilon=0.2, adapt_threshold=0.7)
        # Compile a pattern
        for i in range(20):
            f.perceive("auto pattern")
        assert f.state == FiberState.COMPILED
        # Now process should be automatic
        tile = f.perceive("auto pattern")
        assert tile.state == FiberState.COMPILED
        assert tile.confidence == 1.0

    def test_novelty_detection(self):
        f = NerveFiber("test-6", epsilon=0.2, adapt_threshold=0.7)
        # Compile one pattern
        for i in range(20):
            f.perceive("old pattern")
        assert f.state == FiberState.COMPILED
        # Send a NOVEL pattern
        tile = f.perceive("something completely different!")
        # Should detect novelty (pattern_id won't match compiled)
        assert tile.pattern_id != NerveFiber._hash_signal("old pattern")

    def test_reset(self):
        f = NerveFiber("test-7", epsilon=0.2, adapt_threshold=0.5)
        for i in range(10):
            f.perceive("reset me")
        f.reset()
        assert f.state == FiberState.PERCEIVING
        assert f.confidence == 0.0

    def test_stats(self):
        f = NerveFiber("test-8", model_type="jepa")
        f.perceive("signal")
        stats = f.stats
        assert stats["fiber_id"] == "test-8"
        assert stats["model_type"] == "jepa"
        assert stats["total_signals"] == 1

    def test_repr(self):
        f = NerveFiber("test-9")
        r = repr(f)
        assert "test-9" in r
        assert "perceiving" in r

    def test_deterministic_hash(self):
        h1 = NerveFiber._hash_signal("test")
        h2 = NerveFiber._hash_signal("test")
        assert h1 == h2
        h3 = NerveFiber._hash_signal("different")
        assert h1 != h3

    def test_feature_extraction(self):
        f = NerveFiber("test-10")
        features = f._extract_features("hello123")
        assert features["length"] == 8
        assert features["contains_digits"] is True
        assert features["contains_alpha"] is True


# ── Routing Tests ────────────────────────────────────────────


class TestRoute:
    def test_fire_strong_route(self):
        r = Route("src", "dst", strength=0.99)
        fires = sum(1 for _ in range(100) if r.fire(chaos=0.0))
        assert fires > 90  # Almost always fires

    def test_fire_weak_route(self):
        r = Route("src", "dst", strength=0.01)
        fires = sum(1 for _ in range(100) if r.fire(chaos=0.0))
        assert fires < 10  # Almost never fires

    def test_chaos_fires_weak_route(self):
        r = Route("src", "dst", strength=0.01)
        fires = sum(1 for _ in range(100) if r.fire(chaos=0.5))
        assert fires > 10  # Chaos helps fire

    def test_reinforce_success(self):
        r = Route("src", "dst", strength=0.5)
        for _ in range(10):
            r.reinforce(True)
        assert r.strength > 0.5
        assert r.reception > 0.5

    def test_reinforce_failure(self):
        r = Route("src", "dst", strength=0.5)
        for _ in range(10):
            r.fire(chaos=0.0)
            r.reinforce(False)
        assert r.strength < 0.5

    def test_decay(self):
        r = Route("src", "dst", strength=0.9)
        r.decay(0.9)
        assert r.strength < 0.9

    def test_repr(self):
        r = Route("src", "dst")
        assert "src→dst" in repr(r)


class TestHebbianChannel:
    def test_activate_strengthens(self):
        ch = HebbianChannel("a", "b")
        initial = ch.weight
        ch.activate()
        assert ch.weight > initial

    def test_decay_weakens(self):
        ch = HebbianChannel("a", "b", initial_weight=0.5)
        ch.decay(0.9)
        assert ch.weight < 0.5

    def test_repr(self):
        ch = HebbianChannel("a", "b")
        assert "a↔b" in repr(ch)


class TestRoutingLayer:
    def test_add_route(self):
        rl = RoutingLayer()
        r = rl.add_route("fiber-1", "agent-1")
        assert r.source == "fiber-1"

    def test_fire_routes(self):
        rl = RoutingLayer(chaos=0.0)
        rl.add_route("fiber-1", "agent-1", strength=0.99)
        rl.add_route("fiber-1", "agent-2", strength=0.01)
        fired = rl.fire("fiber-1")
        assert "agent-1" in fired

    def test_feedback(self):
        rl = RoutingLayer()
        rl.add_route("fiber-1", "agent-1", strength=0.5)
        rl.feedback("fiber-1", "agent-1", True)
        routes = rl.get_strongest_routes("fiber-1")
        assert routes[0].strength > 0.5

    def test_strongest_routes(self):
        rl = RoutingLayer()
        rl.add_route("f", "a", strength=0.9)
        rl.add_route("f", "b", strength=0.5)
        rl.add_route("f", "c", strength=0.7)
        top = rl.get_strongest_routes("f", top_k=2)
        assert len(top) == 2
        assert top[0].strength >= top[1].strength

    def test_hebbian_channel(self):
        rl = RoutingLayer()
        rl.add_channel("a", "b")
        w = rl.get_channel_weight("a", "b")
        assert w > 0
        w2 = rl.get_channel_weight("x", "y")
        assert w2 == 0.0

    def test_decay_all(self):
        rl = RoutingLayer()
        rl.add_route("f", "a", strength=0.9)
        rl.add_channel("a", "b", weight=0.5)
        rl.decay_all()
        assert rl.get_strongest_routes("f")[0].strength < 0.9


# ── Adaptation Tests ─────────────────────────────────────────


class TestShoeTracker:
    def test_step_creates_shoe(self):
        st = ShoeTracker()
        shoe = st.step("pattern-1", FiberState.PERCEIVING)
        assert shoe.steps == 1
        assert shoe.notice_level == 1.0

    def test_adaptation_increases(self):
        st = ShoeTracker()
        # Perceiving → high notice
        st.step("p1", FiberState.PERCEIVING)
        # Adapting → notice drops
        for _ in range(5):
            st.step("p1", FiberState.ADAPTING)
        # Compiled → notice drops fast
        for _ in range(5):
            st.step("p1", FiberState.COMPILED)
        assert st.adaptation_score > 0.5

    def test_novelty_resets_notice(self):
        st = ShoeTracker()
        for _ in range(10):
            st.step("p1", FiberState.COMPILED)
        # Novelty alert!
        st.step("p1", FiberState.NOVELTY_ALERT)
        shoe = st._shoes["p1"]
        assert shoe.notice_level > 0.0  # Back to noticing

    def test_compiled_count(self):
        st = ShoeTracker()
        st.step("p1", FiberState.COMPILED)
        st.step("p2", FiberState.PERCEIVING)
        st.step("p3", FiberState.COMPILED)
        assert st.compiled_count == 2
        assert st.total_patterns == 3


class TestAdaptationEngine:
    def test_register_and_process(self):
        engine = AdaptationEngine()
        fiber = NerveFiber("f1", epsilon=0.2)
        engine.register(fiber)
        result = engine.process_signal("f1", "hello")
        assert "tile" in result
        # First signal: confidence goes from 0 → 0.2, state moves to ADAPTING
        assert result["fiber_state"] == FiberState.ADAPTING

    def test_system_status(self):
        engine = AdaptationEngine()
        engine.register(NerveFiber("f1", epsilon=0.2))
        engine.register(NerveFiber("f2", epsilon=0.1))
        status = engine.system_status()
        assert status["total_fibers"] == 2
        assert status["perceiving"] == 2

    def test_adaptation_progresses(self):
        engine = AdaptationEngine()
        fiber = NerveFiber("f1", epsilon=0.2, adapt_threshold=0.5)
        engine.register(fiber)
        # Process many signals
        for _ in range(30):
            engine.process_signal("f1", "learn this")
        status = engine.system_status()
        # Should have progressed beyond perceiving
        assert status["compiled"] + status["adapting"] > 0

    def test_unknown_fiber(self):
        engine = AdaptationEngine()
        result = engine.process_signal("unknown", "signal")
        assert "error" in result
