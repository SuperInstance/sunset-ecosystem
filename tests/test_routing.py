"""Tests for RoutingLayer — firing, Hebbian activation, feedback."""
import numpy as np
import pytest

from nerve.routing import RoutingLayer, Route, HebbianChannel


class TestFireFast:
    """Vectorized route firing correctness."""

    def test_fire_fast_deterministic(self, routing_layer):
        """Same input → same output with seeded random."""
        np.random.seed(42)
        fired1 = routing_layer.fire_fast("fiber-0")
        np.random.seed(42)
        fired2 = routing_layer.fire_fast("fiber-0")
        assert fired1 == fired2, "fire_fast not deterministic with same seed"

    def test_strong_routes_always_fire(self, routing_layer):
        """Routes with strength > 0.9 always fire (compile threshold)."""
        routing_layer.add_route("fiber-0", "room-strong", strength=0.95)
        # Run many times — should always fire
        for _ in range(20):
            fired = routing_layer.fire_fast("fiber-0")
            assert "room-strong" in fired, "Strong route did not fire"

    def test_empty_source_returns_empty(self, routing_layer):
        """fire_fast on unknown source returns empty list."""
        fired = routing_layer.fire_fast("nonexistent-fiber")
        assert fired == [], f"Expected empty list, got {fired}"

    def test_dest_filter_works(self, routing_layer):
        """Filtering by destinations works."""
        routing_layer.add_route("fiber-0", "room-a", strength=1.0)
        routing_layer.add_route("fiber-0", "room-b", strength=1.0)
        fired = routing_layer.fire_fast("fiber-0", destinations=["room-a"])
        assert "room-a" in fired
        assert "room-b" not in fired


class TestHebbian:
    """Hebbian channel activation."""

    def test_co_firing_strengthens_channel(self, routing_layer):
        """Channels strengthen after rooms co-fire."""
        routing_layer.add_route("fiber-0", "room-1", strength=1.0)
        routing_layer.add_route("fiber-0", "room-2", strength=1.0)
        # Pre-create channel
        key = routing_layer._channel_key("room-1", "room-2")
        ch = HebbianChannel("room-1", "room-2")
        routing_layer._channels[key] = ch
        before = ch.weight

        routing_layer.fire_fast("fiber-0")
        after = ch.weight
        assert after > before, f"Channel did not strengthen: {before} -> {after}"

    def test_channel_exists_after_activation(self, routing_layer):
        """fire_fast only activates PRE-EXISTING channels; does not auto-create."""
        routing_layer.add_route("fiber-0", "room-x", strength=1.0)
        routing_layer.add_route("fiber-0", "room-y", strength=1.0)
        # Pre-create the channel
        key = routing_layer._channel_key("room-x", "room-y")
        routing_layer._channels[key] = HebbianChannel("room-x", "room-y")
        routing_layer.fire_fast("fiber-0")
        assert key in routing_layer._channels, "Channel lost after co-fire"
        assert routing_layer._channels[key].weight > 0.1, "Channel not activated"


class TestFeedback:
    """Route reinforcement."""

    def test_success_increases_strength(self, routing_layer):
        """Positive feedback increases route strength."""
        routing_layer.add_route("fiber-0", "room-test", strength=0.5)
        route = routing_layer._routes[routing_layer._route_key("fiber-0", "room-test")]
        before = route.strength
        routing_layer.feedback("fiber-0", "room-test", success=True)
        after = route.strength
        assert after > before, f"Strength did not increase: {before} -> {after}"

    def test_failure_decreases_strength(self, routing_layer):
        """Negative feedback decreases route strength."""
        routing_layer.add_route("fiber-0", "room-test", strength=0.5)
        route = routing_layer._routes[routing_layer._route_key("fiber-0", "room-test")]
        before = route.strength
        routing_layer.feedback("fiber-0", "room-test", success=False)
        after = route.strength
        assert after < before, f"Strength did not decrease: {before} -> {after}"

    def test_batch_feedback(self, routing_layer):
        """Batch feedback updates multiple routes."""
        for i in range(5):
            routing_layer.add_route(f"fiber-{i}", "room-test", strength=0.5)
        updates = [(f"fiber-{i}", "room-test", True) for i in range(5)]
        routing_layer.feedback_batch(updates)
        for i in range(5):
            route = routing_layer._routes[routing_layer._route_key(f"fiber-{i}", "room-test")]
            assert route.strength > 0.5, f"Route {i} not reinforced"


# ── New top-level unit tests (Task requirements) ──

def test_fire_fast_deterministic(routing_layer):
    """Same input and seed must produce the same output."""
    np.random.seed(42)
    room_ids = [f"src_{i}" for i in range(5)]
    out1 = routing_layer.fire_fast("test_source", room_ids, chaos=0.1)
    np.random.seed(42)
    out2 = routing_layer.fire_fast("test_source", room_ids, chaos=0.1)
    assert out1 == out2


def test_hebbian_activation(routing_layer):
    """Channels strengthen after co-firing."""
    np.random.seed(42)
    # Ensure both routes are strong enough to fire compiled (deterministic)
    routing_layer.routes["src_0→dst_0"].strength = 0.95
    routing_layer.add_route("src_0", "dst_1", strength=0.95)
    # Pre-create channel so fire_fast can activate it
    key = routing_layer._channel_key("dst_0", "dst_1")
    routing_layer.channels[key] = HebbianChannel("dst_0", "dst_1")
    routing_layer.fire_fast("src_0", ["dst_0", "dst_1"], chaos=0.0)
    pre = routing_layer.channels[key].weight
    routing_layer.fire_fast("src_0", ["dst_0", "dst_1"], chaos=0.0)
    post = routing_layer.channels[key].weight
    assert post > pre


def test_compile_threshold(routing_layer):
    """Routes with strength > 0.9 always fire when compiled."""
    np.random.seed(42)
    route = routing_layer.routes["src_0→dst_0"]
    route.strength = 0.95
    result = routing_layer.fire_fast("src_0", ["dst_0"], chaos=0.0)
    assert "dst_0" in result


def test_feedback_reinforcement(routing_layer):
    """Successful feedback increases route strength."""
    np.random.seed(42)
    pre = routing_layer.routes["src_0→dst_0"].strength
    routing_layer.feedback("src_0", "dst_0", success=True)
    post = routing_layer.routes["src_0→dst_0"].strength
    assert post > pre
