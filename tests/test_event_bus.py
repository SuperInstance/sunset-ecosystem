"""Tests for event_bus.py — Fleet pub/sub event bus.

Run: python3 -m pytest tests/test_event_bus.py -v --tb=short
"""
from __future__ import annotations

import pytest

from nexus.event_bus import EventBus, Event


class TestEventBusBasics:
    def test_create(self):
        bus = EventBus()
        assert bus.subscriber_count() == 0

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.*", lambda e: received.append(e.payload))
        count = bus.publish("test.event", {"data": 42})
        assert count == 1
        assert len(received) == 1
        assert received[0]["data"] == 42

    def test_wildcard_match(self):
        bus = EventBus()
        received = []
        bus.subscribe("breeding.*", lambda e: received.append(e.topic))
        bus.publish("breeding.spawn", {})
        bus.publish("breeding.mutate", {})
        bus.publish("mesh.update", {})
        assert len(received) == 2

    def test_multiple_subscribers(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe("topic", lambda e: a.append(1))
        bus.subscribe("topic", lambda e: b.append(1))
        bus.publish("topic", {})
        assert len(a) == 1
        assert len(b) == 1

    def test_no_subscribers(self):
        bus = EventBus()
        count = bus.publish("topic", {})
        assert count == 0

    def test_unsubscribe(self):
        bus = EventBus()
        sub = bus.subscribe("topic", lambda e: None)
        assert bus.subscriber_count() == 1
        bus.unsubscribe(sub)
        assert bus.subscriber_count() == 0

    def test_unsubscribe_invalid(self):
        bus = EventBus()
        sub = bus.subscribe("topic", lambda e: None)
        bus.unsubscribe(sub)
        assert bus.unsubscribe(sub) is False

    def test_safe_publish_no_crash(self):
        bus = EventBus()
        bus.subscribe("topic", lambda e: (_ for _ in ()).throw(ValueError("boom")))
        count = bus.publish_safe("topic", {})
        # Subscriber was called but errored, so count is 0 (not counted as successful)
        assert count == 0

    def test_event_timestamp(self):
        bus = EventBus()
        event = None
        def capture(e):
            nonlocal event
            event = e
        bus.subscribe("topic", capture)
        bus.publish("topic", {"x": 1})
        assert event is not None
        assert event.topic == "topic"
        assert event.timestamp > 0

    def test_metrics(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.publish("a", {})
        m = bus.metrics()
        assert m["subscribers"] == 2
        assert m["published"] == 1

    def test_health(self):
        bus = EventBus()
        bus.subscribe("topic", lambda e: None, name="sub1")
        bus.publish("topic", {})
        h = bus.health()
        assert "sub1" in h
        assert h["sub1"]["count"] == 1

    def test_reset(self):
        bus = EventBus()
        bus.subscribe("topic", lambda e: None)
        bus.publish("topic", {})
        bus.reset()
        assert bus.subscriber_count() == 0
        assert bus.metrics()["published"] == 0

    def test_topics(self):
        bus = EventBus()
        bus.subscribe("a.*", lambda e: None)
        bus.subscribe("b.*", lambda e: None)
        assert bus.topics() == {"a.*", "b.*"}

    def test_repr(self):
        bus = EventBus()
        assert "EventBus" in repr(bus)
