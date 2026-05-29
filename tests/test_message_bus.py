"""Tests for message_bus.py — Pub/sub message bus with wildcards.

Run: python3 -m pytest tests/test_message_bus.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.message_bus import MessageBus, Message


class TestMessageBus:
    def test_create(self):
        bus = MessageBus()
        assert bus.queue_size() == 0

    def test_publish_and_sync(self):
        bus = MessageBus()
        received = []
        bus.subscribe("room.alpha", lambda msg: received.append(msg.payload))
        bus.publish("room.alpha", {"event": "test"})
        bus.run_sync()
        assert len(received) == 1
        assert received[0]["event"] == "test"

    def test_wildcard_subscription(self):
        bus = MessageBus()
        received = []
        bus.subscribe("room.*.trap", lambda msg: received.append(msg.topic))
        bus.publish("room.alpha.trap", 1)
        bus.publish("room.beta.trap", 2)
        bus.publish("room.alpha.treasure", 3)
        bus.flush()
        assert len(received) == 2
        assert "room.alpha.trap" in received
        assert "room.beta.trap" in received

    def test_multiple_handlers(self):
        bus = MessageBus()
        a, b = [], []
        bus.subscribe("topic", lambda msg: a.append(1))
        bus.subscribe("topic", lambda msg: b.append(1))
        bus.publish("topic", "data")
        bus.flush()
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe(self):
        bus = MessageBus()
        received = []
        handler = lambda msg: received.append(msg.payload)
        bus.subscribe("topic", handler)
        bus.unsubscribe("topic", handler)
        bus.publish("topic", "data")
        bus.flush()
        assert len(received) == 0

    def test_peek(self):
        bus = MessageBus()
        bus.publish("a", 1)
        bus.publish("b", 2)
        msgs = bus.peek(2)
        assert len(msgs) == 2
        assert msgs[0].topic == "a"
        assert bus.queue_size() == 2

    def test_queue_drop_oldest(self):
        bus = MessageBus(max_queue=2)
        bus.publish("a", 1)
        bus.publish("b", 2)
        bus.publish("c", 3)
        assert bus.queue_size() == 2
        msgs = bus.peek(2)
        assert msgs[0].topic == "b"

    def test_flush(self):
        bus = MessageBus()
        for i in range(5):
            bus.publish("topic", i)
        count = bus.flush()
        assert count == 5
        assert bus.queue_size() == 0

    def test_stats(self):
        bus = MessageBus()
        bus.subscribe("topic", lambda msg: None)
        bus.publish("topic", 1)
        bus.flush()
        stats = bus.stats()
        assert stats["published"] == 1
        assert stats["delivered"] == 1

    def test_subscriber_count(self):
        bus = MessageBus()
        bus.subscribe("a", lambda msg: None)
        bus.subscribe("a", lambda msg: None)
        bus.subscribe("b", lambda msg: None)
        assert bus.subscriber_count("a") == 2
        assert bus.subscriber_count() == 3

    def test_repr(self):
        bus = MessageBus()
        assert "MessageBus" in repr(bus)
