"""Tests for event_stream.py — Persistent event streaming.

Run: python3 -m pytest tests/test_event_stream.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.event_stream import EventStream


class TestEventStream:
    def test_create(self):
        stream = EventStream()
        assert stream.event_count() == 0

    def test_append(self):
        stream = EventStream()
        event = stream.append({"type": "spawn"})
        assert event.offset == 0
        assert event.data["type"] == "spawn"

    def test_append_multiple(self):
        stream = EventStream()
        for i in range(5):
            stream.append({"id": i})
        assert stream.event_count() == 5
        assert stream._offset == 5

    def test_replay(self):
        stream = EventStream()
        stream.append({"type": "a"})
        stream.append({"type": "b"})
        stream.append({"type": "c"})
        events = stream.replay(since_offset=1)
        assert len(events) == 2
        assert events[0].data["type"] == "b"

    def test_replay_topic_filter(self):
        stream = EventStream()
        stream.append({"type": "a"}, topic="topic1")
        stream.append({"type": "b"}, topic="topic2")
        events = stream.replay(topic="topic1")
        assert len(events) == 1
        assert events[0].data["type"] == "a"

    def test_replay_type_filter(self):
        stream = EventStream()
        stream.append({"type": "spawn"})
        stream.append({"type": "die"})
        stream.append({"type": "spawn"})
        events = stream.replay(event_type="spawn")
        assert len(events) == 2

    def test_replay_limit(self):
        stream = EventStream()
        for i in range(10):
            stream.append({"id": i})
        events = stream.replay(limit=3)
        assert len(events) == 3

    def test_latest(self):
        stream = EventStream()
        stream.append({"type": "a"})
        stream.append({"type": "b"})
        latest = stream.latest()
        assert latest is not None
        assert latest.data["type"] == "b"

    def test_latest_topic(self):
        stream = EventStream()
        stream.append({"type": "a"}, topic="t1")
        stream.append({"type": "b"}, topic="t2")
        latest = stream.latest(topic="t1")
        assert latest is not None
        assert latest.data["type"] == "a"

    def test_consumer_group(self):
        stream = EventStream()
        cg = stream.create_consumer_group("group1")
        assert cg.name == "group1"

    def test_consume(self):
        stream = EventStream()
        stream.create_consumer_group("g1")
        stream.append({"type": "a"})
        stream.append({"type": "b"})
        events = stream.consume("g1")
        assert len(events) == 2
        # Second consume should return nothing (already consumed)
        events2 = stream.consume("g1")
        assert len(events2) == 0

    def test_consume_with_limit(self):
        stream = EventStream()
        stream.create_consumer_group("g1")
        for i in range(10):
            stream.append({"id": i})
        events = stream.consume("g1", limit=3)
        assert len(events) == 3

    def test_consumer_offset(self):
        stream = EventStream()
        stream.create_consumer_group("g1")
        stream.append({"type": "a"})
        stream.consume("g1")
        assert stream.consumer_offset("g1") == 1

    def test_reset_consumer_offset(self):
        stream = EventStream()
        stream.create_consumer_group("g1")
        stream.append({"type": "a"})
        stream.consume("g1")
        stream.reset_consumer_offset("g1", offset=0)
        assert stream.consumer_offset("g1") == 0

    def test_max_events_eviction(self):
        stream = EventStream(max_events=3)
        for i in range(5):
            stream.append({"id": i})
        assert stream.event_count() == 3

    def test_ttl_eviction(self):
        stream = EventStream(ttl_seconds=0.1)
        stream.append({"type": "a"})
        time.sleep(0.15)
        stream.append({"type": "b"})
        assert stream.event_count() == 1

    def test_topics(self):
        stream = EventStream()
        stream.append({"type": "a"}, topic="t1")
        stream.append({"type": "b"}, topic="t2")
        assert stream.topics() == ["t1", "t2"]

    def test_stats(self):
        stream = EventStream()
        stream.append({"type": "a"})
        stream.create_consumer_group("g1")
        stats = stream.stats()
        assert stats["total_events"] == 1
        assert stats["consumer_groups"] == 1

    def test_repr(self):
        stream = EventStream()
        assert "EventStream" in repr(stream)
