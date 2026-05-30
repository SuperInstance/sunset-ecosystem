"""Tests for swarm.broadcast — pub/sub with Hebbian strengthening."""

import time

import pytest

from swarm.broadcast import BroadcastMessage, BroadcastingChannel


class TestBroadcastMessage:
    def test_create(self):
        msg = BroadcastMessage(content="hello", source_agent="a1", target_room="r1")
        assert msg.content == "hello"
        assert msg.source_agent == "a1"
        assert msg.target_room == "r1"
        assert 0.0 <= msg.relevance_score <= 1.0
        assert msg.timestamp <= time.time()

    def test_repr(self):
        msg = BroadcastMessage(content="x" * 100, source_agent="a1", target_room="r1")
        r = repr(msg)
        assert "a1" in r
        assert "r1" in r
        assert "..." in r


class TestBroadcastingChannel:
    def test_create(self):
        ch = BroadcastingChannel()
        assert ch.subscription_count == 0

    def test_subscribe(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        assert ch.subscription_count == 1

    def test_unsubscribe(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        ch.unsubscribe("agent1", "room_a")
        assert ch.subscription_count == 0

    def test_broadcast_no_subscribers(self):
        ch = BroadcastingChannel()
        msg = BroadcastMessage(content="hi", target_room="room_a")
        recipients = ch.broadcast(msg)
        assert recipients == []

    def test_broadcast_match(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        msg = BroadcastMessage(content="hi", target_room="room_a")
        recipients = ch.broadcast(msg)
        assert "agent1" in recipients

    def test_broadcast_no_match(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_b")
        msg = BroadcastMessage(content="hi", target_room="room_a")
        recipients = ch.broadcast(msg)
        assert "agent1" not in recipients

    def test_broadcast_multiple_agents(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        ch.subscribe("agent2", "room_a")
        msg = BroadcastMessage(content="hi", target_room="room_a")
        recipients = ch.broadcast(msg)
        assert sorted(recipients) == ["agent1", "agent2"]

    def test_broadcast_hebbian_strengthen(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        msg = BroadcastMessage(content="hi", source_agent="src", target_room="room_a")
        ch.broadcast(msg)
        weight = ch.get_channel_weight("src", "agent1")
        assert weight > 0.0

    def test_receive(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        msg = BroadcastMessage(content="hi", target_room="room_a")
        ch.broadcast(msg)
        received = ch.receive("agent1")
        assert len(received) == 1
        assert received[0].content == "hi"

    def test_receive_clears_queue(self):
        ch = BroadcastingChannel()
        ch.subscribe("agent1", "room_a")
        msg = BroadcastMessage(content="hi", target_room="room_a")
        ch.broadcast(msg)
        ch.receive("agent1")
        received = ch.receive("agent1")
        assert received == []

    def test_feedback_useful(self):
        ch = BroadcastingChannel()
        ch.feedback("src", "dst", useful=True)
        assert ch.get_channel_weight("src", "dst") > 0.0

    def test_feedback_not_useful(self):
        ch = BroadcastingChannel()
        ch.feedback("src", "dst", useful=True)
        initial = ch.get_channel_weight("src", "dst")
        ch.feedback("src", "dst", useful=False)
        assert ch.get_channel_weight("src", "dst") < initial

    def test_max_queue(self):
        ch = BroadcastingChannel(max_queue_size=2)
        ch.subscribe("agent1", "room_a")
        for i in range(5):
            msg = BroadcastMessage(content=i, target_room="room_a")
            ch.broadcast(msg)
        received = ch.receive("agent1")
        assert len(received) == 2
        assert received[-1].content == 4  # latest kept

    def test_repr(self):
        ch = BroadcastingChannel()
        assert "BroadcastingChannel" in repr(ch)
