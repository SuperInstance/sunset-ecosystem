import pytest
from fleet.event_bus import Event, EventBus


class TestEvent:
    def test_to_dict(self):
        e = Event(
            event_type="test",
            payload={"x": 1},
            timestamp=0.0,
            source="node1",
        )
        d = e.to_dict()
        assert d["event_type"] == "test"
        assert d["payload"]["x"] == 1


class TestEventBus:
    def test_init(self):
        bus = EventBus()
        assert bus.fleet_node_id == "default"
        assert bus._subscribers == {}

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e))
        bus.publish("test", {"x": 1})
        assert len(received) == 1
        assert received[0].payload["x"] == 1

    def test_unsubscribe(self):
        bus = EventBus()
        handler = lambda e: None
        bus.subscribe("test", handler)
        assert bus.unsubscribe("test", handler) is True
        assert bus.unsubscribe("test", handler) is False

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish("a", {"x": 1})
        bus.publish("b", {"y": 2})
        assert len(received) == 2

    def test_get_history(self):
        bus = EventBus()
        bus.publish("test", {"x": 1})
        bus.publish("test", {"x": 2})
        bus.publish("other", {"y": 1})
        history = bus.get_history("test")
        assert len(history) == 2
        assert history[0].payload["x"] == 1

    def test_get_history_limit(self):
        bus = EventBus()
        for i in range(10):
            bus.publish("test", {"x": i})
        history = bus.get_history("test", limit=5)
        assert len(history) == 5

    def test_get_stats(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.publish("test", {"x": 1})
        stats = bus.get_stats()
        assert stats["subscribers"]["test"] == 1
        assert stats["history_size"] == 1

    def test_export_json(self):
        bus = EventBus()
        bus.publish("test", {"x": 1})
        j = bus.export_json()
        assert "test" in j
        assert "history" in j

    def test_to_dict(self):
        bus = EventBus()
        bus.publish("test", {"x": 1})
        d = bus.to_dict()
        assert "stats" in d
