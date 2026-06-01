"""tests/test_heartbeat_bridge.py — Heartbeat Protocol tests."""

import pytest
import json
import os
import tempfile
from fleet.heartbeat_bridge import Heartbeat, HeartbeatState, ServiceCheck, TaskTile


class TestHeartbeatState:
    def test_roundtrip(self):
        s = HeartbeatState()
        s.acknowledged.add("tile-1")
        s.task_count = 3
        d = s.to_dict()
        assert d["acknowledged"] == ["tile-1"]
        assert d["task_count"] == 3
        s2 = HeartbeatState.from_dict(d)
        assert s2.acknowledged == {"tile-1"}
        assert s2.task_count == 3

    def test_from_empty_dict(self):
        s = HeartbeatState.from_dict({})
        assert s.acknowledged == set()
        assert s.last_check == 0.0


class TestHeartbeat:
    def test_init(self):
        hb = Heartbeat(plato_url="http://test:8080")
        assert hb.plato_url == "http://test:8080"
        assert hb.registry_room == "fleet-registry"

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            state_file = os.path.join(td, "state.json")
            hb = Heartbeat(state_file=state_file)
            hb.state.acknowledged.add("t1")
            hb.save_state()
            assert os.path.exists(state_file)

            hb2 = Heartbeat(state_file=state_file)
            assert "t1" in hb2.state.acknowledged

    def test_discover_rooms_mock(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {"tiles": [{"question": "room: fleet-coord room: test-room"}]}
        rooms = hb.discover_rooms()
        assert "fleet-coord" in rooms
        assert "test-room" in rooms

    def test_discover_rooms_empty(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {"tiles": []}
        rooms = hb.discover_rooms()
        assert "fleet-coord" in rooms

    def test_find_tasks(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {
            "tiles": [
                {"tile_id": "t1", "question": "TASK: build bridge", "source": "FM", "answer": "do it"},
                {"tile_id": "t2", "question": "hello", "source": "O1", "answer": "hi"},
                {"tile_id": "t3", "question": "→O1: fix bug", "source": "JC1", "answer": ""},
            ]
        }
        tasks = hb.find_tasks(rooms=["fleet-coord"])
        assert len(tasks) == 2
        assert tasks[0].tile_id == "t1"
        assert tasks[1].tile_id == "t3"

    def test_find_tasks_acks(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {
            "tiles": [{"tile_id": "t1", "question": "TASK: x", "source": "FM", "answer": ""}]
        }
        hb.ack("t1")
        tasks = hb.find_tasks(rooms=["fleet-coord"])
        assert len(tasks) == 0

    def test_check_services_ok(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {"ok": True}
        results = hb.check_services()
        assert results.get("PLATO") == "ok"

    def test_check_services_fail(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {"error": "timeout"}
        results = hb.check_services()
        assert "unreachable" in results.get("PLATO", "")

    def test_run(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {"tiles": []}
        report = hb.run()
        assert "Heartbeat" in report
        assert "All quiet" in report

    def test_run_with_tasks(self):
        hb = Heartbeat(plato_url="http://test:8080")
        hb._fetch_fn = lambda url, timeout: {
            "tiles": [{"tile_id": "t1", "question": "TASK: build", "source": "FM", "answer": ""}]
        }
        report = hb.run()
        assert "1 new task" in report
        assert "FM" in report

    def test_to_dict(self):
        hb = Heartbeat(plato_url="http://test:8080", state_file="/tmp/state.json")
        d = hb.to_dict()
        assert d["plato_url"] == "http://test:8080"
        assert d["registry_room"] == "fleet-registry"


class TestTaskTile:
    def test_fields(self):
        tile = TaskTile(
            tile_id="t1",
            room="fleet-coord",
            question="TASK: x",
            answer="do it",
            source="FM",
        )
        assert tile.tile_id == "t1"
        assert tile.room == "fleet-coord"
