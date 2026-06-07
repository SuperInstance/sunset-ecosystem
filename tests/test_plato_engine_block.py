"""Tests for fleet/plato_engine_block.py."""
import pytest
from fleet.plato_engine_block import PlatoEngineBlock, Tick


class TestPlatoEngineBlock:
    def test_basic_tick(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("temp", lambda: 42.0)
        tick = engine.tick_sync()
        assert tick.seq == 1
        assert tick.values["temp"] == 42.0

    def test_history_buffer(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0, history_size=5)
        engine.add_sensor("counter", lambda: 1.0)
        for _ in range(10):
            engine.tick_sync()
        assert len(engine._history) == 5

    def test_actuator(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        calls = []
        engine.add_actuator("pump", lambda v: calls.append(v))
        result = engine.handle_command("actuator pump on")
        assert result == "ok"
        assert calls == ["on"]

    def test_actuator_numeric(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        calls = []
        engine.add_actuator("valve", lambda v: calls.append(v))
        result = engine.handle_command("actuator valve 42.5")
        assert result == "ok"
        assert calls == [42.5]

    def test_unknown_actuator(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        result = engine.handle_command("actuator unknown on")
        assert "error" in result

    def test_tick_command(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("x", lambda: 7.0)
        engine.tick_sync()
        result = engine.handle_command("tick")
        import json
        data = json.loads(result)
        assert data["seq"] == 1
        assert data["x"] == 7.0
        assert data["room"] == "test_room"

    def test_history_command(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("x", lambda: 1.0)
        for _ in range(5):
            engine.tick_sync()
        result = engine.handle_command("history 3")
        import json
        data = json.loads(result)
        assert len(data) == 3
        assert data[0]["seq"] == 3
        assert data[-1]["seq"] == 5

    def test_alarm_command(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        result = engine.handle_command("alarm set overheat temp > 100 notify")
        assert result == "ok"
        assert len(engine._alarms) == 1
        assert engine._alarms[0].name == "overheat"

    def test_alarm_list(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_alarm("a1", "x > 1", "act")
        engine.add_alarm("a2", "y < 0", "warn")
        result = engine.handle_command("alarm list")
        import json
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["name"] == "a1"

    def test_alarm_trigger(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("temp", lambda: 105.0)
        engine.add_alarm("hot", "temp > 100", "cool")
        # Need 3 consecutive ticks to trigger
        for _ in range(3):
            engine.tick_sync()
        assert engine._alarms[0].triggered is True
        assert "__alarm_hot" in engine._history[-1].values

    def test_alarm_no_trigger(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("temp", lambda: 50.0)
        engine.add_alarm("hot", "temp > 100", "cool")
        for _ in range(3):
            engine.tick_sync()
        assert engine._alarms[0].triggered is False

    def test_help_command(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        result = engine.handle_command("help")
        assert "tick" in result
        assert "history" in result
        assert "actuator" in result

    def test_unknown_command(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        result = engine.handle_command("dance")
        assert "error" in result

    def test_empty_line(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        result = engine.handle_command("")
        assert result == ""

    def test_sensor_error_handling(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("bad", lambda: 1 / 0)
        tick = engine.tick_sync()
        assert tick.values["bad"] != tick.values["bad"]  # nan

    def test_multiple_sensors(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("a", lambda: 1.0)
        engine.add_sensor("b", lambda: 2.0)
        tick = engine.tick_sync()
        assert tick.values["a"] == 1.0
        assert tick.values["b"] == 2.0

    def test_subscribe(self):
        engine = PlatoEngineBlock("test_room", tick_hz=10.0)
        engine.add_sensor("x", lambda: 1.0)
        result = engine.handle_command("subscribe")
        assert result == "subscribed"
        assert len(engine._subscribers) == 1
