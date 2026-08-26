"""Tests for nexus.fleet_event_bus.FleetEventBus.

Covers synchronous and async publish/subscribe, filtering, history,
threading safety, and the CCC-OS → sunset-ecosystem integration
patterns identified in the fleet mesh analysis.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from nexus.fleet_event_bus import FleetEventBus, FleetEvent


class TestBasicPubSub:
    """Single-threaded sync publish / sync subscribe."""

    def test_emit_triggers_handler(self):
        bus = FleetEventBus()
        received: list[FleetEvent] = []
        bus.on("ping", lambda ev: received.append(ev))

        bus.emit({"type": "ping", "msg": "hello"}, source="test")

        assert len(received) == 1
        assert received[0].type == "ping"
        assert received[0].payload["msg"] == "hello"
        assert received[0].source == "test"

    def test_multiple_handlers_same_type(self):
        bus = FleetEventBus()
        a: list[str] = []
        b: list[str] = []
        bus.on("tick", lambda ev: a.append("a"))
        bus.on("tick", lambda ev: b.append("b"))

        bus.emit({"type": "tick"})

        assert a == ["a"]
        assert b == ["b"]

    def test_no_handlers_does_not_crash(self):
        bus = FleetEventBus()
        bus.emit({"type": "orphan"})  # should not raise

    def test_handler_exception_does_not_kill_bus(self):
        bus = FleetEventBus()
        bus.on("boom", lambda ev: (_ for _ in ()).throw(RuntimeError("bad")))
        bus.on("boom", lambda ev: bus.payloads.append("ok"))
        bus.payloads = []  # type: ignore[attr-defined]

        bus.emit({"type": "boom"})

        assert bus.payloads == ["ok"]  # second handler still ran


class TestFiltering:
    """Per-handler filters."""

    def test_filter_blocks_unwanted_events(self):
        bus = FleetEventBus()
        hits: list[str] = []
        bus.on(
            "alert",
            lambda ev: hits.append(ev.payload["severity"]),
            filter_fn=lambda ev: ev.payload.get("severity") == "critical",
        )

        bus.emit({"type": "alert", "severity": "info"})
        bus.emit({"type": "alert", "severity": "critical"})
        bus.emit({"type": "alert", "severity": "warning"})

        assert hits == ["critical"]

    def test_filter_allows_all_when_none(self):
        bus = FleetEventBus()
        hits: list[str] = []
        bus.on("msg", lambda ev: hits.append(ev.payload["txt"]))

        bus.emit({"type": "msg", "txt": "a"})
        bus.emit({"type": "msg", "txt": "b"})

        assert hits == ["a", "b"]


class TestFleetEventStruct:
    """FleetEvent dataclass round-trips correctly."""

    def test_to_dict(self):
        ev = FleetEvent(type="test", payload={"x": 1}, source="unit")
        d = ev.to_dict()
        assert d["type"] == "test"
        assert d["payload"]["x"] == 1
        assert d["source"] == "unit"
        assert "timestamp" in d
        assert "event_id" in d

    def test_to_json(self):
        ev = FleetEvent(type="test", payload={"x": 1})
        s = ev.to_json()
        assert "test" in s
        assert "x" in s


class TestAsyncHandlers:
    """Async handler scheduling."""

    def test_async_handler_scheduled(self):
        bus = FleetEventBus()
        received: list[str] = []

        async def handler(ev: FleetEvent) -> None:
            received.append(ev.payload["val"])

        bus.on_async("async_ping", handler)

        bus.emit({"type": "async_ping", "val": "async"})

        # Async handlers are scheduled on the event loop.
        # Give the loop a tick to process.
        loop = bus._ensure_loop()
        loop.run_until_complete(asyncio.sleep(0.05))

        assert "async" in received


class TestHistory:
    """Event history retention."""

    def test_recent_events_default(self):
        bus = FleetEventBus()
        for i in range(5):
            bus.emit({"type": "heartbeat", "seq": i})

        recent = bus.recent_events()
        assert len(recent) == 5
        assert recent[0].payload["seq"] == 4  # reversed

    def test_recent_events_filter_by_type(self):
        bus = FleetEventBus()
        bus.emit({"type": "a", "x": 1})
        bus.emit({"type": "b", "x": 2})
        bus.emit({"type": "a", "x": 3})

        only_a = bus.recent_events(event_type="a")
        assert len(only_a) == 2
        assert only_a[0].payload["x"] == 3

    def test_history_limit(self):
        bus = FleetEventBus()
        bus._history_limit = 3
        for i in range(10):
            bus.emit({"type": "flood", "seq": i})

        assert len(bus._history) == 3


class TestStats:
    """Introspection helpers."""

    def test_stats_basic(self):
        bus = FleetEventBus()
        bus.on("alpha", lambda ev: None)
        bus.on("alpha", lambda ev: None)
        bus.on("beta", lambda ev: None)

        stats = bus.stats()
        assert stats["total_event_types"] == 2
        assert stats["total_handlers"] == 3
        assert stats["history_size"] == 0


class TestOff:
    """Handler removal."""

    def test_off_removes_handler(self):
        bus = FleetEventBus()
        hits: list[str] = []
        fn = lambda ev: hits.append("hit")
        bus.on("x", fn)
        bus.emit({"type": "x"})
        assert hits == ["hit"]

        removed = bus.off("x", fn)
        assert removed is True
        bus.emit({"type": "x"})
        assert hits == ["hit"]  # no new hit

    def test_off_returns_false_when_missing(self):
        bus = FleetEventBus()
        removed = bus.off("y", lambda ev: None)
        assert removed is False


class TestThreadSafety:
    """Concurrent publish from multiple threads."""

    def test_concurrent_emit(self):
        bus = FleetEventBus()
        received: list[int] = []
        bus.on("num", lambda ev: received.append(ev.payload["n"]))

        def emitter(n: int) -> None:
            bus.emit({"type": "num", "n": n})

        with ThreadPoolExecutor(max_workers=8) as pool:
            for i in range(50):
                pool.submit(emitter, i)

        time.sleep(0.1)
        assert len(received) == 50
        assert set(received) == set(range(50))


class TestCCCOSIntegrationPattern:
    """Simulate the CCC-OS → sunset-ecosystem ACT_NOW pattern."""

    def test_act_now_spawns_breeder(self):
        bus = FleetEventBus()
        spawned: list[dict] = []

        def on_act_now(ev: FleetEvent) -> None:
            if ev.payload.get("category") == "architecture":
                spawned.append(
                    {
                        "repo": ev.payload["repo"],
                        "priority": ev.payload["priority"],
                    }
                )

        bus.on("ACT_NOW", on_act_now)

        bus.emit(
            {
                "type": "ACT_NOW",
                "category": "architecture",
                "repo": "sunset-ecosystem",
                "priority": "P0",
            },
            source="ccc-os",
        )

        assert len(spawned) == 1
        assert spawned[0]["repo"] == "sunset-ecosystem"
        assert spawned[0]["priority"] == "P0"

    def test_service_down_triggers_rebalance(self):
        bus = FleetEventBus()
        rebalanced: list[str] = []

        bus.on("service_down", lambda ev: rebalanced.append(ev.payload["node"]))

        bus.emit(
            {
                "type": "service_down",
                "node": "plato-pipeline-1",
            },
            source="cocapn-health",
        )

        assert rebalanced == ["plato-pipeline-1"]

    def test_trap_anomaly_quarantines(self):
        bus = FleetEventBus()
        quarantined: list[str] = []

        bus.on(
            "rule_injection_spike",
            lambda ev: quarantined.append(ev.payload["agent_id"]),
            filter_fn=lambda ev: ev.payload.get("rule_count", 0) > 500,
        )

        bus.emit(
            {
                "type": "rule_injection_spike",
                "agent_id": "scout-42",
                "rule_count": 1000,
            },
            source="cocapn-traps",
        )

        assert quarantined == ["scout-42"]


class TestAsyncEmit:
    """emit_async inside an async context."""

    @pytest.mark.asyncio
    async def test_emit_async(self):
        bus = FleetEventBus()
        received: list[str] = []
        bus.on("async_event", lambda ev: received.append(ev.payload["data"]))

        await bus.emit_async({"type": "async_event", "data": "async"})

        assert received == ["async"]
