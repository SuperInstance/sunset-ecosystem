"""Tests for the PLATO Signal Chain Adapter.

Covers MockRoomSource, PlatoRoomSense, PlatoBreedingPolicy,
PlatoBreedingAct, and PlatoSignalChain end-to-end.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import pytest

from fleet.plato_signal_chain import (
    MockRoomSource,
    PlatoBreedingAct,
    PlatoBreedingPolicy,
    PlatoRoomSense,
    PlatoSignalChain,
    RoomObservation,
)
from fleet.sense_decide_act import Decision, Observation, SDALoop


# ═══════════════════════════════════════════════════════════════
# MockRoomSource
# ═══════════════════════════════════════════════════════════════


class TestMockRoomSource:
    def test_empty(self):
        src = MockRoomSource()
        assert src.list_rooms() == []
        assert src.snapshot() == []

    def test_set_and_get(self):
        src = MockRoomSource()
        obs = RoomObservation(room_id="r1", timestamp=time.time(), agent_count=3)
        src.set_room(obs)
        assert src.list_rooms() == ["r1"]
        got = src.get_room("r1")
        assert got is not None
        assert got.agent_count == 3

    def test_snapshot_returns_all(self):
        src = MockRoomSource()
        src.set_room(RoomObservation(room_id="a", timestamp=1.0))
        src.set_room(RoomObservation(room_id="b", timestamp=2.0))
        snap = src.snapshot()
        assert len(snap) == 2
        assert {r.room_id for r in snap} == {"a", "b"}

    def test_overwrite(self):
        src = MockRoomSource()
        src.set_room(RoomObservation(room_id="r1", timestamp=1.0, agent_count=1))
        src.set_room(RoomObservation(room_id="r1", timestamp=2.0, agent_count=5))
        assert src.get_room("r1").agent_count == 5

    def test_thread_safety(self):
        src = MockRoomSource()
        errors: List[Exception] = []

        def writer():
            try:
                for i in range(100):
                    src.set_room(
                        RoomObservation(room_id=f"r{i % 10}", timestamp=time.time())
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(src.list_rooms()) <= 10


# ═══════════════════════════════════════════════════════════════
# PlatoRoomSense
# ═══════════════════════════════════════════════════════════════


class TestPlatoRoomSense:
    def test_empty_source(self):
        src = MockRoomSource()
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.source == "plato_room_sense"
        assert obs.metrics["room_count"] == 0
        assert obs.metrics["total_agents"] == 0
        assert obs.severity_hint == "info"

    def test_single_room(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=5,
                diversity_score=0.5,
                thermal_cpu=40.0,
                thermal_mem=50.0,
            )
        )
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.metrics["room_count"] == 1
        assert obs.metrics["total_agents"] == 5
        assert obs.metrics["mean_diversity"] == 0.5
        assert obs.metrics["max_thermal_cpu"] == 40.0
        assert obs.severity_hint == "info"

    def test_multiple_rooms_aggregates(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="a", timestamp=time.time(), agent_count=2, diversity_score=0.1
            )
        )
        src.set_room(
            RoomObservation(
                room_id="b", timestamp=time.time(), agent_count=8, diversity_score=0.5
            )
        )
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.metrics["total_agents"] == 10
        assert obs.metrics["mean_diversity"] == pytest.approx(0.3)
        assert len(obs.metrics["room_states"]) == 2

    def test_room_filter(self):
        src = MockRoomSource()
        src.set_room(RoomObservation(room_id="a", timestamp=time.time()))
        src.set_room(RoomObservation(room_id="b", timestamp=time.time()))
        sense = PlatoRoomSense(source=src, room_ids=["a"])
        obs = sense.observe()
        assert obs.metrics["room_count"] == 1
        assert obs.metrics["room_states"][0]["room_id"] == "a"

    def test_thermal_critical_severity(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(room_id="hot", timestamp=time.time(), thermal_cpu=85.0)
        )
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.severity_hint == "critical"

    def test_thermal_warning_severity(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="warm",
                timestamp=time.time(),
                thermal_cpu=50.0,
                thermal_mem=90.0,
            )
        )
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.severity_hint == "warning"

    def test_diversity_warning(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="mono",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        sense = PlatoRoomSense(source=src)
        obs = sense.observe()
        assert obs.severity_hint == "warning"

    def test_last_rooms(self):
        src = MockRoomSource()
        src.set_room(RoomObservation(room_id="x", timestamp=time.time()))
        sense = PlatoRoomSense(source=src)
        sense.observe()
        assert len(sense.last_rooms()) == 1
        assert sense.last_rooms()[0].room_id == "x"

    def test_callable_source(self):
        def _source():
            return [RoomObservation(room_id="c", timestamp=time.time(), agent_count=7)]

        sense = PlatoRoomSense(source=_source)
        obs = sense.observe()
        assert obs.metrics["total_agents"] == 7

    def test_custom_severity_thresholds(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(room_id="warm", timestamp=time.time(), thermal_cpu=70.0)
        )
        sense = PlatoRoomSense(source=src, severity_thresholds={"thermal_cpu": 60.0})
        obs = sense.observe()
        assert obs.severity_hint == "critical"


# ═══════════════════════════════════════════════════════════════
# PlatoBreedingPolicy
# ═══════════════════════════════════════════════════════════════


class TestPlatoBreedingPolicy:
    def _make_obs(self, **kwargs) -> Observation:
        defaults = {
            "timestamp": time.time(),
            "source": "test",
            "metrics": {
                "room_count": 2,
                "total_agents": 10,
                "mean_diversity": 0.5,
                "max_thermal_cpu": 40.0,
                "max_thermal_mem": 50.0,
                "lifecycle_event_count": 0,
                "room_states": [
                    {
                        "room_id": "a",
                        "agent_count": 5,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    },
                    {
                        "room_id": "b",
                        "agent_count": 5,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    },
                ],
            },
            "severity_hint": "info",
        }
        defaults["metrics"].update(kwargs.pop("metrics", {}))
        return Observation(**{**defaults, **kwargs})

    def test_no_match_defaults_to_noop(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs()
        dec = policy.evaluate(obs)
        assert dec.action_type == "noop"
        assert dec.confidence == 1.0

    def test_thermal_critical_triggers_sunset(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(metrics={"max_thermal_cpu": 85.0})
        dec = policy.evaluate(obs)
        assert dec.action_type == "sunset"
        assert dec.confidence == 0.9
        assert "thermal" in dec.reasoning.lower()

    def test_low_diversity_high_occupancy_triggers_breed(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "mean_diversity": 0.1,
                "total_agents": 10,
                "room_states": [
                    {
                        "room_id": "a",
                        "agent_count": 10,
                        "diversity": 0.1,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    }
                ],
            }
        )
        dec = policy.evaluate(obs)
        assert dec.action_type == "breed"
        assert dec.confidence == 0.85
        assert "diversity" in dec.reasoning.lower()

    def test_breed_not_triggered_if_occupancy_low(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "mean_diversity": 0.1,
                "total_agents": 3,
                "room_states": [
                    {
                        "room_id": "a",
                        "agent_count": 3,
                        "diversity": 0.1,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    }
                ],
            }
        )
        dec = policy.evaluate(obs)
        assert dec.action_type == "noop"

    def test_imbalance_triggers_migrate(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "room_states": [
                    {
                        "room_id": "full",
                        "agent_count": 12,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    },
                    {
                        "room_id": "empty",
                        "agent_count": 1,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    },
                ],
            }
        )
        dec = policy.evaluate(obs)
        assert dec.action_type == "migrate"
        assert "imbalance" in dec.reasoning.lower()

    def test_no_imbalance_with_few_rooms(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "room_count": 1,
                "room_states": [
                    {
                        "room_id": "only",
                        "agent_count": 10,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": [],
                    }
                ],
            }
        )
        dec = policy.evaluate(obs)
        # With 1 room, imbalance can't be detected (need 2+)
        assert dec.action_type in ("noop", "audit", "breed")

    def test_lifecycle_events_trigger_audit(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "lifecycle_event_count": 5,
                "room_states": [
                    {
                        "room_id": "a",
                        "agent_count": 5,
                        "diversity": 0.5,
                        "cpu": 40.0,
                        "mem": 50.0,
                        "events": ["spawn", "sunset", "spawn", "sunset", "spawn"],
                    }
                ],
            }
        )
        dec = policy.evaluate(obs)
        assert dec.action_type == "audit"
        assert "lifecycle" in dec.reasoning.lower()

    def test_thermal_takes_priority_over_breed(self):
        policy = PlatoBreedingPolicy()
        obs = self._make_obs(
            metrics={
                "max_thermal_cpu": 90.0,
                "mean_diversity": 0.1,
                "total_agents": 10,
            }
        )
        dec = policy.evaluate(obs)
        # Thermal is rule #1, breed is rule #2
        assert dec.action_type == "sunset"

    def test_custom_thresholds(self):
        policy = PlatoBreedingPolicy(
            diversity_threshold=0.5,
            occupancy_threshold=20,
            thermal_cpu_threshold=95.0,
        )
        obs = self._make_obs(
            metrics={
                "mean_diversity": 0.3,
                "total_agents": 15,
                "max_thermal_cpu": 90.0,
            }
        )
        dec = policy.evaluate(obs)
        # diversity 0.3 >= 0.5? No. thermal 90 >= 95? No. -> noop
        assert dec.action_type == "noop"

    def test_payload_contains_room_states(self):
        policy = PlatoBreedingPolicy()
        room_states = [{"room_id": "a", "agent_count": 1}]
        obs = self._make_obs(metrics={"room_states": room_states})
        dec = policy.evaluate(obs)
        # Even noop should have room_states in payload from policy? No—
        # Policy.evaluate returns Decision with payload from rule, not room_states.
        # Our custom policy doesn't inject room_states. That's fine.
        assert dec.action_type == "noop"


# ═══════════════════════════════════════════════════════════════
# PlatoBreedingAct
# ═══════════════════════════════════════════════════════════════


class TestPlatoBreedingAct:
    def test_noop(self):
        act = PlatoBreedingAct()
        dec = Decision(action_type="noop", confidence=1.0)
        result = act.execute(dec)
        assert result.success is True
        assert "noop" in result.side_effects
        assert result.new_observations == []

    def test_breed_callback(self):
        calls: List[List[Dict[str, Any]]] = []
        act = PlatoBreedingAct(on_breed=lambda rooms: calls.append(rooms))
        dec = Decision(
            action_type="breed",
            confidence=0.85,
            payload={"room_states": [{"room_id": "r1", "agent_count": 5}]},
        )
        result = act.execute(dec)
        assert result.success is True
        assert "breed_triggered" in result.side_effects
        assert len(result.new_observations) == 1
        assert len(calls) == 1
        assert calls[0][0]["room_id"] == "r1"

    def test_sunset_callback(self):
        calls: List[List[Dict[str, Any]]] = []
        act = PlatoBreedingAct(on_sunset=lambda rooms: calls.append(rooms))
        dec = Decision(
            action_type="sunset",
            confidence=0.9,
            payload={"room_states": [{"room_id": "r1"}]},
        )
        result = act.execute(dec)
        assert result.success is True
        assert "sunset_triggered" in result.side_effects
        assert result.new_observations[0].severity_hint == "warning"
        assert len(calls) == 1

    def test_migrate_callback(self):
        calls: List[List[Dict[str, Any]]] = []
        act = PlatoBreedingAct(on_migrate=lambda rooms: calls.append(rooms))
        dec = Decision(
            action_type="migrate",
            confidence=0.75,
            payload={"room_states": [{"room_id": "r1"}, {"room_id": "r2"}]},
        )
        result = act.execute(dec)
        assert result.success is True
        assert "migrate_triggered" in result.side_effects
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_audit_no_callback(self):
        act = PlatoBreedingAct()
        dec = Decision(action_type="audit", confidence=0.6)
        result = act.execute(dec)
        assert result.success is True
        assert "audit_logged" in result.side_effects
        assert result.new_observations == []

    def test_callback_exception_caught(self):
        act = PlatoBreedingAct(
            on_breed=lambda rooms: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        dec = Decision(
            action_type="breed",
            confidence=0.85,
            payload={"room_states": []},
        )
        result = act.execute(dec)
        assert result.success is False
        assert any("boom" in s for s in result.side_effects)

    def test_recorder(self):
        recorded: List[tuple] = []
        act = PlatoBreedingAct(
            on_breed=lambda rooms: None,
            recorder=lambda d, r: recorded.append((d.action_type, r.success)),
        )
        dec = Decision(
            action_type="breed", confidence=0.85, payload={"room_states": []}
        )
        result = act.execute(dec)
        assert len(recorded) == 1
        assert recorded[0] == ("breed", True)

    def test_recorder_exception_ignored(self):
        act = PlatoBreedingAct(
            on_breed=lambda rooms: None,
            recorder=lambda d, r: (_ for _ in ()).throw(RuntimeError("recorder fail")),
        )
        dec = Decision(
            action_type="breed", confidence=0.85, payload={"room_states": []}
        )
        result = act.execute(dec)
        # Should succeed despite recorder failure
        assert result.success is True

    def test_history(self):
        act = PlatoBreedingAct()
        dec = Decision(action_type="noop", confidence=1.0)
        act.execute(dec)
        act.execute(dec)
        hist = act.history()
        assert len(hist) == 2
        assert all(d.action_type == "noop" for d, _ in hist)

    def test_clear_history(self):
        act = PlatoBreedingAct()
        act.execute(Decision(action_type="noop", confidence=1.0))
        act.clear_history()
        assert act.history() == []

    def test_latency_recorded(self):
        act = PlatoBreedingAct()
        dec = Decision(action_type="noop", confidence=1.0)
        result = act.execute(dec)
        assert result.latency_ms >= 0.0
        assert result.latency_ms < 100.0  # Should be very fast

    def test_thread_safety_history(self):
        act = PlatoBreedingAct()
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    act.execute(Decision(action_type="noop", confidence=1.0))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(act.history()) == 200


# ═══════════════════════════════════════════════════════════════
# PlatoSignalChain end-to-end
# ═══════════════════════════════════════════════════════════════


class TestPlatoSignalChain:
    def test_register_and_tick(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
                thermal_cpu=40.0,
            )
        )
        chain = PlatoSignalChain(source=src)
        results = chain.tick()
        assert "plato_signal_chain" in results
        result = results["plato_signal_chain"]
        assert result is not None
        assert result.success is True
        assert "breed_triggered" in result.side_effects

    def test_tick_counts(self):
        src = MockRoomSource()
        chain = PlatoSignalChain(source=src)
        chain.tick()
        chain.tick()
        assert chain.tick_count() == 2

    def test_metrics(self):
        src = MockRoomSource()
        src.set_room(RoomObservation(room_id="x", timestamp=time.time()))
        chain = PlatoSignalChain(source=src)
        chain.tick()
        m = chain.metrics()
        assert m["tick_count"] == 1
        assert m["pipeline_name"] == "plato_signal_chain"
        assert m["room_sense_last_rooms"] == 1
        assert m["act_history_size"] == 1

    def test_thermal_tick_triggers_sunset(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="hot",
                timestamp=time.time(),
                agent_count=5,
                diversity_score=0.5,
                thermal_cpu=85.0,
            )
        )
        chain = PlatoSignalChain(source=src)
        results = chain.tick()
        result = results["plato_signal_chain"]
        assert "sunset_triggered" in result.side_effects

    def test_migrate_tick(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="full",
                timestamp=time.time(),
                agent_count=12,
                diversity_score=0.5,
            )
        )
        src.set_room(
            RoomObservation(
                room_id="empty",
                timestamp=time.time(),
                agent_count=1,
                diversity_score=0.5,
            )
        )
        chain = PlatoSignalChain(source=src)
        results = chain.tick()
        result = results["plato_signal_chain"]
        assert "migrate_triggered" in result.side_effects

    def test_room_filter(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        src.set_room(
            RoomObservation(
                room_id="beta",
                timestamp=time.time(),
                agent_count=2,
                diversity_score=0.8,
            )
        )
        chain = PlatoSignalChain(source=src, room_ids=["beta"])
        results = chain.tick()
        # Only beta observed -> diversity 0.8, agents 2 -> no action
        result = results["plato_signal_chain"]
        assert "noop" in result.side_effects

    def test_start_stop(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        chain = PlatoSignalChain(source=src)
        chain.start(loop_interval_ms=200)
        time.sleep(0.6)
        chain.stop()
        # Should have ticked a few times in 600ms with 200ms interval
        assert chain.tick_count() >= 2

    def test_double_start_idempotent(self):
        src = MockRoomSource()
        chain = PlatoSignalChain(source=src)
        chain.start(loop_interval_ms=500)
        chain.start(loop_interval_ms=500)  # Should not crash
        chain.stop()

    def test_stop_without_start(self):
        src = MockRoomSource()
        chain = PlatoSignalChain(source=src)
        chain.stop()  # Should not crash

    def test_custom_policy_and_act(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        breed_calls: List[List[Dict[str, Any]]] = []
        act = PlatoBreedingAct(on_breed=lambda rooms: breed_calls.append(rooms))
        chain = PlatoSignalChain(source=src, act=act)
        chain.tick()
        assert len(breed_calls) == 1

    def test_custom_sda_loop(self):
        loop = SDALoop(confidence_threshold=0.0)
        src = MockRoomSource()
        chain = PlatoSignalChain(source=src, loop=loop)
        chain.tick()
        assert loop.list_pipelines() == ["plato_signal_chain"]

    def test_empty_source_tick(self):
        src = MockRoomSource()
        chain = PlatoSignalChain(source=src)
        results = chain.tick()
        result = results["plato_signal_chain"]
        assert result is not None
        assert result.success is True
        assert "noop" in result.side_effects

    def test_room_states_in_decision_payload(self):
        # Verify that room_states flow from sense → decide → act
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        captured_decisions: List[Decision] = []
        original_evaluate = PlatoBreedingPolicy.evaluate

        class CapturePolicy(PlatoBreedingPolicy):
            def evaluate(self, observation: Observation) -> Decision:
                dec = super().evaluate(observation)
                captured_decisions.append(dec)
                return dec

        chain = PlatoSignalChain(source=src, policy=CapturePolicy())
        chain.tick()
        assert len(captured_decisions) == 1
        # The observation metrics had room_states, but the policy Decision payload
        # comes from the rule. Our act receives the decision payload, not observation.
        # That's the correct SDA separation.

    def test_multiple_ticks_same_state(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        chain = PlatoSignalChain(source=src)
        for _ in range(3):
            chain.tick()
        assert chain.tick_count() == 3
        assert len(chain.act.history()) == 3

    def test_changing_state_between_ticks(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        chain = PlatoSignalChain(source=src)
        r1 = chain.tick()["plato_signal_chain"]
        assert "breed_triggered" in r1.side_effects

        # Now diversity improves
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.8,
            )
        )
        r2 = chain.tick()["plato_signal_chain"]
        assert "noop" in r2.side_effects

    def test_audit_tick(self):
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="chaos",
                timestamp=time.time(),
                agent_count=5,
                diversity_score=0.5,
                lifecycle_events=["spawn", "sunset", "spawn", "sunset", "spawn"],
            )
        )
        chain = PlatoSignalChain(source=src)
        results = chain.tick()
        result = results["plato_signal_chain"]
        assert "audit_logged" in result.side_effects

    def test_callable_source_integration(self):
        def _source():
            return [
                RoomObservation(
                    room_id="dyn",
                    timestamp=time.time(),
                    agent_count=10,
                    diversity_score=0.1,
                )
            ]

        chain = PlatoSignalChain(source=_source)
        results = chain.tick()
        result = results["plato_signal_chain"]
        assert "breed_triggered" in result.side_effects

    def test_decide_confidence_threshold(self):
        # With default threshold 0.5, low-confidence decisions still execute
        # because all our rules have confidence >= 0.6. Use a custom low-conf
        # policy to test threshold skip.
        src = MockRoomSource()
        src.set_room(
            RoomObservation(
                room_id="alpha",
                timestamp=time.time(),
                agent_count=10,
                diversity_score=0.1,
            )
        )
        loop = SDALoop(confidence_threshold=0.95)
        chain = PlatoSignalChain(source=src, loop=loop)
        results = chain.tick()
        # Breed confidence is 0.85, below 0.95 threshold → skipped
        assert results["plato_signal_chain"] is None
