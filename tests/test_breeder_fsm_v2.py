"""Tests for BreederFSMV2 lifecycle state machine."""
from __future__ import annotations

import threading
import time

import pytest

from swarm.breeder_fsm_v2 import BreederFSMV2, LifecycleState, TransitionError


class TestLifecycleTransitions:
    def test_egg_to_compete(self):
        fsm = BreederFSMV2("agent-1")
        assert fsm.current_state == LifecycleState.EGG
        fsm.transition_to(LifecycleState.COMPETE)
        assert fsm.current_state == LifecycleState.COMPETE

    def test_compete_to_survive(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.COMPETE)
        fsm.transition_to(LifecycleState.SURVIVE)
        assert fsm.current_state == LifecycleState.SURVIVE

    def test_survive_to_breed(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.SURVIVE)
        fsm.transition_to(LifecycleState.BREED)
        assert fsm.current_state == LifecycleState.BREED

    def test_breed_to_egg(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.BREED)
        fsm.transition_to(LifecycleState.EGG)
        assert fsm.current_state == LifecycleState.EGG

    def test_sunset_to_archive(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.SUNSET)
        fsm.transition_to(LifecycleState.ARCHIVE)
        assert fsm.current_state == LifecycleState.ARCHIVE

    def test_invalid_transition_raises(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.EGG)
        with pytest.raises(TransitionError):
            fsm.transition_to(LifecycleState.BREED)

    def test_archive_is_terminal(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.ARCHIVE)
        assert fsm.get_status()["is_terminal"]
        with pytest.raises(TransitionError):
            fsm.transition_to(LifecycleState.EGG)


class TestConvenienceMethods:
    def test_incubate(self):
        fsm = BreederFSMV2("agent-1")
        fsm.incubate()
        assert fsm.current_state == LifecycleState.COMPETE

    def test_win(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.COMPETE)
        fsm.win()
        assert fsm.current_state == LifecycleState.SURVIVE

    def test_breed(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.SURVIVE)
        fsm.breed()
        assert fsm.current_state == LifecycleState.BREED

    def test_spawn_child(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.BREED)
        fsm.spawn_child()
        assert fsm.current_state == LifecycleState.EGG

    def test_sunset_from_egg(self):
        fsm = BreederFSMV2("agent-1")
        fsm.sunset()
        assert fsm.current_state == LifecycleState.SUNSET

    def test_sunset_from_compete(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.COMPETE)
        fsm.sunset()
        assert fsm.current_state == LifecycleState.SUNSET

    def test_archive(self):
        fsm = BreederFSMV2("agent-1", initial_state=LifecycleState.SUNSET)
        fsm.archive()
        assert fsm.current_state == LifecycleState.ARCHIVE


class TestGuards:
    def test_entry_guard_blocks(self):
        calls = 0
        def guard():
            nonlocal calls
            calls += 1
            return False

        fsm = BreederFSMV2(
            "agent-1",
            state_configs={LifecycleState.COMPETE: StateConfig(entry_guard=guard)},
        )
        with pytest.raises(TransitionError):
            fsm.incubate()
        assert calls == 1

    def test_exit_guard_blocks(self):
        calls = 0
        def guard():
            nonlocal calls
            calls += 1
            return False

        fsm = BreederFSMV2(
            "agent-1",
            state_configs={LifecycleState.EGG: StateConfig(exit_guard=guard)},
        )
        with pytest.raises(TransitionError):
            fsm.incubate()
        assert calls == 1


class TestTimeout:
    def test_timeout_triggers_auto_transition(self):
        fsm = BreederFSMV2(
            "agent-1",
            state_configs={
                LifecycleState.EGG: StateConfig(
                    timeout_sec=0.01,
                    auto_transition=LifecycleState.COMPETE,
                ),
            },
        )
        time.sleep(0.02)
        result = fsm.check_timeout()
        assert result is not None
        assert fsm.current_state == LifecycleState.COMPETE

    def test_no_timeout_no_transition(self):
        fsm = BreederFSMV2("agent-1")
        result = fsm.check_timeout()
        assert result is None
        assert fsm.current_state == LifecycleState.EGG


class TestHistory:
    def test_history_records_transitions(self):
        fsm = BreederFSMV2("agent-1")
        fsm.incubate()
        fsm.win()
        history = fsm.get_history()
        assert len(history) == 2
        assert history[0].from_state == LifecycleState.EGG
        assert history[0].to_state == LifecycleState.COMPETE
        assert history[1].from_state == LifecycleState.COMPETE
        assert history[1].to_state == LifecycleState.SURVIVE

    def test_history_includes_reason(self):
        fsm = BreederFSMV2("agent-1")
        fsm.transition_to(LifecycleState.COMPETE, reason="test")
        history = fsm.get_history()
        assert history[0].reason == "test"


class TestThreadSafety:
    def test_concurrent_reads(self):
        fsm = BreederFSMV2("agent-1")
        states = []
        def reader():
            for _ in range(100):
                states.append(fsm.current_state)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(s == LifecycleState.EGG for s in states)

    def test_concurrent_transitions(self):
        fsm = BreederFSMV2("agent-1")
        errors = []
        def transitioner():
            try:
                fsm.incubate()
            except TransitionError:
                errors.append("error")

        threads = [threading.Thread(target=transitioner) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed, rest should error (already transitioned)
        assert fsm.current_state == LifecycleState.COMPETE
        assert len(errors) == 4


class TestStatus:
    def test_status_fields(self):
        fsm = BreederFSMV2("agent-1")
        status = fsm.get_status()
        assert status["agent_id"] == "agent-1"
        assert status["current_state"] == "EGG"
        assert status["n_transitions"] == 0
        assert not status["is_terminal"]

    def test_status_after_transitions(self):
        fsm = BreederFSMV2("agent-1")
        fsm.incubate()
        fsm.win()
        status = fsm.get_status()
        assert status["n_transitions"] == 2
        assert status["current_state"] == "SURVIVE"


class TestCanTransition:
    def test_can_transition_valid(self):
        fsm = BreederFSMV2("agent-1")
        assert fsm.can_transition_to(LifecycleState.COMPETE)
        assert fsm.can_transition_to(LifecycleState.SUNSET)

    def test_can_transition_invalid(self):
        fsm = BreederFSMV2("agent-1")
        assert not fsm.can_transition_to(LifecycleState.BREED)
        assert not fsm.can_transition_to(LifecycleState.ARCHIVE)


# Need to import StateConfig for guard tests
from swarm.breeder_fsm_v2 import StateConfig
