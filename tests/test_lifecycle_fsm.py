"""Tests for AgentLifecycleFSM.

Covers the canonical transition graph, guard methods, and history.
"""

from __future__ import annotations

import time

import pytest

from swarm.lifecycle_fsm import (
    AgentLifecycleFSM,
    LifecycleState,
    LifecycleTransitionError,
    TransitionRecord,
)


class TestValidTransitions:
    """Every valid arc in the canonical graph."""

    def test_egg_to_compete(self):
        fsm = AgentLifecycleFSM(agent_id=1)
        assert fsm.get_state() == LifecycleState.EGG
        fsm.transition(LifecycleState.COMPETE, reason="init")
        assert fsm.get_state() == LifecycleState.COMPETE

    def test_compete_to_survive(self):
        fsm = AgentLifecycleFSM(agent_id=2)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SURVIVE, reason="tournament_win")
        assert fsm.get_state() == LifecycleState.SURVIVE

    def test_compete_to_sunset(self):
        fsm = AgentLifecycleFSM(agent_id=3)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SUNSET, reason="tournament_loss")
        assert fsm.get_state() == LifecycleState.SUNSET

    def test_survive_to_breed(self):
        fsm = AgentLifecycleFSM(agent_id=4)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SURVIVE)
        fsm.transition(LifecycleState.BREED, reason="selected")
        assert fsm.get_state() == LifecycleState.BREED

    def test_survive_to_compete(self):
        fsm = AgentLifecycleFSM(agent_id=5)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SURVIVE)
        fsm.transition(LifecycleState.COMPETE, reason="re_enter")
        assert fsm.get_state() == LifecycleState.COMPETE

    def test_breed_to_egg(self):
        fsm = AgentLifecycleFSM(agent_id=6)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SURVIVE)
        fsm.transition(LifecycleState.BREED)
        fsm.transition(LifecycleState.EGG, reason="child_spawned")
        assert fsm.get_state() == LifecycleState.EGG

    def test_sunset_to_archive(self):
        fsm = AgentLifecycleFSM(agent_id=7)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SUNSET)
        fsm.transition(LifecycleState.ARCHIVE, reason="final_cleanup")
        assert fsm.get_state() == LifecycleState.ARCHIVE


class TestInvalidTransitions:
    """Invalid arcs must raise in strict mode."""

    def test_egg_to_sunset_raises(self):
        fsm = AgentLifecycleFSM(agent_id=10)
        with pytest.raises(LifecycleTransitionError) as exc:
            fsm.transition(LifecycleState.SUNSET)
        assert exc.value.from_state == LifecycleState.EGG
        assert exc.value.to_state == LifecycleState.SUNSET

    def test_egg_to_archive_raises(self):
        fsm = AgentLifecycleFSM(agent_id=11)
        with pytest.raises(LifecycleTransitionError):
            fsm.transition(LifecycleState.ARCHIVE)

    def test_compete_to_breed_raises(self):
        fsm = AgentLifecycleFSM(agent_id=12)
        fsm.transition(LifecycleState.COMPETE)
        with pytest.raises(LifecycleTransitionError):
            fsm.transition(LifecycleState.BREED)

    def test_archive_no_outgoing(self):
        fsm = AgentLifecycleFSM(agent_id=13)
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SUNSET)
        fsm.transition(LifecycleState.ARCHIVE)
        with pytest.raises(LifecycleTransitionError):
            fsm.transition(LifecycleState.EGG)

    def test_non_strict_ignores_invalid(self):
        fsm = AgentLifecycleFSM(agent_id=14, strict=False)
        ok = fsm.transition(LifecycleState.SUNSET)
        assert ok is False
        assert fsm.get_state() == LifecycleState.EGG


class TestFullLifecycleChain:
    """End-to-end: COMPETE → SURVIVE → BREED → EGG."""

    def test_chain(self):
        fsm = AgentLifecycleFSM(agent_id=20)
        fsm.transition(LifecycleState.COMPETE, reason="init")
        fsm.transition(LifecycleState.SURVIVE, reason="win")
        fsm.transition(LifecycleState.BREED, reason="mate")
        fsm.transition(LifecycleState.EGG, reason="child")
        assert fsm.get_state() == LifecycleState.EGG
        history = fsm.get_history()
        # init(EGG→EGG) + COMPETE + SURVIVE + BREED + EGG = 5 records
        assert len(history) == 5
        arcs = [(h.from_state, h.to_state) for h in history]
        assert arcs[0] == (LifecycleState.EGG, LifecycleState.EGG)
        assert arcs[1] == (LifecycleState.EGG, LifecycleState.COMPETE)
        assert arcs[2] == (LifecycleState.COMPETE, LifecycleState.SURVIVE)
        assert arcs[3] == (LifecycleState.SURVIVE, LifecycleState.BREED)
        assert arcs[4] == (LifecycleState.BREED, LifecycleState.EGG)


class TestGuardMethods:
    """can_breed and can_compete predicates."""

    def test_can_breed_only_survive(self):
        fsm = AgentLifecycleFSM(agent_id=30)
        assert fsm.can_breed() is False  # EGG
        fsm.transition(LifecycleState.COMPETE)
        assert fsm.can_breed() is False  # COMPETE
        fsm.transition(LifecycleState.SURVIVE)
        assert fsm.can_breed() is True   # SURVIVE
        fsm.transition(LifecycleState.BREED)
        assert fsm.can_breed() is False  # BREED

    def test_can_compete_egg_and_survive(self):
        fsm = AgentLifecycleFSM(agent_id=31)
        assert fsm.can_compete() is True   # EGG
        fsm.transition(LifecycleState.COMPETE)
        assert fsm.can_compete() is False  # COMPETE
        fsm.transition(LifecycleState.SURVIVE)
        assert fsm.can_compete() is True   # SURVIVE
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SUNSET)
        assert fsm.can_compete() is False  # SUNSET

    def test_is_terminal(self):
        fsm = AgentLifecycleFSM(agent_id=32)
        assert fsm.is_terminal() is False
        fsm.transition(LifecycleState.COMPETE)
        fsm.transition(LifecycleState.SUNSET)
        fsm.transition(LifecycleState.ARCHIVE)
        assert fsm.is_terminal() is True


class TestHistory:
    """Transition history is complete and immutable."""

    def test_history_includes_init(self):
        fsm = AgentLifecycleFSM(agent_id=40)
        hist = fsm.get_history()
        assert len(hist) == 1
        assert hist[0].from_state == LifecycleState.EGG
        assert hist[0].to_state == LifecycleState.EGG
        assert hist[0].reason == "init"

    def test_history_timestamps_monotonic(self):
        fsm = AgentLifecycleFSM(agent_id=41)
        time.sleep(0.01)
        fsm.transition(LifecycleState.COMPETE)
        time.sleep(0.01)
        fsm.transition(LifecycleState.SURVIVE)
        hist = fsm.get_history()
        ts = [h.timestamp for h in hist]
        assert ts == sorted(ts)

    def test_history_copy_is_shallow_safe(self):
        fsm = AgentLifecycleFSM(agent_id=42)
        fsm.transition(LifecycleState.COMPETE)
        h1 = fsm.get_history()
        h2 = fsm.get_history()
        assert h1 is not h2
        assert h1 == h2

    def test_last_transition(self):
        fsm = AgentLifecycleFSM(agent_id=43)
        fsm.transition(LifecycleState.COMPETE)
        last = fsm.last_transition()
        assert last.to_state == LifecycleState.COMPETE

    def test_idempotent_transition_no_history_dup(self):
        fsm = AgentLifecycleFSM(agent_id=44)
        fsm.transition(LifecycleState.COMPETE)
        before = len(fsm.get_history())
        fsm.transition(LifecycleState.COMPETE)
        after = len(fsm.get_history())
        assert before == after


class TestRepr:
    def test_repr(self):
        fsm = AgentLifecycleFSM(agent_id=50)
        assert "agent_id=50" in repr(fsm)
        assert "EGG" in repr(fsm)
