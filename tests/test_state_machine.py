"""Tests for state_machine.py — Finite state machine.

Run: python3 -m pytest tests/test_state_machine.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.state_machine import StateMachine, TransitionNotAllowed


class TestStateMachine:
    def test_create(self):
        fsm = StateMachine(initial="idle")
        assert fsm.current == "idle"

    def test_add_state(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        assert fsm.stats()["states"] == 2  # idle + running

    def test_add_transition(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        assert len(fsm.transition_table()) == 1

    def test_trigger(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        result = fsm.trigger("start")
        assert result == "running"
        assert fsm.current == "running"

    def test_trigger_no_transition(self):
        fsm = StateMachine(initial="idle")
        with pytest.raises(TransitionNotAllowed):
            fsm.trigger("stop")

    def test_guard(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start", guard=lambda can_run=True: can_run)
        assert fsm.trigger("start", can_run=True) == "running"

    def test_guard_rejected(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start", guard=lambda can_run=True: can_run)
        with pytest.raises(TransitionNotAllowed):
            fsm.trigger("start", can_run=False)

    def test_entry_action(self):
        entered = []
        fsm = StateMachine(initial="idle")
        fsm.add_state("running", on_entry=lambda **kw: entered.append("running"))
        fsm.add_transition("idle", "running", trigger="start")
        fsm.trigger("start")
        assert entered == ["running"]

    def test_exit_action(self):
        exited = []
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle", on_exit=lambda **kw: exited.append("idle"))
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        fsm.trigger("start")
        assert exited == ["idle"]

    def test_transition_action(self):
        actions = []
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start", action=lambda **kw: actions.append("transit"))
        fsm.trigger("start")
        assert actions == ["transit"]

    def test_history(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_state("stopped")
        fsm.add_transition("idle", "running", trigger="start")
        fsm.add_transition("running", "stopped", trigger="stop")
        fsm.trigger("start")
        fsm.trigger("stop")
        assert fsm.history == ["idle", "running", "stopped"]

    def test_can(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        assert fsm.can("start") is True
        assert fsm.can("stop") is False

    def test_can_with_guard(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start", guard=lambda ready=True: ready)
        assert fsm.can("start", ready=True) is True
        assert fsm.can("start", ready=False) is False

    def test_available_triggers(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_state("stopped")
        fsm.add_transition("idle", "running", trigger="start")
        fsm.add_transition("idle", "stopped", trigger="kill")
        assert sorted(fsm.available_triggers()) == ["kill", "start"]

    def test_reset(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        fsm.trigger("start")
        assert fsm.current == "running"
        fsm.reset()
        assert fsm.current == "idle"
        assert fsm.history == ["idle"]

    def test_stats(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", trigger="start")
        fsm.trigger("start")
        stats = fsm.stats()
        assert stats["states"] == 2
        assert stats["transitions"] == 1
        assert stats["current"] == "running"
        assert stats["transition_count"] == 1

    def test_repr(self):
        fsm = StateMachine(initial="idle")
        assert "StateMachine" in repr(fsm)
