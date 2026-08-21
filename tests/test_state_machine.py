"""Tests for state_machine.py — FSM with transitions, guards, callbacks.

Run: python3 -m pytest tests/test_state_machine.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.state_machine import StateMachine


class TestStateMachine:
    def test_create(self):
        fsm = StateMachine(initial="idle")
        assert fsm.state() == "idle"
        assert fsm.stats()["states"] == 0

    def test_add_state(self):
        fsm = StateMachine()
        fsm.add_state("idle")
        assert "idle" in fsm.states()

    def test_add_state_with_callbacks(self):
        fsm = StateMachine()
        entered = []
        exited = []
        fsm.add_state("idle", on_exit=lambda: exited.append(True))
        fsm.add_state("running", on_enter=lambda: entered.append(True))
        fsm.add_transition("idle", "running", event="start")
        fsm.trigger("start")
        assert entered == [True]
        assert exited == [True]

    def test_add_transition(self):
        fsm = StateMachine()
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start")
        assert "start" in fsm.transitions_from("idle")

    def test_trigger(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start")
        assert fsm.trigger("start") is True
        assert fsm.state() == "running"

    def test_trigger_no_transition(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start")
        assert fsm.trigger("stop") is False
        assert fsm.state() == "idle"

    def test_guard(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start", guard=lambda: False)
        assert fsm.trigger("start") is False
        assert fsm.state() == "idle"

    def test_guard_pass(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start", guard=lambda: True)
        assert fsm.trigger("start") is True
        assert fsm.state() == "running"

    def test_multiple_transitions(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_state("paused")
        fsm.add_transition("idle", "running", event="start")
        fsm.add_transition("running", "paused", event="pause")
        fsm.add_transition("paused", "running", event="resume")
        fsm.trigger("start")
        fsm.trigger("pause")
        assert fsm.state() == "paused"
        fsm.trigger("resume")
        assert fsm.state() == "running"

    def test_set_state(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle", on_exit=lambda: None)
        fsm.add_state("running", on_enter=lambda: None)
        fsm.set_state("running")
        assert fsm.state() == "running"

    def test_can_trigger(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start", guard=lambda: False)
        assert fsm.can_trigger("start") is False
        fsm.add_transition("idle", "running", event="go")
        assert fsm.can_trigger("go") is True

    def test_history(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start")
        fsm.trigger("start")
        history = fsm.history()
        assert len(history) == 1
        assert history[0]["event"] == "start"

    def test_stats(self):
        fsm = StateMachine(initial="idle")
        fsm.add_state("idle")
        fsm.add_state("running")
        fsm.add_transition("idle", "running", event="start")
        fsm.trigger("start")
        stats = fsm.stats()
        assert stats["state"] == "running"
        assert stats["states"] == 2
        assert stats["transitions"] == 1
        assert stats["transition_count"] == 1

    def test_repr(self):
        fsm = StateMachine()
        assert "StateMachine" in repr(fsm)
