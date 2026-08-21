"""Tests for workflow_engine.py — State machine workflow engine.

Run: python3 -m pytest tests/test_workflow_engine.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.workflow_engine import WorkflowEngine


class TestWorkflowEngine:
    def test_create(self):
        engine = WorkflowEngine("test")
        assert engine.stats()["states"] == 0

    def test_add_state(self):
        engine = WorkflowEngine()
        engine.add_state("pending")
        assert engine.stats()["states"] == 1

    def test_add_transition(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        assert engine.stats()["transitions"] == 1

    def test_start(self):
        engine = WorkflowEngine()
        engine.add_state("pending")
        engine.start("pending", {"budget": 100})
        assert engine.current() == "pending"

    def test_transition(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        engine.start("pending")
        assert engine.transition("approved") is True
        assert engine.current() == "approved"

    def test_transition_with_guard(self):
        engine = WorkflowEngine()
        engine.add_transition(
            "pending", "approved", guard=lambda ctx: ctx.get("budget", 0) > 0
        )
        engine.start("pending", {"budget": 0})
        assert engine.transition("approved") is False
        engine._context["budget"] = 100
        assert engine.transition("approved") is True

    def test_transition_with_action(self):
        engine = WorkflowEngine()
        called = [False]
        engine.add_transition(
            "pending", "approved", action=lambda ctx: called.__setitem__(0, True)
        )
        engine.start("pending")
        engine.transition("approved")
        assert called[0] is True

    def test_invalid_transition(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        engine.start("pending")
        assert engine.transition("rejected") is False

    def test_can_transition(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        engine.start("pending")
        assert engine.can_transition("approved") is True
        assert engine.can_transition("rejected") is False

    def test_history(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        engine.add_transition("approved", "deployed")
        engine.start("pending")
        engine.transition("approved")
        engine.transition("deployed")
        assert engine.history() == ["pending", "approved", "deployed"]

    def test_available_transitions(self):
        engine = WorkflowEngine()
        engine.add_transition("pending", "approved")
        engine.add_transition("pending", "rejected")
        engine.start("pending")
        assert sorted(engine.available_transitions()) == ["approved", "rejected"]

    def test_repr(self):
        engine = WorkflowEngine("test")
        assert "WorkflowEngine" in repr(engine)
