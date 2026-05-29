"""Workflow engine with state machine transitions.

Defines workflows as directed graphs of states with transitions,
validators, and actions. Used for fleet job lifecycle management,
approval flows, and multi-step processes.

Usage:
    engine = WorkflowEngine()
    engine.add_state("pending")
    engine.add_state("approved")
    engine.add_transition("pending", "approved", guard=lambda ctx: ctx["budget"] > 0)
    engine.start("pending", {"budget": 100})
    engine.transition("approved")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class Transition:
    """A state transition rule."""

    from_state: str
    to_state: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None


class WorkflowEngine:
    """
    State machine workflow engine.

    :param name: Workflow name.
    """

    def __init__(self, name: str = "workflow"):
        self.name = name
        self._states: Set[str] = set()
        self._transitions: Dict[str, List[Transition]] = {}
        self._current: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._history: List[str] = []

    # ------------------------------------------------------------------
    # Definition
    # ------------------------------------------------------------------

    def add_state(self, state: str) -> None:
        """Register a valid state."""
        self._states.add(state)

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        guard: Optional[Callable[[Dict[str, Any]], bool]] = None,
        action: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Register a transition between states."""
        self._states.add(from_state)
        self._states.add(to_state)
        key = from_state
        if key not in self._transitions:
            self._transitions[key] = []
        self._transitions[key].append(Transition(from_state, to_state, guard, action))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start(self, state: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Start the workflow in a given state."""
        self._current = state
        self._context = context or {}
        self._history = [state]

    def transition(self, to_state: str) -> bool:
        """
        Attempt to transition to a new state.

        :returns: True if transition succeeded.
        """
        if self._current is None:
            raise RuntimeError("Workflow not started")
        candidates = self._transitions.get(self._current, [])
        for t in candidates:
            if t.to_state == to_state:
                if t.guard and not t.guard(self._context):
                    return False
                if t.action:
                    t.action(self._context)
                self._current = to_state
                self._history.append(to_state)
                return True
        return False

    def can_transition(self, to_state: str) -> bool:
        """Check if a transition is possible without executing it."""
        if self._current is None:
            return False
        candidates = self._transitions.get(self._current, [])
        for t in candidates:
            if t.to_state == to_state:
                if t.guard and not t.guard(self._context):
                    return False
                return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def current(self) -> Optional[str]:
        return self._current

    def history(self) -> List[str]:
        return list(self._history)

    def available_transitions(self) -> List[str]:
        """Get list of states reachable from current state."""
        if self._current is None:
            return []
        candidates = self._transitions.get(self._current, [])
        return [t.to_state for t in candidates if not t.guard or t.guard(self._context)]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "states": len(self._states),
            "transitions": sum(len(v) for v in self._transitions.values()),
            "current": self._current,
            "history_length": len(self._history),
        }

    def __repr__(self) -> str:
        return f"<WorkflowEngine {self.name} states={len(self._states)} current={self._current}>"
