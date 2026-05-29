"""FSM with transitions, guards, and entry/exit callbacks.

Implements a finite state machine with named states, guarded transitions,
entry/exit callbacks, and event-driven state changes. Used for fleet
entity lifecycle, breeding daemon states, and service orchestration.

Usage:
    fsm = StateMachine(initial="idle")
    fsm.add_state("idle", on_enter=lambda: print("entered idle"))
    fsm.add_transition("idle", "running", event="start", guard=lambda: True)
    fsm.trigger("start")
    assert fsm.state() == "running"
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class StateMachine:
    """
    Finite state machine with guarded transitions.
    """

    def __init__(self, initial: str = "idle"):
        self._state = initial
        self._states: Dict[str, Dict[str, Any]] = {}
        self._transitions: Dict[str, List[Dict[str, Any]]] = {}  # from_state -> transitions
        self._history: List[Dict[str, Any]] = []
        self._transition_count = 0

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def add_state(
        self,
        name: str,
        on_enter: Optional[Callable[[], Any]] = None,
        on_exit: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Register a state with optional callbacks.

        :param name: State identifier.
        :param on_enter: Called when entering state.
        :param on_exit: Called when leaving state.
        """
        self._states[name] = {
            "on_enter": on_enter,
            "on_exit": on_exit,
        }

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        event: str,
        guard: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a state transition.

        :param from_state: Source state.
        :param to_state: Destination state.
        :param event: Trigger event name.
        :param guard: Optional guard function (must return True).
        """
        if from_state not in self._transitions:
            self._transitions[from_state] = []
        self._transitions[from_state].append({
            "to_state": to_state,
            "event": event,
            "guard": guard,
        })

    # ------------------------------------------------------------------
    # Triggering
    # ------------------------------------------------------------------

    def trigger(self, event: str) -> bool:
        """
        Trigger an event to attempt a transition.

        :param event: Event name.
        :returns: True if transition succeeded.
        """
        candidates = self._transitions.get(self._state, [])
        for t in candidates:
            if t["event"] == event:
                guard = t.get("guard")
                if guard is not None and not guard():
                    continue
                # Execute transition
                self._leave_state()
                self._state = t["to_state"]
                self._enter_state()
                self._transition_count += 1
                self._history.append({
                    "from": self._state,
                    "to": t["to_state"],
                    "event": event,
                })
                return True
        return False

    def _leave_state(self) -> None:
        callbacks = self._states.get(self._state, {})
        on_exit = callbacks.get("on_exit")
        if on_exit:
            on_exit()

    def _enter_state(self) -> None:
        callbacks = self._states.get(self._state, {})
        on_enter = callbacks.get("on_enter")
        if on_enter:
            on_enter()

    def set_state(self, state: str) -> None:
        """Forcefully set state (no transition checking)."""
        self._leave_state()
        self._state = state
        self._enter_state()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def state(self) -> str:
        return self._state

    def states(self) -> List[str]:
        return list(self._states.keys())

    def transitions_from(self, state: str) -> List[str]:
        """List events that can trigger from a state."""
        return [t["event"] for t in self._transitions.get(state, [])]

    def can_trigger(self, event: str) -> bool:
        """Check if event can trigger from current state."""
        candidates = self._transitions.get(self._state, [])
        for t in candidates:
            if t["event"] == event:
                guard = t.get("guard")
                if guard is None or guard():
                    return True
        return False

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "states": len(self._states),
            "transitions": sum(len(t) for t in self._transitions.values()),
            "transition_count": self._transition_count,
            "history_size": len(self._history),
        }

    def __repr__(self) -> str:
        return f"<StateMachine state={self._state} transitions={self.stats()['transitions']}>"
