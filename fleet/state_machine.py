"""state_machine.py — Finite state machine for agent and service lifecycle.

Provides:
1. States, transitions, and guards
2. Entry/exit actions per state
3. Transition triggers with arguments
4. State history (previous states)
5. Visualization-friendly transition table

Usage:
    fsm = StateMachine(initial="idle")
    fsm.add_state("idle", on_entry=notify_idle)
    fsm.add_state("running", on_entry=start_agent)
    fsm.add_transition("idle", "running", trigger="start", guard=can_start)
    fsm.trigger("start", agent_id="a1")
    assert fsm.current == "running"
"""
from __future__ import annotations

__all__ = [
    "StateMachine",
    "State",
    "Transition",
    "TransitionNotAllowed",
]

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TransitionNotAllowed(Exception):
    """Raised when a transition is not defined or guard rejects it."""


@dataclass
class State:
    """A state with optional entry/exit actions."""
    name: str
    on_entry: Callable[..., Any] | None = None
    on_exit: Callable[..., Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """A transition between states."""
    source: str
    target: str
    trigger: str
    guard: Callable[..., bool] | None = None
    action: Callable[..., Any] | None = None


class StateMachine:
    """Finite state machine for lifecycle management."""

    def __init__(self, initial: str) -> None:
        self._initial = initial
        self._current = initial
        self._states: dict[str, State] = {initial: State(name=initial)}
        self._transitions: list[Transition] = []
        self._history: list[str] = [initial]
        self._transition_count = 0

    def add_state(
        self,
        name: str,
        on_entry: Callable[..., Any] | None = None,
        on_exit: Callable[..., Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add a state."""
        self._states[name] = State(
            name=name,
            on_entry=on_entry,
            on_exit=on_exit,
            data=data or {},
        )

    def add_transition(
        self,
        source: str,
        target: str,
        trigger: str,
        guard: Callable[..., bool] | None = None,
        action: Callable[..., Any] | None = None,
    ) -> None:
        """Add a transition."""
        self._transitions.append(Transition(
            source=source,
            target=target,
            trigger=trigger,
            guard=guard,
            action=action,
        ))

    def trigger(self, trigger_name: str, **kwargs: Any) -> str:
        """Trigger a transition."""
        # Find matching transition
        candidates = [
            t for t in self._transitions
            if t.source == self._current and t.trigger == trigger_name
        ]
        if not candidates:
            raise TransitionNotAllowed(
                f"No transition from '{self._current}' on trigger '{trigger_name}'"
            )

        for t in candidates:
            if t.guard is not None and not t.guard(**kwargs):
                continue

            # Execute exit action
            old_state = self._states.get(self._current)
            if old_state and old_state.on_exit:
                try:
                    old_state.on_exit(**kwargs)
                except Exception as e:
                    logger.warning(f"Exit action error for {self._current}: {e}")

            # Execute transition action
            if t.action:
                try:
                    t.action(**kwargs)
                except Exception as e:
                    logger.warning(f"Transition action error: {e}")

            self._current = t.target
            self._history.append(t.target)
            self._transition_count += 1

            # Execute entry action
            new_state = self._states.get(t.target)
            if new_state and new_state.on_entry:
                try:
                    new_state.on_entry(**kwargs)
                except Exception as e:
                    logger.warning(f"Entry action error for {t.target}: {e}")

            return t.target

        raise TransitionNotAllowed(
            f"All guards rejected transition from '{self._current}' on '{trigger_name}'"
        )

    def can(self, trigger_name: str, **kwargs: Any) -> bool:
        """Check if a trigger is currently valid (at least one transition + guard passes)."""
        candidates = [
            t for t in self._transitions
            if t.source == self._current and t.trigger == trigger_name
        ]
        if not candidates:
            return False
        return any(t.guard is None or t.guard(**kwargs) for t in candidates)

    @property
    def current(self) -> str:
        return self._current

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def available_triggers(self) -> list[str]:
        """List triggers valid from current state."""
        return sorted({t.trigger for t in self._transitions if t.source == self._current})

    def transition_table(self) -> list[dict[str, Any]]:
        """Get all transitions as dicts."""
        return [
            {
                "source": t.source,
                "target": t.target,
                "trigger": t.trigger,
                "has_guard": t.guard is not None,
            }
            for t in self._transitions
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "states": len(self._states),
            "transitions": len(self._transitions),
            "current": self._current,
            "transition_count": self._transition_count,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self._current = self._initial
        self._history = [self._initial]
        self._transition_count = 0

    def __repr__(self) -> str:
        return f"StateMachine(current={self._current}, states={len(self._states)})"
