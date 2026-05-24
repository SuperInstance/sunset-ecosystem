"""Breeder FSM V2 — Complete lifecycle state machine for agent breeding.

Provides `BreederFSMV2` which manages the canonical 6-state lifecycle:
    EGG → COMPETE → SURVIVE → BREED → SUNSET → ARCHIVE

Each state has entry/exit guards, timeout handling, and auto-transition
triggers. The FSM is deterministic and thread-safe.

Usage::

    from swarm.breeder_fsm_v2 import BreederFSMV2, LifecycleState
    fsm = BreederFSMV2(agent_id="agent-1")
    fsm.transition_to(LifecycleState.COMPETE)
    assert fsm.current_state == LifecycleState.COMPETE
"""
from __future__ import annotations

__all__ = ["BreederFSMV2", "LifecycleState", "TransitionError"]

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Re-export canonical LifecycleState from lifecycle_fsm to avoid dual enum problem
from swarm.lifecycle_fsm import LifecycleState

log = logging.getLogger(__name__)


class TransitionError(ValueError):
    """Raised when an invalid state transition is attempted."""
    pass


@dataclass
class StateConfig:
    """Configuration for a lifecycle state."""
    timeout_sec: float | None = None
    auto_transition: LifecycleState | None = None
    entry_guard: Callable[[], bool] | None = None
    exit_guard: Callable[[], bool] | None = None


@dataclass
class TransitionRecord:
    """Immutable record of a state transition."""
    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: float
    reason: str


class BreederFSMV2:
    """Thread-safe finite state machine for agent lifecycle management.

    Implements the canonical 6-state breeding lifecycle with:
      - Configurable timeouts per state
      - Auto-transition on timeout
      - Entry/exit guards
      - Full transition history
    """

    # Valid transitions: source → set of destinations
    VALID_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
        LifecycleState.EGG: {LifecycleState.COMPETE, LifecycleState.SUNSET},
        LifecycleState.COMPETE: {LifecycleState.SURVIVE, LifecycleState.SUNSET},
        LifecycleState.SURVIVE: {LifecycleState.BREED, LifecycleState.SUNSET},
        LifecycleState.BREED: {LifecycleState.EGG, LifecycleState.SUNSET},
        LifecycleState.SUNSET: {LifecycleState.ARCHIVE},
        LifecycleState.ARCHIVE: set(),  # terminal
    }

    def __init__(
        self,
        agent_id: str,
        initial_state: LifecycleState = LifecycleState.EGG,
        state_configs: dict[LifecycleState, StateConfig] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self._state = initial_state
        self._lock = threading.RLock()
        self._history: list[TransitionRecord] = []
        self._state_entry_time: float = time.time()

        # Default state configs
        self._configs: dict[LifecycleState, StateConfig] = {
            LifecycleState.EGG: StateConfig(timeout_sec=30.0, auto_transition=LifecycleState.COMPETE),
            LifecycleState.COMPETE: StateConfig(timeout_sec=300.0),
            LifecycleState.SURVIVE: StateConfig(timeout_sec=60.0),
            LifecycleState.BREED: StateConfig(timeout_sec=30.0),
            LifecycleState.SUNSET: StateConfig(timeout_sec=10.0, auto_transition=LifecycleState.ARCHIVE),
            LifecycleState.ARCHIVE: StateConfig(),
        }
        if state_configs:
            self._configs.update(state_configs)

    # ── Public API ─────────────────────────────────────────────────

    @property
    def current_state(self) -> LifecycleState:
        """Current lifecycle state (thread-safe read)."""
        with self._lock:
            return self._state

    def transition_to(
        self,
        new_state: LifecycleState,
        reason: str = "manual",
    ) -> TransitionRecord:
        """Attempt a state transition.

        Raises `TransitionError` if the transition is invalid or guards fail.
        """
        with self._lock:
            current = self._state

            # Validate transition
            if new_state not in self.VALID_TRANSITIONS.get(current, set()):
                raise TransitionError(
                    f"Invalid transition: {current.name} → {new_state.name}"
                )

            # Check exit guard
            config = self._configs.get(current, StateConfig())
            if config.exit_guard and not config.exit_guard():
                raise TransitionError(
                    f"Exit guard failed for {current.name}"
                )

            # Check entry guard
            new_config = self._configs.get(new_state, StateConfig())
            if new_config.entry_guard and not new_config.entry_guard():
                raise TransitionError(
                    f"Entry guard failed for {new_state.name}"
                )

            # Execute transition
            self._state = new_state
            self._state_entry_time = time.time()
            record = TransitionRecord(
                from_state=current,
                to_state=new_state,
                timestamp=self._state_entry_time,
                reason=reason,
            )
            self._history.append(record)
            log.info(
                "Agent %s: %s → %s (%s)",
                self.agent_id, current.name, new_state.name, reason,
            )
            return record

    def check_timeout(self) -> TransitionRecord | None:
        """Check if current state has timed out and auto-transition.

        Returns the transition record if auto-transition occurred,
        None otherwise.
        """
        with self._lock:
            current = self._state
            config = self._configs.get(current, StateConfig())

            if config.timeout_sec is None or config.auto_transition is None:
                return None

            elapsed = time.time() - self._state_entry_time
            if elapsed < config.timeout_sec:
                return None

            # Auto-transition
            return self.transition_to(
                config.auto_transition,
                reason=f"timeout after {elapsed:.1f}s",
            )

    def get_history(self) -> list[TransitionRecord]:
        """Return full transition history."""
        with self._lock:
            return list(self._history)

    def get_time_in_state(self) -> float:
        """Return seconds spent in current state."""
        with self._lock:
            return time.time() - self._state_entry_time

    def can_transition_to(self, state: LifecycleState) -> bool:
        """Check if a transition to `state` is currently valid."""
        with self._lock:
            return state in self.VALID_TRANSITIONS.get(self._state, set())

    def get_status(self) -> dict[str, Any]:
        """Return current FSM status."""
        with self._lock:
            config = self._configs.get(self._state, StateConfig())
            return {
                "agent_id": self.agent_id,
                "current_state": self._state.name,
                "time_in_state_sec": self.get_time_in_state(),
                "timeout_sec": config.timeout_sec,
                "auto_transition": config.auto_transition.name if config.auto_transition else None,
                "n_transitions": len(self._history),
                "is_terminal": self._state == LifecycleState.ARCHIVE,
            }

    # ── Lifecycle convenience methods ────────────────────────────────

    def incubate(self) -> TransitionRecord:
        """Transition EGG → COMPETE."""
        return self.transition_to(LifecycleState.COMPETE, reason="incubation_complete")

    def win(self) -> TransitionRecord:
        """Transition COMPETE → SURVIVE."""
        return self.transition_to(LifecycleState.SURVIVE, reason="fitness_threshold_met")

    def breed(self) -> TransitionRecord:
        """Transition SURVIVE → BREED."""
        return self.transition_to(LifecycleState.BREED, reason="breeding_triggered")

    def spawn_child(self) -> TransitionRecord:
        """Transition BREED → EGG (new generation)."""
        return self.transition_to(LifecycleState.EGG, reason="child_spawned")

    def sunset(self) -> TransitionRecord:
        """Transition any state → SUNSET."""
        with self._lock:
            # SUNSET is reachable from most states
            if LifecycleState.SUNSET in self.VALID_TRANSITIONS.get(self._state, set()):
                return self.transition_to(LifecycleState.SUNSET, reason="sunset_called")
            raise TransitionError(f"Cannot sunset from {self._state.name}")

    def archive(self) -> TransitionRecord:
        """Transition SUNSET → ARCHIVE."""
        return self.transition_to(LifecycleState.ARCHIVE, reason="archived")
