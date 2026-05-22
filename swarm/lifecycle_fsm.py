"""AgentLifecycleFSM — finite-state machine for fleet agent lifecycles.

States
------
EGG      : Vector exists, no room allocated, not yet competing.
COMPETE  : Active in tournament, chaos decaying.
SURVIVE  : Pareto non-dominated, stable activity, eligible to breed.
BREED    : Actively breeding (producing a child).
SUNSET   : Retired, room freed, awaiting archival.
ARCHIVE  : Permanently archived, no longer tracked in hot memory.

Transitions
-----------
EGG  → COMPETE   (init / room allocation)
COMPETE → SURVIVE  (won tournament)
COMPETE → SUNSET   (lost tournament)
SURVIVE → BREED    (selected for breeding)
SURVIVE → COMPETE  (re-enter tournament)
BREED   → EGG      (child spawned)
SUNSET  → ARCHIVE  (final cleanup)

All other transitions are invalid and raise LifecycleTransitionError.
"""

from __future__ import annotations

__all__ = [
    "AgentLifecycleFSM",
    "LifecycleState",
    "LifecycleTransitionError",
    "TransitionRecord",
]

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class LifecycleState(Enum):
    """Explicit lifecycle states for every agent in the fleet."""

    EGG = auto()      # Vector exists in table, no room allocated
    COMPETE = auto()  # Active, chaos decaying
    SURVIVE = auto()  # Pareto non-dominated, stable activity
    BREED = auto()    # Actively breeding
    SUNSET = auto()   # Retired, room freed
    ARCHIVE = auto()  # Permanently archived


class LifecycleTransitionError(ValueError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, from_state: LifecycleState, to_state: LifecycleState) -> None:
        super().__init__(
            f"Invalid transition: {from_state.name} → {to_state.name}"
        )
        self.from_state = from_state
        self.to_state = to_state


@dataclass(frozen=True)
class TransitionRecord:
    """Immutable record of a single state change."""

    from_state: LifecycleState
    to_state: LifecycleState
    timestamp: float
    reason: Optional[str] = None


class AgentLifecycleFSM:
    """Finite-state machine governing a single agent's lifecycle.

    Args:
        agent_id: Unique identifier for the agent.
        initial_state: Starting state (default EGG).
        strict: If True (default), invalid transitions raise
            ``LifecycleTransitionError``.  If False, they are silently
            ignored.
    """

    # Canonical transition graph
    _VALID: dict[LifecycleState, set[LifecycleState]] = {
        LifecycleState.EGG: {LifecycleState.COMPETE},
        LifecycleState.COMPETE: {LifecycleState.SURVIVE, LifecycleState.SUNSET},
        LifecycleState.SURVIVE: {LifecycleState.BREED, LifecycleState.COMPETE},
        LifecycleState.BREED: {LifecycleState.EGG},
        LifecycleState.SUNSET: {LifecycleState.ARCHIVE},
        LifecycleState.ARCHIVE: set(),  # terminal
    }

    def __init__(
        self,
        agent_id: int,
        *,
        initial_state: LifecycleState = LifecycleState.EGG,
        strict: bool = True,
    ) -> None:
        self._agent_id = agent_id
        self._state = initial_state
        self._strict = strict
        self._history: list[TransitionRecord] = []

        # Seed history with the initial state
        import time

        self._history.append(
            TransitionRecord(
                from_state=initial_state,
                to_state=initial_state,
                timestamp=time.time(),
                reason="init",
            )
        )

    # ── public API ──────────────────────────────────────────

    @property
    def agent_id(self) -> int:
        return self._agent_id

    def transition(self, to_state: LifecycleState, *, reason: Optional[str] = None) -> bool:
        """Attempt to move the agent to *to_state*.

        Returns:
            True if the transition succeeded, False if it was rejected
            (only possible when ``strict=False``).

        Raises:
            LifecycleTransitionError: When ``strict=True`` and the
                transition is not in the canonical graph.
        """
        import time

        if to_state == self._state:
            # Idempotent — no-op
            return True

        if to_state not in self._VALID.get(self._state, set()):
            if self._strict:
                raise LifecycleTransitionError(self._state, to_state)
            return False

        record = TransitionRecord(
            from_state=self._state,
            to_state=to_state,
            timestamp=time.time(),
            reason=reason,
        )
        self._history.append(record)
        self._state = to_state
        return True

    def get_state(self) -> LifecycleState:
        """Current lifecycle state."""
        return self._state

    def get_history(self) -> list[TransitionRecord]:
        """Full immutable copy of transition history."""
        return list(self._history)

    def can_breed(self) -> bool:
        """True iff the agent is eligible to initiate breeding.

        Only ``SURVIVE`` agents may breed.
        """
        return self._state == LifecycleState.SURVIVE

    def can_compete(self) -> bool:
        """True iff the agent may enter or re-enter tournament.

        ``EGG`` and ``SURVIVE`` agents can compete.
        """
        return self._state in {LifecycleState.EGG, LifecycleState.SURVIVE}

    def is_terminal(self) -> bool:
        """True if the agent has reached an absorbing state."""
        return self._state == LifecycleState.ARCHIVE

    def last_transition(self) -> TransitionRecord:
        """Most recent transition record."""
        return self._history[-1]

    def __repr__(self) -> str:
        return (
            f"AgentLifecycleFSM(agent_id={self._agent_id}, "
            f"state={self._state.name}, strict={self._strict})"
        )
