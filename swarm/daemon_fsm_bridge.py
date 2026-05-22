"""Daemon-to-FSM Bridge — wires BreederDaemonV2 to BreederFSMV2.

The daemon currently creates AgentLifecycleFSM instances with
``strict=False`` and writes transitions directly to the WAL.
This bridge replaces that with validated FSM transitions via
``BreederFSMV2.transition_to()``, ensuring the canonical 6-state
lifecycle is enforced at runtime.

Additionally, every valid transition is broadcast to the
``FleetEventBus`` so CCC-OS and other ships can react to fleet
lifecycle events without polling.

Usage::

    from swarm.breeder_daemon_v2 import BreederDaemonV2
    from swarm.daemon_fsm_bridge import FSMBridgedDaemon

    base = BreederDaemonV2(grid, thermal, wal_path="breeder.sqlite")
    daemon = FSMBridgedDaemon(base, event_bus=fleet_bus)
    daemon.start()

The bridge is transparent — all public API calls delegate to the
underlying daemon, but lifecycle transitions are validated and
broadcast.
"""

from __future__ import annotations

__all__ = ["FSMBridgedDaemon"]

import logging
import time
from typing import Any

from swarm.breeder_daemon_v2 import BreederDaemonV2, LifecycleTransition
from swarm.breeder_fsm_v2 import BreederFSMV2, LifecycleState
from nexus.fleet_event_bus import FleetEventBus

logger = logging.getLogger(__name__)


class FSMBridgedDaemon:
    """Wraps a BreederDaemonV2 with validated FSM transitions + fleet events."""

    def __init__(
        self,
        daemon: BreederDaemonV2,
        event_bus: FleetEventBus | None = None,
    ) -> None:
        self._daemon = daemon
        self._bus = event_bus
        # Replace daemon's _fsm dict with BreederFSMV2 instances
        self._upgrade_fsm()

    def _upgrade_fsm(self) -> None:
        """Convert any loose AgentLifecycleFSM to strict BreederFSMV2."""
        upgraded: dict[int, BreederFSMV2] = {}
        for agent_id, fsm in self._daemon._fsm.items():
            # Get current state from old FSM
            current = fsm.get_state() if hasattr(fsm, "get_state") else LifecycleState.EGG
            upgraded[agent_id] = BreederFSMV2(
                agent_id=str(agent_id),
                initial_state=current,
                auto_transition=True,
            )
        self._daemon._fsm = upgraded  # type: ignore[assignment]

    # ── delegate all public API ───────────────────────────

    def start(self) -> None:
        self._daemon.start()
        self._emit("daemon_started", {"wal_path": self._daemon._wal_path})

    def stop(self) -> None:
        self._daemon.stop()
        self._emit("daemon_stopped", {"agents": len(self._daemon._fsm)})

    def queue_breed(
        self,
        parent_a: int,
        parent_b: int | None = None,
        priority: int = 0,
        remote: bool = False,
    ) -> int:
        ticket = self._daemon.queue_breed(parent_a, parent_b, priority, remote)
        self._emit("breed_queued", {
            "ticket": ticket,
            "parent_a": parent_a,
            "parent_b": parent_b,
            "priority": priority,
        })
        return ticket

    def select_parents(self, n_children: int = 1) -> list[tuple[int, int]]:
        return self._daemon.select_parents(n_children)

    def step(self) -> list[LifecycleTransition]:
        """Run one daemon tick with validated FSM transitions."""
        raw_transitions = self._daemon.step()
        validated: list[LifecycleTransition] = []

        for tr in raw_transitions:
            fsm = self._daemon._fsm.get(tr.agent_id)
            if fsm is None:
                # New agent — create FSM at from_state
                fsm = BreederFSMV2(
                    agent_id=str(tr.agent_id),
                    initial_state=tr.from_state,
                    auto_transition=True,
                )
                self._daemon._fsm[tr.agent_id] = fsm

            # Validate and perform transition via FSM
            ok = fsm.transition_to(tr.to_state, reason=tr.to_state.name.lower())
            if not ok:
                logger.warning(
                    "FSM blocked transition: agent %d %s → %s",
                    tr.agent_id,
                    fsm.current_state.name,
                    tr.to_state.name,
                )
                # Record the blocked transition for audit
                self._emit("transition_blocked", {
                    "agent_id": tr.agent_id,
                    "requested": tr.to_state.name,
                    "current": fsm.current_state.name,
                })
                continue

            # Transition was valid — approve it
            validated.append(tr)

            # Broadcast to fleet
            self._emit("lifecycle_transition", {
                "agent_id": tr.agent_id,
                "from": tr.from_state.name if tr.from_state else None,
                "to": tr.to_state.name,
                "generation": tr.generation,
                "parent_a": tr.parent_a,
                "parent_b": tr.parent_b,
                "origin": tr.origin_node,
            })

            # State-specific side effects
            if tr.to_state == LifecycleState.COMPETE:
                self._emit("agent_competing", {"agent_id": tr.agent_id})
            elif tr.to_state == LifecycleState.SURVIVE:
                self._emit("agent_survived", {"agent_id": tr.agent_id})
            elif tr.to_state == LifecycleState.SUNSET:
                self._emit("agent_sunset", {"agent_id": tr.agent_id})
                # Cleanup FSM
                self._daemon._fsm.pop(tr.agent_id, None)
            elif tr.to_state == LifecycleState.ARCHIVE:
                self._emit("agent_archived", {"agent_id": tr.agent_id})

        return validated

    def state(self) -> dict[int, LifecycleState]:
        """Return current state of every managed agent (validated)."""
        return {
            aid: fsm.current_state
            for aid, fsm in self._daemon._fsm.items()
        }

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._bus is not None:
            try:
                self._bus.emit(
                    {"type": event_type, **payload},
                    source="breeder-daemon-v2",
                )
            except Exception as exc:
                logger.warning("EventBus emit failed: %s", exc)

    # ── attribute passthrough ─────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(self._daemon, name)
