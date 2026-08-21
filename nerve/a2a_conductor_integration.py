"""A2A Conductor Integration — Wire A2A metronome tasks into FleetConductor.

Extends ``FleetConductor`` (defined in ``nexus/fleet_conductor.py``) with
Agent-to-Agent task negotiation for distributed beat sync:

  - dispatch_sync_task(peer_node_id, target_beat) → send BeatSyncTask
  - handle_bpm_proposal(task) → accept / counter / reject
  - handle_drift_alert(task) → phase_nudge, skip_jump, or partition
  - register_a2a_handlers() → map task type strings to handler callables

Usage (after conductor instantiation)::

    conductor = FleetConductor("ship-jetson-01", "http://nexus.fleet.local:4047")
    conductor.register_local_scheduler(scheduler)
    conductor.register_a2a_handlers()   # enable A2A task handling
"""

from __future__ import annotations

__all__ = [
    "FleetConductorA2AExtension",
    "register_a2a_handlers",
]

import logging
from typing import Any, Callable, Optional

from nexus.fleet_conductor import FleetConductor
from nerve.a2a_metronome_tasks import (
    A2AMetronomeResult,
    BeatSyncTask,
    BPMNegotiationTask,
    DriftAlertTask,
    TickAsTask,
)

logger = logging.getLogger(__name__)

# Registry of task-type → handler callable.
# Populated by ``register_a2a_handlers()``.
_A2A_HANDLER_REGISTRY: dict[str, Callable[[Any, Any], A2AMetronomeResult]] = {}


# ── Handler functions (free-standing, patchable in tests) ──


def handle_beat_sync_task(
    conductor: FleetConductor,
    task: BeatSyncTask,
) -> A2AMetronomeResult:
    """Process an incoming BeatSyncTask.

    1. Validates the task.
    2. Executes it against a dummy / local grid placeholder.
    3. If a local scheduler is registered, computes the delta and
       potentially triggers a phase nudge or skip-jump.
    """
    task.validate()

    # Build a minimal context for execution (no RoomGrid needed for sync)
    result = task.execute(None)

    delta_beats = result.payload.get("delta_beats", 0.0)
    delta_ns = result.payload.get("delta_ns", 0)
    beat_duration_ns = result.payload.get("beat_duration_ns", 500_000_000)
    drift_ms = abs(delta_ns) / 1_000_000.0

    # Trigger conductor-level drift correction if scheduler is linked
    if conductor._scheduler is not None:
        if drift_ms > conductor.max_drift_ms:
            if delta_beats >= 1.0:
                # Upgrade to skip-jump via existing conductor machinery
                consensus = BeatState.now(beat_number=task.target_beat)
                conductor._apply_skip_jump(consensus)
            else:
                # Phase nudge
                beat_duration_ms = beat_duration_ns / 1_000_000.0
                conductor._apply_phase_nudge(
                    BeatState.now(beat_number=task.target_beat),
                    drift_ms,
                    beat_duration_ms,
                )

    logger.info(
        "BeatSyncTask from %s → delta_beats=%.3f drift_ms=%.3f",
        task.node_id,
        delta_beats,
        drift_ms,
    )
    return result


def handle_bpm_proposal_task(
    conductor: FleetConductor,
    task: BPMNegotiationTask,
) -> A2AMetronomeResult:
    """Evaluate a BPM proposal and return an accept/counter/reject result.

    If the local scheduler supports ``set_bpm``, this handler also
    mutates the scheduler when the proposal is accepted.
    """
    task.validate()

    # Evaluate against local constraints
    result = task.execute(None)

    action = result.payload.get("action")
    accepted_bpm = result.payload.get("accepted_bpm")

    if action == "accept" and accepted_bpm is not None:
        if conductor._scheduler is not None and hasattr(conductor._scheduler, "bpm"):
            old_bpm = conductor._scheduler.bpm
            conductor._scheduler.bpm = accepted_bpm
            logger.info(
                "BPM accepted on %s: %.1f → %.1f (ramp %d ms)",
                conductor.node_id,
                old_bpm,
                accepted_bpm,
                task.ramp_ms,
            )

    logger.info(
        "BPMNegotiationTask from %s → action=%s bpm=%s",
        task.node_id,
        action,
        accepted_bpm or result.payload.get("counter_bpm") or "rejected",
    )
    return result


def handle_drift_alert_task(
    conductor: FleetConductor,
    task: DriftAlertTask,
) -> A2AMetronomeResult:
    """Classify a drift alert and apply the appropriate correction.

    Returns the *recommended* action (which may differ from the
    node's *requested* action).  The conductor enforces fleet-wide
    policy: skip-jump when drift >= 1 beat, partition when drift
    is > 3× threshold, otherwise phase nudge.
    """
    task.validate()

    # Run classification logic
    result = task.execute(None)
    recommended = result.payload.get("recommended_action", "none")

    if recommended == "skip_jump":
        target = task.target_beat or 0
        consensus = BeatState.now(beat_number=target)
        conductor._apply_skip_jump(consensus)
        logger.warning(
            "DriftAlert skip-jump on %s: target_beat=%d",
            conductor.node_id,
            target,
        )

    elif recommended == "partition":
        conductor._handle_partition()
        logger.warning(
            "DriftAlert partition on %s: drift_ms=%.2f",
            conductor.node_id,
            task.drift_ms,
        )

    elif recommended == "phase_nudge":
        beat_duration_ms = conductor._beat_duration_ms()
        conductor._apply_phase_nudge(
            BeatState.now(beat_number=task.target_beat or 0),
            task.drift_ms,
            beat_duration_ms,
        )
        logger.info(
            "DriftAlert phase_nudge on %s: %.3f ms",
            conductor.node_id,
            task.drift_ms,
        )

    else:
        logger.debug(
            "DriftAlert no-action on %s: drift_ms=%.2f below threshold",
            conductor.node_id,
            task.drift_ms,
        )

    return result


def handle_tick_as_task(
    conductor: FleetConductor,
    task: TickAsTask,
) -> A2AMetronomeResult:
    """Execute a tick remotely via the local scheduler's grid.

    Falls back to executing on a synthetic grid if no scheduler is
    linked (useful for headless conductor nodes that proxy tasks).
    """
    task.validate()

    grid = None
    if conductor._scheduler is not None and hasattr(conductor._scheduler, "grid"):
        grid = conductor._scheduler.grid

    if grid is None:
        return A2AMetronomeResult(
            status="failed",
            task_type=TickAsTask.TASK_TYPE,
            error="No grid available on this conductor",
        )

    return task.execute(grid)


def handle_unknown_task(
    _conductor: FleetConductor,
    task: Any,
) -> A2AMetronomeResult:
    """Fallback handler for unrecognised A2A task types."""
    if isinstance(task, dict):
        task_type = task.get("type", "unknown")
    else:
        task_type = getattr(task, "TASK_TYPE", str(type(task)))
    return A2AMetronomeResult(
        status="failed",
        task_type=task_type,
        error=f"No handler registered for task type '{task_type}'",
    )


# ── Registration ────────────────────────────────────────────


def register_a2a_handlers(conductor: FleetConductor) -> None:
    """Bind A2A metronome task handlers to a FleetConductor instance.

    This mutates ``conductor._a2a_handlers`` (creating it if necessary)
    and populates the module-level ``_A2A_HANDLER_REGISTRY``.
    """
    handlers: dict[str, Callable[[Any, Any], A2AMetronomeResult]] = {
        BeatSyncTask.TASK_TYPE: handle_beat_sync_task,
        BPMNegotiationTask.TASK_TYPE: handle_bpm_proposal_task,
        DriftAlertTask.TASK_TYPE: handle_drift_alert_task,
        TickAsTask.TASK_TYPE: handle_tick_as_task,
    }

    # Store on the conductor instance for local dispatch
    conductor._a2a_handlers = handlers

    # Update module-level registry (used by cross-node dispatch)
    _A2A_HANDLER_REGISTRY.clear()
    _A2A_HANDLER_REGISTRY.update(handlers)

    logger.info(
        "A2A handlers registered on %s: %s",
        conductor.node_id,
        list(handlers.keys()),
    )


# ── Conductor mixin / extension helpers ───────────────────


class FleetConductorA2AExtension:
    """Mixin-style helper that adds A2A dispatch methods to FleetConductor.

    Rather than subclassing ``FleetConductor`` (which would break
    existing tests and import chains), this class provides free
    methods that accept a ``FleetConductor`` instance as their first
    argument.  They are attached to the conductor at runtime by
    ``register_a2a_handlers()``.
    """

    @staticmethod
    def dispatch_sync_task(
        conductor: FleetConductor,
        peer_node_id: str,
        target_beat: int,
    ) -> dict:
        """Build and serialise a ``BeatSyncTask`` for a peer node.

        In a full implementation this POSTs the payload to the peer's
        ``/a2a/tasks/send`` endpoint via the fleet Nexus.  Here we
        return the serialised task dict so the caller can transmit it.
        """
        task = BeatSyncTask(
            node_id=conductor.node_id,
            target_beat=target_beat,
            timestamp_ns=time.time_ns(),
            bpm=getattr(conductor._scheduler, "bpm", 120.0)
            if conductor._scheduler
            else 120.0,
        )
        payload = task.to_dict()
        logger.debug("dispatch_sync_task → %s: beat=%d", peer_node_id, target_beat)
        return payload

    @staticmethod
    def dispatch_bpm_proposal(
        conductor: FleetConductor,
        peer_node_id: str,
        proposed_bpm: float,
        ramp_ms: int = 2000,
        reason: str = "",
    ) -> dict:
        """Build and serialise a ``BPMNegotiationTask`` for a peer node."""
        task = BPMNegotiationTask(
            node_id=conductor.node_id,
            proposed_bpm=proposed_bpm,
            ramp_ms=ramp_ms,
            reason=reason,
            response_action="propose",
        )
        return task.to_dict()

    @staticmethod
    def dispatch_drift_alert(
        conductor: FleetConductor,
        peer_node_id: str,
        drift_ms: float,
        drift_beats: float,
        requested_action: str = "phase_nudge",
    ) -> dict:
        """Build and serialise a ``DriftAlertTask`` for a peer node."""
        task = DriftAlertTask(
            node_id=conductor.node_id,
            drift_ms=drift_ms,
            drift_beats=drift_beats,
            threshold_ms=conductor.max_drift_ms,
            requested_action=requested_action,
            target_beat=getattr(conductor._scheduler, "beat_number", 0)
            if conductor._scheduler
            else 0,
        )
        return task.to_dict()

    @staticmethod
    def handle_incoming_task(
        conductor: FleetConductor,
        task_dict: dict,
    ) -> A2AMetronomeResult:
        """Route an incoming A2A task dict to the correct handler.

        This is the entry point for the Nexus ``/a2a/tasks/send``
        handler: deserialise the dict, look up the handler, and run it.
        """
        task_type = task_dict.get("type", "unknown")
        handlers = getattr(conductor, "_a2a_handlers", _A2A_HANDLER_REGISTRY)

        handler = handlers.get(task_type, handle_unknown_task)

        # Deserialise the task object from the dict
        task: Any
        if task_type == BeatSyncTask.TASK_TYPE:
            task = BeatSyncTask.from_dict(task_dict)
        elif task_type == BPMNegotiationTask.TASK_TYPE:
            task = BPMNegotiationTask.from_dict(task_dict)
        elif task_type == DriftAlertTask.TASK_TYPE:
            task = DriftAlertTask.from_dict(task_dict)
        elif task_type == TickAsTask.TASK_TYPE:
            task = TickAsTask.from_dict(task_dict)
        else:
            return handle_unknown_task(conductor, task_dict)

        return handler(conductor, task)


# Re-export for convenience
from nexus.fleet_conductor import BeatState  # noqa: E402,F811
import time  # noqa: E402
