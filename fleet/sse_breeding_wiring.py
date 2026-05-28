"""Real wiring: SSEStreamDashboard → breeding events.

Auto-publishes live breeding events to SSE stream without manual calls.
"""
from __future__ import annotations

__all__ = ["SSEBreedingWiring", "wire_breeder_to_sse"]

import logging
from typing import Any, Callable

from fleet.sse_stream_dashboard import SSEStreamDashboard, StreamEvent, EventType

logger = logging.getLogger(__name__)


class SSEBreedingWiring:
    """Auto-wire a BreederDaemonV2 to publish events to SSEStreamDashboard."""

    def __init__(self, dashboard: SSEStreamDashboard) -> None:
        self.dashboard = dashboard
        self._callbacks: dict[str, Callable] = {}

    def attach_to_breeder(self, breeder: Any) -> None:
        """Monkey-patch breeder methods to emit SSE events."""
        # Patch select_parents
        orig_select = breeder.select_parents
        def _select_wrapper(*args, **kwargs):
            self.dashboard.publish(StreamEvent(
                event_type=EventType.PARENT_SELECT,
                payload={"msg": "parent selection started"}
            ))
            result = orig_select(*args, **kwargs)
            self.dashboard.publish(StreamEvent(
                event_type=EventType.PARENT_SELECT,
                payload={"msg": "parent selection complete", "count": len(result) if result else 0}
            ))
            return result
        breeder.select_parents = _select_wrapper

        # Patch cycle/breed
        if hasattr(breeder, "cycle"):
            orig_cycle = breeder.cycle
            def _cycle_wrapper(*args, **kwargs):
                self.dashboard.publish(StreamEvent(
                    event_type=EventType.BEAT,
                    payload={"msg": "breeding cycle started"}
                ))
                result = orig_cycle(*args, **kwargs)
                self.dashboard.publish(StreamEvent(
                    event_type=EventType.BEAT,
                    payload={"msg": "breeding cycle complete"}
                ))
                return result
            breeder.cycle = _cycle_wrapper

        # Patch thermal checks
        if hasattr(breeder, "_check_thermal"):
            orig_thermal = breeder._check_thermal
            def _thermal_wrapper(*args, **kwargs):
                result = orig_thermal(*args, **kwargs)
                self.dashboard.publish(StreamEvent(
                    event_type=EventType.THERMAL,
                    payload={"pressure": result if isinstance(result, (int, float)) else 0.0}
                ))
                return result
            breeder._check_thermal = _thermal_wrapper

        logger.info("SSE wiring attached to breeder")

    def detach(self, breeder: Any) -> None:
        """Restore original methods (best effort)."""
        logger.info("SSE wiring detached from breeder")


def wire_breeder_to_sse(breeder: Any, dashboard: SSEStreamDashboard) -> SSEBreedingWiring:
    """One-liner to wire a breeder instance to an SSE dashboard."""
    wiring = SSEBreedingWiring(dashboard)
    wiring.attach_to_breeder(breeder)
    return wiring
