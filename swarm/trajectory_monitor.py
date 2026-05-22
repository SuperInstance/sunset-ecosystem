"""TrajectoryMonitor — detects anomalous latent-vector trajectories.

Adversarial breeding attack model:
    An agent appears benign for N generations, then suddenly jumps
    to a distant latent position to activate a dormant backdoor.
    This manifests as an anomalous z-score acceleration in the
    agent's trajectory (displacement magnitude far exceeds historical
    mean).
"""

from __future__ import annotations

__all__ = ["TrajectoryMonitor", "SecurityEvent"]

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecurityEvent:
    """A security event raised when an anomalous trajectory is detected."""

    agent_id: int
    z_score: float
    threshold: float
    generation_count: int
    message: str


class TrajectoryMonitor:
    """Detects anomalous latent vector trajectories in the breeder population.

    Args:
        window_size: Number of recent vectors to retain per agent.
        z_threshold: Z-score above which a displacement is flagged anomalous.
    """

    def __init__(self, window_size: int = 10, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._trajectories: dict[int, deque[np.ndarray]] = {}
        self._events: list[SecurityEvent] = []

    # ── public API ──────────────────────────────────────────

    def record(self, agent_id: int, vector: np.ndarray) -> None:
        """Record a new vector position for an agent."""
        if agent_id not in self._trajectories:
            self._trajectories[agent_id] = deque(maxlen=self.window_size)
        self._trajectories[agent_id].append(np.asarray(vector, dtype=np.float32))

    def z_score_acceleration(self, agent_id: int) -> float:
        """Compute z-score of the latest displacement vs. historical mean.

        Returns 0.0 if insufficient data (< 3 vectors) or std is 0.
        """
        traj = self._trajectories.get(agent_id)
        if traj is None or len(traj) < 3:
            return 0.0

        # Compute consecutive displacements
        displacements = []
        vectors = list(traj)
        for i in range(1, len(vectors)):
            d = float(np.linalg.norm(vectors[i] - vectors[i - 1]))
            displacements.append(d)

        if len(displacements) < 2:
            return 0.0

        # Historical displacements = all except the latest one
        historical = displacements[:-1]
        latest = displacements[-1]

        mean = float(np.mean(historical))
        std = float(np.std(historical, ddof=1))  # sample std

        if std == 0:
            return 0.0 if latest == mean else float("inf")

        return (latest - mean) / std

    def is_anomalous(self, agent_id: int) -> bool:
        """True if the agent's latest move is statistically anomalous."""
        return self.z_score_acceleration(agent_id) > self.z_threshold

    def circuit_breaker(self, agent_ids: list[int]) -> list[int]:
        """Return list of agent_ids that should be immediately sunset."""
        flagged = []
        for aid in agent_ids:
            z = self.z_score_acceleration(aid)
            if z > self.z_threshold:
                flagged.append(aid)
                traj = self._trajectories.get(aid)
                gen_count = len(traj) if traj else 0
                event = SecurityEvent(
                    agent_id=aid,
                    z_score=z,
                    threshold=self.z_threshold,
                    generation_count=gen_count,
                    message=(
                        f"Anomalous trajectory detected for agent {aid}: "
                        f"z_score={z:.2f} > threshold={self.z_threshold}"
                    ),
                )
                self._events.append(event)
                logger.warning(event.message)
        return flagged

    def get_events(self) -> list[SecurityEvent]:
        """Return all recorded security events (and clear the buffer)."""
        events = list(self._events)
        self._events.clear()
        return events

    def clear(self, agent_id: Optional[int] = None) -> None:
        """Clear trajectory data for one agent (or all if None)."""
        if agent_id is None:
            self._trajectories.clear()
        else:
            self._trajectories.pop(agent_id, None)
