"""Node liveness detection with failure suspicion.

Tracks heartbeat arrivals from fleet nodes. Uses a phi-accrual style
suspicion level to gracefully handle network jitter before declaring
a node dead.

Usage:
    monitor = HeartbeatMonitor(suspicion_threshold=8.0)
    monitor.beat("node-2")
    monitor.beat("node-2")
    status = monitor.status("node-2")  # "healthy"
    time.sleep(10)
    status = monitor.status("node-2")  # "suspected" or "dead"
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """
    Phi-accrual style heartbeat monitor.

    :param suspicion_threshold: Phi value above which a node is suspected dead.
    :param max_history: Number of inter-arrival intervals to keep per node.
    :param dead_timeout: Absolute timeout (seconds) to declare dead.
    """

    def __init__(
        self,
        suspicion_threshold: float = 8.0,
        max_history: int = 100,
        dead_timeout: float = 30.0,
    ):
        self._threshold = suspicion_threshold
        self._max_history = max_history
        self._dead_timeout = dead_timeout
        self._nodes: Dict[str, "_NodeState"] = {}

    # ------------------------------------------------------------------
    # Heartbeat tracking
    # ------------------------------------------------------------------

    def beat(self, node_id: str) -> None:
        """Record a heartbeat from a node."""
        now = time.time()
        if node_id not in self._nodes:
            self._nodes[node_id] = _NodeState(node_id)
        state = self._nodes[node_id]
        if state.last_beat > 0:
            interval = now - state.last_beat
            state.intervals.append(interval)
            if len(state.intervals) > self._max_history:
                state.intervals.pop(0)
        state.last_beat = now
        state.suspected = False

    def status(self, node_id: str) -> str:
        """
        Return node status: "healthy", "suspected", or "dead".
        """
        if node_id not in self._nodes:
            return "dead"
        state = self._nodes[node_id]
        now = time.time()
        elapsed = now - state.last_beat

        # Absolute timeout
        if elapsed > self._dead_timeout:
            return "dead"

        # Phi suspicion
        if len(state.intervals) >= 2:
            phi = self._compute_phi(elapsed, state.intervals)
            if phi >= self._threshold:
                state.suspected = True
                return "suspected"

        return "healthy"

    def phi(self, node_id: str) -> float:
        """Return current phi value for a node."""
        if node_id not in self._nodes:
            return float("inf")
        state = self._nodes[node_id]
        elapsed = time.time() - state.last_beat
        if len(state.intervals) < 2:
            return 0.0
        return self._compute_phi(elapsed, state.intervals)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def remove(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            return True
        return False

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def healthy_nodes(self) -> List[str]:
        return [n for n in self._nodes if self.status(n) == "healthy"]

    def suspected_nodes(self) -> List[str]:
        return [n for n in self._nodes if self.status(n) == "suspected"]

    def dead_nodes(self) -> List[str]:
        return [n for n in self._nodes if self.status(n) == "dead"]

    # ------------------------------------------------------------------
    # Phi computation
    # ------------------------------------------------------------------

    def _compute_phi(self, elapsed: float, intervals: List[float]) -> float:
        """Compute phi suspicion level from elapsed time and interval distribution."""
        mean = statistics.mean(intervals)
        if len(intervals) >= 2:
            try:
                std = statistics.stdev(intervals)
            except statistics.StatisticsError:
                std = 0.0
        else:
            std = 0.0
        if std == 0:
            # If no variance, use simple linear suspicion
            return elapsed / mean if mean > 0 else float("inf")
        # Phi = -log10(1 - CDF(elapsed))
        # Approximate with exponential distribution
        import math

        cdf = 0.5 * (1 + math.erf((elapsed - mean) / (std * math.sqrt(2))))
        if cdf >= 1.0:
            return float("inf")
        try:
            return -math.log10(1 - cdf)
        except ValueError:
            return float("inf")

    def __repr__(self) -> str:
        return f"<HeartbeatMonitor nodes={len(self._nodes)}>"


@dataclass
class _NodeState:
    """Internal state for a monitored node."""

    node_id: str
    last_beat: float = 0.0
    intervals: List[float] = field(default_factory=list)
    suspected: bool = False
