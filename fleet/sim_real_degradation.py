"""SIM/REAL degradation stack for fleet health monitoring.

Implements Pattern 3 from the SuperInstance audit: maintains two data
modes (SIMulation and REAL) with a three-tier degradation stack that
detects when REAL data becomes unreliable and smoothly degrades to
SIMULATION mode.

Reference: holodeck-rust — sim_real_degradation stack pattern.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DegradationLevel(enum.IntEnum):
    """Three-tier degradation stack."""

    GREEN = 0  # All REAL data, full confidence
    YELLOW = 1  # Partial degradation — some REAL data unreliable
    RED = 2  # Full SIMULATION mode — all data synthetic


@dataclass
class DataSource:
    """A single data source with health metrics."""

    name: str
    mode: str = "REAL"  # "REAL" or "SIM"
    confidence: float = 1.0  # [0, 1] — confidence in this source
    latency_ms: float = 0.0  # Last known latency
    last_update: float = field(
        default_factory=time.time
    )  # Unix timestamp of last update
    stale_threshold_ms: float = 5000.0  # Threshold to consider stale

    def is_stale(self, now: Optional[float] = None) -> bool:
        """Check if data source hasn't updated within threshold."""
        now = now or time.time()
        elapsed_ms = (now - self.last_update) * 1000.0
        return elapsed_ms > self.stale_threshold_ms

    def health_score(self, now: Optional[float] = None) -> float:
        """Compute health score [0, 1] for this source."""
        now = now or time.time()
        if self.is_stale(now):
            return 0.0
        # Confidence decays with latency: latency = 0 → 1.0, latency = threshold → 0.5
        latency_factor = max(
            0.0, 1.0 - (self.latency_ms / (self.stale_threshold_ms * 2))
        )
        return self.confidence * latency_factor


@dataclass
class DegradationState:
    """Current degradation state for a subsystem."""

    level: DegradationLevel = DegradationLevel.GREEN
    sources: dict[str, DataSource] = field(default_factory=dict)
    sim_override: bool = False  # Force SIM mode (e.g., for testing)
    timestamp: float = field(default_factory=time.time)

    def overall_health(self, now: Optional[float] = None) -> float:
        """Aggregate health across all sources."""
        if not self.sources:
            return 1.0 if self.level == DegradationLevel.GREEN else 0.0
        scores = [s.health_score(now) for s in self.sources.values()]
        return sum(scores) / len(scores)

    def active_sources(self, now: Optional[float] = None) -> list[str]:
        """List sources that are not stale."""
        now = now or time.time()
        return [name for name, src in self.sources.items() if not src.is_stale(now)]


class SimRealDegradationStack:
    """Manages SIM/REAL degradation for a fleet subsystem.

    Automatically transitions between GREEN → YELLOW → RED based on:
    - Data source staleness
    - Health score thresholds
    - Manual overrides

    The stack is **sticky**: once degraded, it requires a sustained
    period of healthy data to recover (hysteresis).
    """

    # Hysteresis thresholds
    DEGRADE_TO_YELLOW = 0.7  # Health < this → degrade to YELLOW
    DEGRADE_TO_RED = 0.3  # Health < this → degrade to RED
    RECOVER_TO_GREEN = 0.85  # Health > this + sustained → recover to GREEN
    RECOVER_TO_YELLOW = 0.6  # Health > this + sustained → recover from RED
    SUSTAINED_SECONDS = 5.0  # Seconds of healthy data before recovery

    def __init__(self, subsystem: str) -> None:
        self.subsystem = subsystem
        self.state = DegradationState()
        self._health_history: list[tuple[float, float]] = []  # (timestamp, health)
        self._on_transition: list[
            Callable[[DegradationLevel, DegradationLevel], None]
        ] = []

    # ── public API ───────────────────────────────────────────

    def register_source(
        self, name: str, mode: str = "REAL", confidence: float = 1.0
    ) -> None:
        """Register a new data source."""
        self.state.sources[name] = DataSource(
            name=name,
            mode=mode,
            confidence=confidence,
            last_update=time.time(),
        )

    def update_source(
        self, name: str, latency_ms: float, confidence: Optional[float] = None
    ) -> None:
        """Update a data source with fresh metrics."""
        if name not in self.state.sources:
            raise KeyError(f"Source '{name}' not registered")
        src = self.state.sources[name]
        src.latency_ms = latency_ms
        src.last_update = time.time()
        if confidence is not None:
            src.confidence = confidence

    def tick(self, now: Optional[float] = None) -> DegradationLevel:
        """Evaluate health and transition degradation level.

        Returns the current level after evaluation.
        """
        now = now or time.time()
        health = self.state.overall_health(now)

        # Record health history for sustained-recovery check
        self._health_history.append((now, health))
        self._prune_history(now)

        old_level = self.state.level
        new_level = old_level

        if self.state.sim_override:
            new_level = DegradationLevel.RED
        else:
            new_level = self._evaluate_level(health, old_level, now)

        if new_level != old_level:
            logger.warning(
                "[%s] Degradation transition: %s → %s (health=%.2f)",
                self.subsystem,
                old_level.name,
                new_level.name,
                health,
            )
            self.state.level = new_level
            self.state.timestamp = now
            for cb in self._on_transition:
                cb(old_level, new_level)

        return self.state.level

    def force_sim(self, enabled: bool = True) -> None:
        """Manually force SIMULATION mode (testing or maintenance)."""
        self.state.sim_override = enabled

    def on_transition(
        self, callback: Callable[[DegradationLevel, DegradationLevel], None]
    ) -> None:
        """Register a callback for level transitions."""
        self._on_transition.append(callback)

    @property
    def level(self) -> DegradationLevel:
        return self.state.level

    @property
    def is_real(self) -> bool:
        """True if all data sources are REAL and healthy."""
        return self.state.level == DegradationLevel.GREEN

    @property
    def is_sim(self) -> bool:
        """True if in full SIMULATION mode."""
        return self.state.level == DegradationLevel.RED

    def select_value(self, real_value: Any, sim_value: Any) -> Any:
        """Select between REAL and SIM value based on current level.

        GREEN  → real_value
        YELLOW → weighted blend (fades from real to sim)
        RED    → sim_value
        """
        level = self.state.level
        if level == DegradationLevel.GREEN:
            return real_value
        if level == DegradationLevel.RED:
            return sim_value

        # YELLOW: blend based on overall health
        health = self.state.overall_health()
        # health at DEGRADE_TO_YELLOW (0.7) → 0% real, RECOVER_TO_GREEN (0.85) → 100% real
        t = (health - self.DEGRADE_TO_YELLOW) / (
            self.RECOVER_TO_GREEN - self.DEGRADE_TO_YELLOW
        )
        t = max(0.0, min(1.0, t))

        if isinstance(real_value, (int, float)) and isinstance(sim_value, (int, float)):
            return real_value * t + sim_value * (1.0 - t)
        # For non-numeric, threshold at 0.5
        return real_value if t > 0.5 else sim_value

    def __repr__(self) -> str:
        health = self.state.overall_health()
        srcs = len(self.state.sources)
        active = len(self.state.active_sources())
        return (
            f"SimRealDegradationStack({self.subsystem}, "
            f"level={self.state.level.name}, "
            f"health={health:.2f}, "
            f"sources={active}/{srcs})"
        )

    # ── internals ────────────────────────────────────────────

    def _evaluate_level(
        self, health: float, current: DegradationLevel, now: float
    ) -> DegradationLevel:
        """Determine new level with hysteresis."""
        if current == DegradationLevel.GREEN:
            if health < self.DEGRADE_TO_RED:
                return DegradationLevel.RED
            if health < self.DEGRADE_TO_YELLOW:
                return DegradationLevel.YELLOW
            return DegradationLevel.GREEN

        if current == DegradationLevel.YELLOW:
            if health < self.DEGRADE_TO_RED:
                return DegradationLevel.RED
            if health >= self.RECOVER_TO_GREEN and self._sustained_above(
                self.RECOVER_TO_GREEN, now
            ):
                return DegradationLevel.GREEN
            return DegradationLevel.YELLOW

        if current == DegradationLevel.RED:
            if health >= self.RECOVER_TO_GREEN and self._sustained_above(
                self.RECOVER_TO_GREEN, now
            ):
                return DegradationLevel.GREEN
            if health >= self.RECOVER_TO_YELLOW and self._sustained_above(
                self.RECOVER_TO_YELLOW, now
            ):
                return DegradationLevel.YELLOW
            return DegradationLevel.RED

        return current

    def _sustained_above(self, threshold: float, now: float) -> bool:
        """Check if health has been above threshold for sustained duration."""
        if not self._health_history:
            return False
        # Check the window [now - SUSTAINED_SECONDS, now]
        window_start = now - self.SUSTAINED_SECONDS
        recent = [(t, h) for t, h in self._health_history if t >= window_start]
        if not recent:
            return False
        # All readings in window must be above threshold
        return all(h >= threshold for _, h in recent)

    def _prune_history(self, now: float) -> None:
        """Remove old health history entries."""
        cutoff = now - (self.SUSTAINED_SECONDS * 2)
        self._health_history = [(t, h) for t, h in self._health_history if t >= cutoff]


class FleetDegradationMonitor:
    """Fleet-wide SIM/REAL monitor managing multiple subsystems."""

    def __init__(self) -> None:
        self._stacks: dict[str, SimRealDegradationStack] = {}

    def register_subsystem(self, name: str) -> SimRealDegradationStack:
        """Create and register a new subsystem degradation stack."""
        stack = SimRealDegradationStack(name)
        self._stacks[name] = stack
        return stack

    def tick_all(self, now: Optional[float] = None) -> dict[str, DegradationLevel]:
        """Evaluate all subsystems. Returns {name: level}."""
        return {name: stack.tick(now) for name, stack in self._stacks.items()}

    def fleet_health(self) -> float:
        """Aggregate health across all subsystems."""
        if not self._stacks:
            return 1.0
        scores = [s.state.overall_health() for s in self._stacks.values()]
        return sum(scores) / len(scores)

    def degraded_subsystems(self) -> list[str]:
        """List subsystems not in GREEN."""
        return [
            name
            for name, stack in self._stacks.items()
            if stack.level != DegradationLevel.GREEN
        ]

    def __repr__(self) -> str:
        return f"FleetDegradationMonitor({len(self._stacks)} subsystems)"
