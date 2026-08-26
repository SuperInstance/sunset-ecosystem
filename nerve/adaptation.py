"""Adaptation Engine — Tracks the shoe metaphor lifecycle across all fibers.

Manages the progression: feel every edge → dampen → muscle memory → novelty alert.

The AdaptationEngine is the coordinator that watches all nerve fibers and
decides when the system's overall "cognitive load" is decreasing for routine
tasks, freeing capacity for novel reasoning.

The ShoeTracker provides a simple metric: how many of your "shoes" have
you stopped noticing?
"""

from __future__ import annotations

__all__ = ["AdaptationEngine", "ShoeTracker"]

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .fiber import NerveFiber, FiberState


@dataclass
class ShoeState:
    """Tracks the adaptation state of a single 'shoe' (signal pattern).

    Attributes:
        pattern_id: The pattern this shoe represents.
        put_on_time: When the shoe was first put on (first encountered).
        notice_level: How much you still notice it (1.0 = full, 0.0 = automatic).
        steps: How many times this pattern has been processed.
        last_state: The fiber state at last observation.
    """

    pattern_id: str
    put_on_time: float = field(default_factory=time.time)
    notice_level: float = 1.0
    steps: int = 0
    last_state: FiberState = FiberState.PERCEIVING

    def __repr__(self) -> str:
        return (
            f"ShoeState(pattern={self.pattern_id[:8]}..., "
            f"notice={self.notice_level:.2f}, steps={self.steps}, "
            f"state={self.last_state.value})"
        )


class ShoeTracker:
    """Tracks the 'shoe metaphor' across all nerve fibers.

    How many shoes are you still noticing? How many have become muscle memory?

    This provides a simple metric for the system's adaptation progress:
    - 0% adapted = everything is new, full cognitive load
    - 100% adapted = everything is compiled, maximum free capacity
    """

    def __init__(self) -> None:
        self._shoes: dict[str, ShoeState] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        total = len(self._shoes)
        compiled = sum(
            1 for s in self._shoes.values() if s.last_state == FiberState.COMPILED
        )
        return f"ShoeTracker(total={total}, compiled={compiled})"

    def step(self, pattern_id: str, fiber_state: FiberState) -> ShoeState:
        """Record a processing step for a pattern.

        Args:
            pattern_id: The pattern being processed.
            fiber_state: The fiber's current lifecycle state.

        Returns:
            Updated ShoeState.
        """
        with self._lock:
            if pattern_id not in self._shoes:
                self._shoes[pattern_id] = ShoeState(pattern_id=pattern_id)

            shoe = self._shoes[pattern_id]
            shoe.steps += 1
            shoe.last_state = fiber_state

            # Notice level decays with each step
            if fiber_state == FiberState.COMPILED:
                shoe.notice_level = max(0.0, shoe.notice_level - 0.2)
            elif fiber_state == FiberState.ADAPTING:
                shoe.notice_level = max(0.1, shoe.notice_level - 0.05)
            elif fiber_state == FiberState.NOVELTY_ALERT:
                shoe.notice_level = min(1.0, shoe.notice_level + 0.5)

            return shoe

    @property
    def adaptation_score(self) -> float:
        """Overall adaptation score — 0% (everything new) to 100% (everything compiled).

        This is the muscle memory metric. Higher = more free cognitive capacity.
        """
        with self._lock:
            if not self._shoes:
                return 0.0
            return 1.0 - (
                sum(s.notice_level for s in self._shoes.values()) / len(self._shoes)
            )

    @property
    def compiled_count(self) -> int:
        """Number of patterns that have reached muscle memory."""
        with self._lock:
            return sum(
                1 for s in self._shoes.values() if s.last_state == FiberState.COMPILED
            )

    @property
    def total_patterns(self) -> int:
        """Total number of unique patterns tracked."""
        with self._lock:
            return len(self._shoes)


class AdaptationEngine:
    """Coordinates nerve fiber adaptation across the whole ecosystem.

    Watches all fibers, tracks shoe states, and provides system-wide
    adaptation metrics.

    Args:
        fibers: Dictionary of fiber_id → NerveFiber to manage.
    """

    def __init__(self, fibers: dict[str, NerveFiber] | None = None) -> None:
        self._fibers: dict[str, NerveFiber] = fibers or {}
        self._tracker = ShoeTracker()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"AdaptationEngine(fibers={len(self._fibers)}, "
            f"adaptation={self.adaptation_score:.1%})"
        )

    def register(self, fiber: NerveFiber) -> None:
        """Register a nerve fiber for adaptation tracking."""
        with self._lock:
            self._fibers[fiber.fiber_id] = fiber

    @property
    def adaptation_score(self) -> float:
        """System-wide adaptation score."""
        return self._tracker.adaptation_score

    @property
    def shoe_tracker(self) -> ShoeTracker:
        """Access the shoe tracker directly."""
        return self._tracker

    def process_signal(
        self,
        fiber_id: str,
        signal: Any,
    ) -> dict[str, Any]:
        """Process a signal through a specific fiber and track adaptation.

        Args:
            fiber_id: Which fiber to use.
            signal: The raw signal.

        Returns:
            Dict with the sensory tile and adaptation metrics.
        """
        fiber = self._fibers.get(fiber_id)
        if not fiber:
            return {"error": f"Unknown fiber: {fiber_id}"}

        tile = fiber.perceive(signal)
        shoe = self._tracker.step(tile.pattern_id, tile.state)

        return {
            "tile": tile,
            "shoe": shoe,
            "adaptation_score": self.adaptation_score,
            "fiber_state": fiber.state,
            "fiber_confidence": fiber.confidence,
        }

    def system_status(self) -> dict[str, Any]:
        """Get system-wide adaptation status."""
        fibers_status = {}
        compiled = 0
        perceiving = 0
        adapting = 0
        novelty = 0

        for fid, fiber in self._fibers.items():
            fibers_status[fid] = fiber.stats
            if fiber.state == FiberState.COMPILED:
                compiled += 1
            elif fiber.state == FiberState.PERCEIVING:
                perceiving += 1
            elif fiber.state == FiberState.ADAPTING:
                adapting += 1
            elif fiber.state == FiberState.NOVELTY_ALERT:
                novelty += 1

        return {
            "total_fibers": len(self._fibers),
            "compiled": compiled,
            "perceiving": perceiving,
            "adapting": adapting,
            "novelty_alerts": novelty,
            "adaptation_score": self.adaptation_score,
            "total_patterns": self._tracker.total_patterns,
            "compiled_patterns": self._tracker.compiled_count,
        }
