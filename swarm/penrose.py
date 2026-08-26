"""Penrose lattice agent distribution — aperiodic diversity guarantee.

Uses the golden angle (≈137.5°) for sunflower-style distribution.
Each agent starts at a unique position — no two see the problem the same way.
"""

from __future__ import annotations

__all__ = ["PenrosePosition", "assign_positions", "compute_overlap"]

import math
from dataclasses import dataclass

# Golden ratio and golden angle
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618
GOLDEN_ANGLE = 2 * math.pi / (PHI * PHI)  # ≈ 137.5° in radians


@dataclass
class PenrosePosition:
    """A position on the Penrose-like lattice.

    Each position is unique. The golden angle spacing guarantees
    no two agents line up on any axis.

    Attributes:
        agent_id: The agent at this position.
        x: X coordinate.
        y: Y coordinate.
        ring: Which ring of the spiral (0 = center).
        angle: Angle from the positive x-axis (radians).
    """

    agent_id: str
    x: float
    y: float
    ring: int
    angle: float

    def __repr__(self) -> str:
        return (
            f"PenrosePosition({self.agent_id}, "
            f"({self.x:.2f}, {self.y:.2f}), ring={self.ring})"
        )

    def distance_to(self, other: PenrosePosition) -> float:
        """Euclidean distance to another position."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def assign_positions(agent_ids: list[str]) -> list[PenrosePosition]:
    """Assign agents to positions on a golden-angle spiral.

    Uses the Vogel model: r = c * sqrt(n), θ = n * golden_angle.
    This produces a sunflower-like distribution where no two
    agents are at the same angle.

    Args:
        agent_ids: List of agent IDs to position.

    Returns:
        List of PenrosePosition, one per agent.
    """
    positions: list[PenrosePosition] = []
    n = len(agent_ids)

    for i, agent_id in enumerate(agent_ids):
        angle = i * GOLDEN_ANGLE
        ring = int(math.sqrt(i + 1))
        r = math.sqrt(i + 1)  # Vogel model

        x = r * math.cos(angle)
        y = r * math.sin(angle)

        positions.append(
            PenrosePosition(
                agent_id=agent_id,
                x=x,
                y=y,
                ring=ring,
                angle=angle % (2 * math.pi),
            )
        )

    return positions


def compute_overlap(
    pos_a: PenrosePosition,
    pos_b: PenrosePosition,
    radius: float = 1.0,
) -> float:
    """Compute perspective overlap between two positions.

    Overlap is 1.0 if agents are at the same position, 0.0 if
    they're more than 2*radius apart. Linear interpolation between.

    Args:
        pos_a: First position.
        pos_b: Second position.
        radius: Each agent's "perception radius".

    Returns:
        Overlap score (0.0 to 1.0).
    """
    dist = pos_a.distance_to(pos_b)
    max_dist = 2 * radius
    if dist >= max_dist:
        return 0.0
    return 1.0 - (dist / max_dist)


def minimum_overlap(positions: list[PenrosePosition], radius: float = 1.0) -> float:
    """Find the worst-case (minimum) overlap between any two positions.

    Higher = less diverse. Lower = more diverse perspectives.

    Args:
        positions: All agent positions.
        radius: Perception radius.

    Returns:
        The minimum overlap across all pairs.
    """
    if len(positions) < 2:
        return 0.0

    min_overlap = 1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            ov = compute_overlap(positions[i], positions[j], radius)
            min_overlap = min(min_overlap, ov)
    return min_overlap
