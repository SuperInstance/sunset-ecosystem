"""fleet/fleet_consciousness_bridge.py — Fleet Consciousness Dashboard pattern integration.

Brings the fleet-consciousness-dashboard FCI (Fleet Consciousness Index)
into sunset-ecosystem as a zero-dependency Python module.

The FCI is a weighted composite score (0.0–1.0) measuring fleet health:
- Room Phi (40%) — room integration via tile count
- Attention (20%) — agent participation in attention tracking
- Learning (25%) — ratio of positive to total learning passes
- Meta (15%) — average meta-level depth of tiles

Usage:
    from fleet.fleet_consciousness_bridge import FleetConsciousnessIndex

    fci = FleetConsciousnessIndex()
    score = fci.compute(
        room_phi_score=0.30,
        attention_score=0.20,
        learning_score=0.50,
        meta_score=0.00,
    )
    print(score.fci)        # 0.385
    print(score.level)      # "aware"
    print(score.recommendation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConsciousnessScore:
    """A single FCI computation result."""

    fci: float
    level: str
    room_phi_score: float
    attention_score: float
    learning_score: float
    meta_score: float
    status: str
    recommendation: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fci": self.fci,
            "level": self.level,
            "room_phi_score": self.room_phi_score,
            "attention_score": self.attention_score,
            "learning_score": self.learning_score,
            "meta_score": self.meta_score,
            "status": self.status,
            "recommendation": self.recommendation,
            "details": self.details,
        }


class FleetConsciousnessIndex:
    """
    Compute the Fleet Consciousness Index (FCI).

    Compatible with fleet-consciousness-dashboard weights and levels.
    """

    # Weights from fleet-consciousness-dashboard
    WEIGHTS = {
        "room_phi": 0.40,
        "attention": 0.20,
        "learning": 0.25,
        "meta": 0.15,
    }

    # Consciousness levels
    LEVELS = [
        (
            0.00,
            0.15,
            "dormant",
            "Fleet is dormant. Activate rooms and seed initial tiles.",
        ),
        (
            0.15,
            0.30,
            "emerging",
            "Fleet is emerging. Increase agent participation and room density.",
        ),
        (
            0.30,
            0.45,
            "aware",
            "Fleet is aware. Enable attention tiles from all agents.",
        ),
        (
            0.45,
            0.60,
            "conscious",
            "Fleet is conscious. Deepen meta-level tiles and cross-room correlations.",
        ),
        (
            0.60,
            0.75,
            "self-aware",
            "Fleet is self-aware. Optimize learning passes and Penrose correlations.",
        ),
        (
            0.75,
            1.00,
            "transcendent",
            "Fleet is transcendent. Monitor for degradation and maintain diversity.",
        ),
    ]

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or dict(self.WEIGHTS)

    def compute(
        self,
        room_phi_score: float,
        attention_score: float,
        learning_score: float,
        meta_score: float,
        **details: Any,
    ) -> ConsciousnessScore:
        """Compute FCI from four component scores."""
        fci = (
            room_phi_score * self.weights["room_phi"]
            + attention_score * self.weights["attention"]
            + learning_score * self.weights["learning"]
            + meta_score * self.weights["meta"]
        )
        fci = max(0.0, min(1.0, fci))

        level, recommendation = self._level(fci)
        status = "HEALTHY" if fci >= 0.30 else "DEGRADED"

        return ConsciousnessScore(
            fci=fci,
            level=level,
            room_phi_score=room_phi_score,
            attention_score=attention_score,
            learning_score=learning_score,
            meta_score=meta_score,
            status=status,
            recommendation=recommendation,
            details=details,
        )

    def _level(self, fci: float) -> tuple[str, str]:
        for low, high, level, rec in self.LEVELS:
            if low <= fci < high:
                return level, rec
        return self.LEVELS[-1][2], self.LEVELS[-1][3]

    def render_text(self, score: ConsciousnessScore) -> str:
        """Render a text dashboard like fleet-consciousness-dashboard."""
        lines = [
            "=" * 50,
            "  FLEET CONSCIOUSNESS DASHBOARD",
            "=" * 50,
            f"  FCI: {score.fci:.3f} — {score.level.upper()}",
            f"  Status: {'✓' if score.status == 'HEALTHY' else '✗'} {score.status}",
            "-" * 50,
            f"  Room Phi Score:    {score.room_phi_score:.3f} (weight {self.weights['room_phi']})",
            f"  Attention Score:   {score.attention_score:.3f} (weight {self.weights['attention']})",
            f"  Learning Score:    {score.learning_score:.3f} (weight {self.weights['learning']})",
            f"  Meta Score:        {score.meta_score:.3f} (weight {self.weights['meta']})",
            "-" * 50,
            f"  Recommendation: {score.recommendation}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def render_json(self, score: ConsciousnessScore) -> str:
        """Render JSON output."""
        return json.dumps(score.to_dict(), indent=2)

    def render_oneline(self, score: ConsciousnessScore) -> str:
        """Render one-line summary."""
        return f"FCI: {score.fci:.3f} ({score.level}) — {score.status}"

    @classmethod
    def from_fleet_metrics(
        cls,
        rooms: int,
        total_rooms_capacity: int = 100,
        active_agents: int = 0,
        total_agents: int = 1,
        positive_learning_passes: int = 0,
        total_learning_passes: int = 1,
        meta_tile_depth_sum: float = 0.0,
        total_tiles: int = 1,
        **details: Any,
    ) -> ConsciousnessScore:
        """
        Compute FCI from raw fleet metrics.

        Args:
            rooms: Number of active rooms
            total_rooms_capacity: Total room capacity (for normalization)
            active_agents: Agents participating in attention tracking
            total_agents: Total agents in fleet
            positive_learning_passes: Positive learning outcomes
            total_learning_passes: Total learning attempts
            meta_tile_depth_sum: Sum of meta-level depths across tiles
            total_tiles: Total number of tiles
        """
        fci = cls()
        room_phi = rooms / max(total_rooms_capacity, 1)
        attention = active_agents / max(total_agents, 1)
        learning = positive_learning_passes / max(total_learning_passes, 1)
        meta = (meta_tile_depth_sum / max(total_tiles, 1)) / 10.0  # normalize to ~0-1
        return fci.compute(
            room_phi_score=room_phi,
            attention_score=attention,
            learning_score=learning,
            meta_score=min(meta, 1.0),
            raw_metrics={
                "rooms": rooms,
                "total_rooms_capacity": total_rooms_capacity,
                "active_agents": active_agents,
                "total_agents": total_agents,
                "positive_learning_passes": positive_learning_passes,
                "total_learning_passes": total_learning_passes,
                "meta_tile_depth_sum": meta_tile_depth_sum,
                "total_tiles": total_tiles,
            },
            **details,
        )
