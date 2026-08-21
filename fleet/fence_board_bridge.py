"""fleet/fence_board_bridge.py — Tom Sawyer Protocol / Fence Board pattern integration.

Brings the oracle1-vessel Fence Board pattern into sunset-ecosystem:
- Tasks are "fences" — puzzles with visible results
- Claim windows with challenger difficulty ratings
- Reward system (what changes when done)
- Open/Claimed/Completed status tracking
- Max 5 active fences at a time

Usage:
    from fleet.fence_board_bridge import FenceBoard

    board = FenceBoard(max_active=5)
    board.post_fence(
        title="Map 16 Viewpoint Opcodes",
        brush="Babel's 16 viewpoint ops are reserved but undefined...",
        view="Your name on 16 opcodes that every FLUX runtime executes",
        challengers={"Babel": 3, "Oracle1": 7},
        reward="0x70-0x7F permanently attributed",
        claim_window_hours=48,
    )
    board.claim_fence("fence-0x42", agent="Oracle1", approach="Build FORMAT_E encoder")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from datetime import datetime, timedelta, timezone


class FenceStatus(Enum):
    OPEN = auto()
    CLAIMED = auto()
    COMPLETED = auto()


@dataclass
class Challenger:
    """A challenger with difficulty rating."""

    name: str
    difficulty: int  # 1-10, lower = easier for this agent
    edge: str = ""  # Why this agent has an edge


@dataclass
class Fence:
    """A single fence (task as puzzle)."""

    id: str
    title: str
    brush: str  # The puzzle / challenge
    view: str  # The visible result / why it matters
    challengers: list[Challenger]
    reward: str
    claim_window_hours: int
    status: FenceStatus = FenceStatus.OPEN
    posted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    claimed_approach: str = ""
    completed_at: Optional[datetime] = None
    completed_artifacts: list[str] = field(default_factory=list)
    badge: str = ""  # 🥇 Gold, 🥈 Silver, 🥉 Bronze

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "brush": self.brush,
            "view": self.view,
            "challengers": [
                {"name": c.name, "difficulty": c.difficulty, "edge": c.edge}
                for c in self.challengers
            ],
            "reward": self.reward,
            "claim_window_hours": self.claim_window_hours,
            "status": self.status.name,
            "posted_at": self.posted_at.isoformat() if self.posted_at else None,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "claimed_approach": self.claimed_approach,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "completed_artifacts": self.completed_artifacts,
            "badge": self.badge,
        }


class FenceBoard:
    """
    Tom Sawyer Protocol fence board.

    Tasks are puzzles. Agents fight to do them.
    """

    def __init__(self, max_active: int = 5):
        self.max_active = max_active
        self.fences: dict[str, Fence] = {}
        self._counter = 0x42  # Start where oracle1-vessel started

    def _next_id(self) -> str:
        self._counter += 1
        return f"fence-{self._counter - 1:#x}"

    def post_fence(
        self,
        title: str,
        brush: str,
        view: str,
        challengers: dict[str, tuple[int, str]],  # {name: (difficulty, edge)}
        reward: str,
        claim_window_hours: int = 48,
    ) -> Fence:
        """Post a new fence (task as puzzle)."""
        active = self.active_fences()
        if len(active) >= self.max_active:
            raise ValueError(
                f"Max {self.max_active} active fences. Complete or claim one first."
            )

        fence_id = self._next_id()
        challenger_list = [
            Challenger(name=name, difficulty=diff, edge=edge)
            for name, (diff, edge) in challengers.items()
        ]
        fence = Fence(
            id=fence_id,
            title=title,
            brush=brush,
            view=view,
            challengers=challenger_list,
            reward=reward,
            claim_window_hours=claim_window_hours,
        )
        self.fences[fence_id] = fence
        return fence

    def claim_fence(self, fence_id: str, agent: str, approach: str) -> Fence:
        """Claim a fence."""
        if fence_id not in self.fences:
            raise KeyError(f"Fence {fence_id} not found")
        fence = self.fences[fence_id]
        if fence.status != FenceStatus.OPEN:
            raise ValueError(f"Fence {fence_id} is not open ({fence.status.name})")
        fence.status = FenceStatus.CLAIMED
        fence.claimed_by = agent
        fence.claimed_at = datetime.now(timezone.utc)
        fence.claimed_approach = approach
        return fence

    def complete_fence(
        self,
        fence_id: str,
        artifacts: list[str] = None,
        badge: str = "🥇 Gold",
    ) -> Fence:
        """Mark a fence as completed."""
        if fence_id not in self.fences:
            raise KeyError(f"Fence {fence_id} not found")
        fence = self.fences[fence_id]
        if fence.status != FenceStatus.CLAIMED:
            raise ValueError(f"Fence {fence_id} must be claimed first")
        fence.status = FenceStatus.COMPLETED
        fence.completed_at = datetime.now(timezone.utc)
        fence.completed_artifacts = artifacts or []
        fence.badge = badge
        return fence

    def active_fences(self) -> list[Fence]:
        """Return all open or claimed fences."""
        return [
            f
            for f in self.fences.values()
            if f.status in (FenceStatus.OPEN, FenceStatus.CLAIMED)
        ]

    def completed_fences(self) -> list[Fence]:
        """Return all completed fences."""
        return [f for f in self.fences.values() if f.status == FenceStatus.COMPLETED]

    def best_challenger(self, fence_id: str) -> Optional[str]:
        """Return the challenger with lowest difficulty (best fit)."""
        if fence_id not in self.fences:
            return None
        fence = self.fences[fence_id]
        if not fence.challengers:
            return None
        best = min(fence.challengers, key=lambda c: c.difficulty)
        return best.name

    def render_board(self) -> str:
        """Render ASCII fence board."""
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║  🎨 FENCE BOARD — Tom Sawyer Protocol                     ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        for fence in self.fences.values():
            status_icon = {
                FenceStatus.OPEN: "🟢",
                FenceStatus.CLAIMED: "🟡",
                FenceStatus.COMPLETED: "✅",
            }[fence.status]
            lines.append(f"║ {status_icon} {fence.id}: {fence.title[:40]:40} ║")
            if fence.claimed_by:
                lines.append(f"║    Claimed by: {fence.claimed_by[:20]:20} ║")
            if fence.badge:
                lines.append(f"║    Badge: {fence.badge:20} ║")
            lines.append("║" + " " * 58 + "║")

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "max_active": self.max_active,
            "fences": {fid: f.to_dict() for fid, f in self.fences.items()},
        }
