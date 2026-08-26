"""fleet/fleet_task_board_bridge.py — Fleet Task Board pattern integration.

Brings the oracle1-vessel task board pattern into sunset-ecosystem:
- CRITICAL / HIGH / MEDIUM / LOW priority levels
- Capability tags ([c], [python], [rust], [testing])
- Owner assignment
- T-minus-event estimates (T-24h, T-0h)
- Status tracking (open/claimed/done)
- Auto-discovery of task board entries from task sources

Usage:
    from fleet.fleet_task_board_bridge import FleetTaskBoard, TaskPriority

    board = FleetTaskBoard()
    board.add_task("Conformance", TaskPriority.CRITICAL, ["c", "python"], owner="JC1")
    board.set_eta("task-1", "T-24h")
    board.complete_task("task-1", commit_hash="abc123")
    print(board.render_text())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from datetime import datetime, timezone


class TaskPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


class TaskStatus(Enum):
    OPEN = auto()
    CLAIMED = auto()
    DONE = auto()


@dataclass
class Task:
    """A single task on the fleet task board."""

    id: str
    title: str
    priority: TaskPriority
    tags: list[str]  # capability tags
    owner: Optional[str] = None
    status: TaskStatus = TaskStatus.OPEN
    eta: str = ""  # T-24h, T-0h, etc.
    description: str = ""
    commit_hash: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    blocked_by: Optional[str] = None  # task ID that blocks this one

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority.name,
            "tags": self.tags,
            "owner": self.owner,
            "status": self.status.name,
            "eta": self.eta,
            "description": self.description,
            "commit_hash": self.commit_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "blocked_by": self.blocked_by,
        }


class FleetTaskBoard:
    """
    Fleet task board with priority levels and capability tags.

    Mirrors the oracle1-vessel TASK-BOARD.md pattern.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"task-{self._counter}"

    def add_task(
        self,
        title: str,
        priority: TaskPriority,
        tags: list[str],
        owner: Optional[str] = None,
        description: str = "",
        blocked_by: Optional[str] = None,
    ) -> Task:
        """Add a new task to the board."""
        task_id = self._next_id()
        task = Task(
            id=task_id,
            title=title,
            priority=priority,
            tags=tags,
            owner=owner,
            description=description,
            blocked_by=blocked_by,
        )
        self.tasks[task_id] = task
        return task

    def claim_task(self, task_id: str, owner: str) -> Task:
        """Claim a task for an owner."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        task = self.tasks[task_id]
        if task.status != TaskStatus.OPEN:
            raise ValueError(f"Task {task_id} is not open")
        task.status = TaskStatus.CLAIMED
        task.owner = owner
        return task

    def set_eta(self, task_id: str, eta: str) -> Task:
        """Set a t-minus estimate."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        self.tasks[task_id].eta = eta
        return self.tasks[task_id]

    def complete_task(self, task_id: str, commit_hash: Optional[str] = None) -> Task:
        """Mark a task as done."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        task = self.tasks[task_id]
        if task.status == TaskStatus.DONE:
            raise ValueError(f"Task {task_id} is already done")
        task.status = TaskStatus.DONE
        task.completed_at = datetime.now(timezone.utc)
        task.commit_hash = commit_hash
        return task

    def unblock_task(self, task_id: str) -> Task:
        """Unblock a task (clear blocked_by)."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")
        self.tasks[task_id].blocked_by = None
        return self.tasks[task_id]

    def by_priority(self) -> list[Task]:
        """Return tasks sorted by priority (CRITICAL first)."""
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }
        return sorted(
            self.tasks.values(), key=lambda t: priority_order.get(t.priority, 99)
        )

    def by_owner(self, owner: str) -> list[Task]:
        """Return tasks for a specific owner."""
        return [t for t in self.tasks.values() if t.owner == owner]

    def critical_path(self) -> list[Task]:
        """Return CRITICAL tasks sorted by creation time."""
        return [
            t
            for t in self.by_priority()
            if t.priority == TaskPriority.CRITICAL and t.status != TaskStatus.DONE
        ]

    def ready_tasks(self) -> list[Task]:
        """Return tasks that are not blocked and not done."""
        return [
            t
            for t in self.tasks.values()
            if t.blocked_by is None and t.status != TaskStatus.DONE
        ]

    def render_text(self) -> str:
        """Render task board as text."""
        lines = ["🔮 FLUX Fleet Task Board", "=" * 50]
        priority_icon = {
            TaskPriority.CRITICAL: "🔴",
            TaskPriority.HIGH: "🟠",
            TaskPriority.MEDIUM: "🟡",
            TaskPriority.LOW: "🟢",
        }
        status_icon = {
            TaskStatus.OPEN: " ",
            TaskStatus.CLAIMED: "►",
            TaskStatus.DONE: "✓",
        }
        for task in self.by_priority():
            icon = priority_icon.get(task.priority, "⚪")
            sicon = status_icon.get(task.status, " ")
            tags = " ".join(f"[{t}]" for t in task.tags)
            owner = f" @{task.owner}" if task.owner else ""
            eta = f" {task.eta}" if task.eta else ""
            blocked = f" [BLOCKED by {task.blocked_by}]" if task.blocked_by else ""
            lines.append(f"{icon} {sicon} {task.title}{owner}{eta}")
            lines.append(f"   {tags}{blocked}")
            if task.commit_hash:
                lines.append(f"   Commit: {task.commit_hash}")
        return "\n".join(lines)

    def render_org_chart(self) -> str:
        """Render fleet org chart from owner assignments."""
        owners: dict[str, list[Task]] = {}
        for task in self.tasks.values():
            if task.owner:
                owners.setdefault(task.owner, []).append(task)
        lines = ["Captain Casey", "  └── Oracle1 🔮 (Managing Director)"]
        for owner, tasks in owners.items():
            if owner == "Casey" or owner == "Oracle1":
                continue
            active = len([t for t in tasks if t.status != TaskStatus.DONE])
            lines.append(f"      ├── {owner} — {active} active")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
        }
