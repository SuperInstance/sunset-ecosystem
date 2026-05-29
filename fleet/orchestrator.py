"""Fleet workload orchestrator with scheduling and resource allocation.

Schedules tasks across fleet nodes based on resource availability,
priorities, and constraints. Used for fleet-wide job scheduling and
resource-aware task placement.

Usage:
    orch = FleetOrchestrator()
    orch.add_node("node-1", {"cpu": 4, "mem": 16})
    orch.submit_task("job-1", {"cpu": 2, "mem": 4}, priority=5)
    assignment = orch.schedule()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeResources:
    """Resource capacities of a fleet node."""

    node_id: str
    total: Dict[str, float]
    used: Dict[str, float] = field(default_factory=dict)

    def available(self, resource: str) -> float:
        return self.total.get(resource, 0) - self.used.get(resource, 0)

    def allocate(self, resources: Dict[str, float]) -> bool:
        for r, amount in resources.items():
            if self.available(r) < amount:
                return False
        for r, amount in resources.items():
            self.used[r] = self.used.get(r, 0) + amount
        return True

    def free(self, resources: Dict[str, float]) -> None:
        for r, amount in resources.items():
            self.used[r] = max(0, self.used.get(r, 0) - amount)


@dataclass
class Task:
    """A schedulable task."""

    task_id: str
    resources: Dict[str, float]
    priority: int = 0
    constraints: Dict[str, Any] = field(default_factory=dict)


class FleetOrchestrator:
    """
    Fleet workload orchestrator.

    :param default_resources: Default resource types to track.
    """

    def __init__(self, default_resources: Optional[List[str]] = None):
        self._default_resources = default_resources or ["cpu", "mem"]
        self._nodes: Dict[str, NodeResources] = {}
        self._tasks: Dict[str, Task] = {}
        self._assignments: Dict[str, str] = {}  # task_id -> node_id

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, resources: Dict[str, float]) -> None:
        """Register a fleet node with its resources."""
        self._nodes[node_id] = NodeResources(
            node_id=node_id,
            total=dict(resources),
        )

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and free assigned tasks."""
        if node_id not in self._nodes:
            return False
        # Free any tasks on this node
        tasks_to_reschedule = [
            tid for tid, nid in self._assignments.items() if nid == node_id
        ]
        for tid in tasks_to_reschedule:
            del self._assignments[tid]
        del self._nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[NodeResources]:
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit_task(
        self,
        task_id: str,
        resources: Dict[str, float],
        priority: int = 0,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Submit a task for scheduling."""
        self._tasks[task_id] = Task(
            task_id=task_id,
            resources=resources,
            priority=priority,
            constraints=constraints or {},
        )

    def remove_task(self, task_id: str) -> bool:
        """Remove a task and free its resources."""
        task = self._tasks.pop(task_id, None)
        if not task:
            return False
        node_id = self._assignments.pop(task_id, None)
        if node_id:
            node = self._nodes.get(node_id)
            if node:
                node.free(task.resources)
        return True

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> Dict[str, str]:
        """
        Schedule all pending tasks to nodes.

        Returns {task_id: node_id} assignments.
        """
        new_assignments: Dict[str, str] = {}
        # Sort by priority (highest first)
        pending = sorted(
            [t for t in self._tasks.values() if t.task_id not in self._assignments],
            key=lambda t: -t.priority,
        )
        for task in pending:
            node_id = self._find_best_node(task)
            if node_id:
                node = self._nodes[node_id]
                if node.allocate(task.resources):
                    self._assignments[task.task_id] = node_id
                    new_assignments[task.task_id] = node_id
        return new_assignments

    def _find_best_node(self, task: Task) -> Optional[str]:
        """Find the best node for a task (most available resources)."""
        candidates: List[str] = []
        for node_id, node in self._nodes.items():
            if self._satisfies_constraints(task, node_id):
                if all(
                    node.available(r) >= amount
                    for r, amount in task.resources.items()
                ):
                    candidates.append(node_id)
        if not candidates:
            return None
        # Pick node with most available resources (best fit)
        return max(
            candidates,
            key=lambda nid: sum(self._nodes[nid].available(r) for r in task.resources),
        )

    def _satisfies_constraints(self, task: Task, node_id: str) -> bool:
        """Check if task constraints are satisfied by node."""
        for key, value in task.constraints.items():
            if key == "node_id":
                if node_id != value:
                    return False
            elif key == "node_id_in":
                if node_id not in value:
                    return False
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_assignment(self, task_id: str) -> Optional[str]:
        """Get node assigned to a task."""
        return self._assignments.get(task_id)

    def node_tasks(self, node_id: str) -> List[str]:
        """Get tasks assigned to a node."""
        return [
            tid for tid, nid in self._assignments.items() if nid == node_id
        ]

    def unassigned_tasks(self) -> List[str]:
        """Get tasks not yet assigned to any node."""
        return [tid for tid in self._tasks if tid not in self._assignments]

    def node_utilization(self, node_id: str) -> Dict[str, float]:
        """Get resource utilization ratios."""
        node = self._nodes.get(node_id)
        if not node:
            return {}
        return {
            r: node.used.get(r, 0) / node.total.get(r, 1)
            for r in node.total
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "nodes": len(self._nodes),
            "tasks": len(self._tasks),
            "assigned": len(self._assignments),
            "unassigned": len(self.unassigned_tasks()),
        }

    def __repr__(self) -> str:
        return (
            f"<FleetOrchestrator nodes={len(self._nodes)} "
            f"tasks={len(self._tasks)} assigned={len(self._assignments)}>"
        )
