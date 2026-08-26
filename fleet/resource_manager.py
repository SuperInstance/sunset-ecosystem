from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ResourceAllocation:
    """Resource allocation for a breeding task."""

    task_id: str
    cpu_cores: float
    memory_mb: float
    gpu_devices: int
    allocated_at: float
    released_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_devices": self.gpu_devices,
            "allocated_at": self.allocated_at,
            "released_at": self.released_at,
        }


class ResourceManager:
    """
    Resource manager for fleet breeding tasks.

    Tracks CPU, memory, GPU allocations and enforces limits.
    """

    def __init__(
        self,
        fleet_node_id: str = "default",
        total_cpu: float = 8.0,
        total_memory_mb: float = 32768.0,
        total_gpu: int = 0,
    ):
        self.fleet_node_id = fleet_node_id
        self.total_cpu = total_cpu
        self.total_memory_mb = total_memory_mb
        self.total_gpu = total_gpu
        self._allocations: Dict[str, ResourceAllocation] = {}

    def allocate(
        self,
        task_id: str,
        cpu_cores: float = 1.0,
        memory_mb: float = 1024.0,
        gpu_devices: int = 0,
    ) -> Optional[ResourceAllocation]:
        """Allocate resources for a task."""
        used_cpu = sum(a.cpu_cores for a in self._allocations.values())
        used_mem = sum(a.memory_mb for a in self._allocations.values())
        used_gpu = sum(a.gpu_devices for a in self._allocations.values())

        if used_cpu + cpu_cores > self.total_cpu:
            return None
        if used_mem + memory_mb > self.total_memory_mb:
            return None
        if used_gpu + gpu_devices > self.total_gpu:
            return None

        alloc = ResourceAllocation(
            task_id=task_id,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_devices=gpu_devices,
            allocated_at=time.time(),
        )
        self._allocations[task_id] = alloc
        return alloc

    def release(self, task_id: str) -> bool:
        """Release resources for a task."""
        if task_id not in self._allocations:
            return False
        self._allocations[task_id].released_at = time.time()
        del self._allocations[task_id]
        return True

    def get_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        return {
            "cpu_cores": sum(a.cpu_cores for a in self._allocations.values()),
            "memory_mb": sum(a.memory_mb for a in self._allocations.values()),
            "gpu_devices": sum(a.gpu_devices for a in self._allocations.values()),
        }

    def get_available(self) -> Dict[str, float]:
        """Get available resources."""
        used = self.get_usage()
        return {
            "cpu_cores": self.total_cpu - used["cpu_cores"],
            "memory_mb": self.total_memory_mb - used["memory_mb"],
            "gpu_devices": self.total_gpu - used["gpu_devices"],
        }

    def get_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages."""
        used = self.get_usage()
        return {
            "cpu": used["cpu_cores"] / self.total_cpu if self.total_cpu > 0 else 0.0,
            "memory": used["memory_mb"] / self.total_memory_mb
            if self.total_memory_mb > 0
            else 0.0,
            "gpu": used["gpu_devices"] / self.total_gpu if self.total_gpu > 0 else 0.0,
        }

    def get_allocations(self) -> List[ResourceAllocation]:
        """Get all active allocations."""
        return list(self._allocations.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.fleet_node_id,
            "total": {
                "cpu": self.total_cpu,
                "memory_mb": self.total_memory_mb,
                "gpu": self.total_gpu,
            },
            "usage": self.get_usage(),
            "available": self.get_available(),
            "utilization": self.get_utilization(),
            "active_allocations": len(self._allocations),
        }
