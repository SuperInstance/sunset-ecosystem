from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class DeploymentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Deployment:
    """A deployment record."""

    deployment_id: str
    version: str
    status: DeploymentStatus
    started_at: float
    finished_at: Optional[float] = None
    canary_percentage: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "version": self.version,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "canary_percentage": self.canary_percentage,
            "errors": self.errors,
        }


class CanaryDeployer:
    """
    Canary deployment for breeding strategies.

    Gradually rolls out new versions while monitoring health.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._deployments: Dict[str, Deployment] = {}
        self._current_version: Optional[str] = None
        self._health_check: Optional[callable] = None

    def set_health_check(self, func: callable):
        """Set a health check function."""
        self._health_check = func

    def deploy(self, version: str, target_percentage: float = 100.0) -> Deployment:
        """Start a new deployment."""
        deploy_id = f"deploy_{int(time.time() * 1000000)}"
        deploy = Deployment(
            deployment_id=deploy_id,
            version=version,
            status=DeploymentStatus.RUNNING,
            started_at=time.time(),
            canary_percentage=0.0,
        )
        self._deployments[deploy_id] = deploy

        # Simulate gradual rollout
        while deploy.canary_percentage < target_percentage:
            step = min(10.0, target_percentage - deploy.canary_percentage)
            deploy.canary_percentage += step

            # Check health after each step
            if self._health_check and not self._health_check():
                deploy.status = DeploymentStatus.FAILED
                deploy.errors.append(
                    f"Health check failed at {deploy.canary_percentage}%"
                )
                deploy.finished_at = time.time()
                return deploy

        deploy.status = DeploymentStatus.SUCCEEDED
        deploy.finished_at = time.time()
        self._current_version = version
        return deploy

    def rollback(self, deployment_id: str) -> bool:
        """Roll back a deployment."""
        deploy = self._deployments.get(deployment_id)
        if not deploy:
            return False
        deploy.status = DeploymentStatus.ROLLED_BACK
        deploy.finished_at = time.time()
        deploy.canary_percentage = 0.0
        return True

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get a deployment by ID."""
        return self._deployments.get(deployment_id)

    def get_all_deployments(self) -> List[Deployment]:
        """Get all deployments."""
        return list(self._deployments.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        statuses = {}
        for d in self._deployments.values():
            s = d.status.value
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total": len(self._deployments),
            "current_version": self._current_version,
            "statuses": statuses,
        }

    def export_json(self) -> str:
        """Export deployments as JSON."""
        return json.dumps(
            {
                "node": self.fleet_node_id,
                "current_version": self._current_version,
                "deployments": [d.to_dict() for d in self._deployments.values()],
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
