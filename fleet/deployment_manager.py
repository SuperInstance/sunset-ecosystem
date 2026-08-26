"""deployment_manager.py — Rolling deployment with health checks.

Provides:
1. Rolling update strategy (one-by-one or batch)
2. Health check gates between deployments
3. Automatic rollback on failure
4. Deployment history and status tracking
5. Canary deployment support

Usage:
    dm = DeploymentManager(health_checker=check_node)
    dm.deploy(nodes=["n1", "n2", "n3"], batch_size=1, health_wait=30.0)
"""

from __future__ import annotations

__all__ = [
    "DeploymentManager",
    "Deployment",
    "DeploymentStrategy",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Deployment:
    """A deployment record."""

    deployment_id: str
    nodes: list[str]
    strategy: str
    status: str = "pending"  # pending, running, success, failed, rolled_back
    started_at: float = 0.0
    finished_at: float | None = None
    node_status: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class DeploymentStrategy:
    """Deployment strategy configuration."""

    batch_size: int = 1
    health_wait: float = 30.0
    max_failures: int = 1
    rollback_on_failure: bool = True
    canary: bool = False
    canary_count: int = 1


class DeploymentManager:
    """Rolling deployment manager with health gates and rollback."""

    def __init__(
        self,
        deploy_fn: Callable[[str], bool],
        health_checker: Callable[[str], bool],
        rollback_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._deploy_fn = deploy_fn
        self._health_checker = health_checker
        self._rollback_fn = rollback_fn
        self._deployments: list[Deployment] = []
        self._counter = 0

    def deploy(
        self,
        nodes: list[str],
        strategy: DeploymentStrategy | None = None,
    ) -> Deployment:
        """Execute a deployment."""
        strat = strategy or DeploymentStrategy()
        self._counter += 1
        dep = Deployment(
            deployment_id=f"deploy-{self._counter}",
            nodes=list(nodes),
            strategy="rolling" if not strat.canary else "canary",
            started_at=time.time(),
        )
        self._deployments.append(dep)

        try:
            if strat.canary:
                self._run_canary(dep, strat)
            else:
                self._run_rolling(dep, strat)
        except Exception as e:
            dep.status = "failed"
            dep.errors.append(str(e))
            logger.error(f"Deployment {dep.deployment_id} failed: {e}")
            if strat.rollback_on_failure and self._rollback_fn:
                self._rollback(dep)

        dep.finished_at = time.time()
        return dep

    def _run_rolling(self, dep: Deployment, strat: DeploymentStrategy) -> None:
        dep.status = "running"
        failures = 0
        for i in range(0, len(dep.nodes), strat.batch_size):
            batch = dep.nodes[i : i + strat.batch_size]
            for node in batch:
                success = self._deploy_node(node, dep, strat)
                if not success:
                    failures += 1
                    if failures >= strat.max_failures:
                        dep.status = "failed"
                        if strat.rollback_on_failure and self._rollback_fn:
                            self._rollback(dep)
                        return

        dep.status = "success"

    def _run_canary(self, dep: Deployment, strat: DeploymentStrategy) -> None:
        dep.status = "running"
        canary_nodes = dep.nodes[: strat.canary_count]
        rest = dep.nodes[strat.canary_count :]

        for node in canary_nodes:
            if not self._deploy_node(node, dep, strat):
                dep.status = "failed"
                if strat.rollback_on_failure and self._rollback_fn:
                    self._rollback(dep)
                return

        for node in rest:
            if not self._deploy_node(node, dep, strat):
                dep.status = "failed"
                if strat.rollback_on_failure and self._rollback_fn:
                    self._rollback(dep)
                return

        dep.status = "success"

    def _deploy_node(
        self, node: str, dep: Deployment, strat: DeploymentStrategy
    ) -> bool:
        logger.info(f"Deploying to {node}...")
        try:
            if not self._deploy_fn(node):
                dep.node_status[node] = "deploy_failed"
                dep.errors.append(f"Deploy failed on {node}")
                return False

            dep.node_status[node] = "deployed"

            # Health check gate
            if strat.health_wait > 0:
                healthy = self._wait_for_health(node, strat.health_wait)
                if not healthy:
                    dep.node_status[node] = "unhealthy"
                    dep.errors.append(f"Health check failed on {node}")
                    return False

            dep.node_status[node] = "healthy"
            return True
        except Exception as e:
            dep.node_status[node] = "error"
            dep.errors.append(f"Exception on {node}: {e}")
            return False

    def _wait_for_health(self, node: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._health_checker(node):
                return True
            time.sleep(1.0)
        return False

    def _rollback(self, dep: Deployment) -> None:
        logger.warning(f"Rolling back deployment {dep.deployment_id}")
        dep.status = "rolled_back"
        for node, status in dep.node_status.items():
            if status in ("deployed", "healthy", "unhealthy"):
                try:
                    if self._rollback_fn:
                        self._rollback_fn(node)
                except Exception as e:
                    logger.error(f"Rollback failed for {node}: {e}")

    def list_deployments(self, status: str | None = None) -> list[Deployment]:
        """List deployments, optionally filtered by status."""
        deps = list(self._deployments)
        if status:
            deps = [d for d in deps if d.status == status]
        return deps

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        """Get deployment by ID."""
        for d in self._deployments:
            if d.deployment_id == deployment_id:
                return d
        return None

    def stats(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for d in self._deployments:
            statuses[d.status] = statuses.get(d.status, 0) + 1
        return {
            "total": len(self._deployments),
            "status_breakdown": statuses,
        }

    def __repr__(self) -> str:
        return f"DeploymentManager(deployments={len(self._deployments)})"
