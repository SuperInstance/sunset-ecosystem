"""Tests for deployment_manager.py — Rolling deployment with health checks.

Run: python3 -m pytest tests/test_deployment_manager.py -v --tb=short
"""
from __future__ import annotations

import time

import pytest

from fleet.deployment_manager import DeploymentManager, DeploymentStrategy


class TestDeploymentManager:
    def test_successful_rolling(self):
        deployed = []
        dm = DeploymentManager(
            deploy_fn=lambda n: deployed.append(n) or True,
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1", "n2", "n3"])
        assert dep.status == "success"
        assert set(deployed) == {"n1", "n2", "n3"}

    def test_batch_size(self):
        deployed = []
        dm = DeploymentManager(
            deploy_fn=lambda n: deployed.append(n) or True,
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1", "n2", "n3"], strategy=DeploymentStrategy(batch_size=2))
        assert dep.status == "success"
        assert len(deployed) == 3

    def test_deploy_failure(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: n != "n2",
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1", "n2", "n3"])
        assert dep.status == "failed"
        assert "n2" in dep.errors[0]

    def test_health_check_failure(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: True,
            health_checker=lambda n: n != "n2",
        )
        dep = dm.deploy(["n1", "n2", "n3"])
        assert dep.status == "failed"
        assert any("n2" in e for e in dep.errors)

    def test_rollback(self):
        rolled_back = []
        dm = DeploymentManager(
            deploy_fn=lambda n: n != "n2",
            health_checker=lambda n: True,
            rollback_fn=lambda n: rolled_back.append(n),
        )
        dep = dm.deploy(["n1", "n2", "n3"])
        assert dep.status == "rolled_back"
        assert "n1" in rolled_back

    def test_canary(self):
        deployed = []
        dm = DeploymentManager(
            deploy_fn=lambda n: deployed.append(n) or True,
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1", "n2", "n3"], strategy=DeploymentStrategy(canary=True, canary_count=1))
        assert dep.status == "success"
        assert deployed[0] == "n1"  # canary first

    def test_node_status(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: True,
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1", "n2"])
        assert dep.node_status["n1"] == "healthy"
        assert dep.node_status["n2"] == "healthy"

    def test_list_deployments(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: True,
            health_checker=lambda n: True,
        )
        dm.deploy(["n1"])
        dm.deploy(["n2"])
        assert len(dm.list_deployments()) == 2
        assert len(dm.list_deployments(status="success")) == 2

    def test_get_deployment(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: True,
            health_checker=lambda n: True,
        )
        dep = dm.deploy(["n1"])
        found = dm.get_deployment(dep.deployment_id)
        assert found is not None
        assert found.deployment_id == dep.deployment_id
        assert dm.get_deployment("missing") is None

    def test_stats(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: True,
            health_checker=lambda n: True,
        )
        dm.deploy(["n1"])
        dm.deploy(["n2"], strategy=DeploymentStrategy(rollback_on_failure=False))
        stats = dm.stats()
        assert stats["total"] == 2
        assert stats["status_breakdown"]["success"] == 2

    def test_no_rollback_fn(self):
        dm = DeploymentManager(
            deploy_fn=lambda n: False,
            health_checker=lambda n: True,
            rollback_fn=None,
        )
        dep = dm.deploy(["n1"])
        assert dep.status == "failed"  # Not rolled_back because no rollback_fn

    def test_repr(self):
        dm = DeploymentManager(deploy_fn=lambda n: True, health_checker=lambda n: True)
        assert "DeploymentManager" in repr(dm)