import pytest
from fleet.canary_deployer import CanaryDeployer, Deployment, DeploymentStatus


class TestDeployment:
    def test_to_dict(self):
        d = Deployment("d1", "v1.0", DeploymentStatus.RUNNING, 0.0)
        dd = d.to_dict()
        assert dd["version"] == "v1.0"
        assert dd["status"] == "running"


class TestCanaryDeployer:
    def test_init(self):
        cd = CanaryDeployer()
        assert cd.fleet_node_id == "default"
        assert cd.get_stats()["total"] == 0

    def test_deploy(self):
        cd = CanaryDeployer()
        deploy = cd.deploy("v1.0", target_percentage=100.0)
        assert deploy.status == DeploymentStatus.SUCCEEDED
        assert deploy.canary_percentage == 100.0
        assert cd.get_stats()["total"] == 1

    def test_deploy_with_health_check(self):
        cd = CanaryDeployer()
        cd.set_health_check(lambda: True)
        deploy = cd.deploy("v1.0", target_percentage=100.0)
        assert deploy.status == DeploymentStatus.SUCCEEDED

    def test_deploy_failing_health_check(self):
        cd = CanaryDeployer()
        cd.set_health_check(lambda: False)
        deploy = cd.deploy("v1.0", target_percentage=100.0)
        assert deploy.status == DeploymentStatus.FAILED
        assert len(deploy.errors) > 0

    def test_rollback(self):
        cd = CanaryDeployer()
        deploy = cd.deploy("v1.0")
        assert cd.rollback(deploy.deployment_id) is True
        assert deploy.status == DeploymentStatus.ROLLED_BACK

    def test_rollback_missing(self):
        cd = CanaryDeployer()
        assert cd.rollback("missing") is False

    def test_get_deployment(self):
        cd = CanaryDeployer()
        deploy = cd.deploy("v1.0")
        retrieved = cd.get_deployment(deploy.deployment_id)
        assert retrieved == deploy

    def test_get_all_deployments(self):
        cd = CanaryDeployer()
        cd.deploy("v1.0")
        cd.deploy("v2.0")
        assert len(cd.get_all_deployments()) == 2

    def test_get_stats(self):
        cd = CanaryDeployer()
        cd.deploy("v1.0")
        stats = cd.get_stats()
        assert stats["total"] == 1
        assert stats["current_version"] == "v1.0"

    def test_export_json(self):
        cd = CanaryDeployer()
        cd.deploy("v1.0")
        j = cd.export_json()
        assert "v1.0" in j
        assert "deployments" in j

    def test_to_dict(self):
        cd = CanaryDeployer()
        cd.deploy("v1.0")
        d = cd.to_dict()
        assert d["stats"]["total"] == 1
