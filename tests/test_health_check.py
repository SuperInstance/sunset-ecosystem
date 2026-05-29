import pytest
from fleet.health_check import (
    HealthCheck,
    HealthCheckSystem,
    HealthStatus,
)


class TestHealthCheck:
    def test_to_dict(self):
        h = HealthCheck(
            name="test",
            status=HealthStatus.HEALTHY,
            response_time_ms=10.0,
            message="OK",
            timestamp=0.0,
        )
        d = h.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "healthy"


class TestHealthCheckSystem:
    def test_init(self):
        h = HealthCheckSystem()
        assert h.fleet_node_id == "default"
        assert h.get_overall_status() == HealthStatus.UNKNOWN

    def test_register_and_check(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: ("healthy", "up"))
        result = h.check("svc")
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "up"

    def test_check_unhealthy(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: (_ for _ in ()).throw(ValueError("down")))
        result = h.check("svc")
        assert result.status == HealthStatus.UNHEALTHY

    def test_check_all(self):
        h = HealthCheckSystem()
        h.register("a", lambda: ("healthy", "up"))
        h.register("b", lambda: ("healthy", "up"))
        results = h.check_all()
        assert len(results) == 2
        assert all(r.status == HealthStatus.HEALTHY for r in results.values())

    def test_dependencies(self):
        h = HealthCheckSystem()
        h.register("db", lambda: ("healthy", "up"))
        h.register("app", lambda: ("healthy", "up"), dependencies=["db"])
        results = h.check_all()
        assert "db" in results
        assert "app" in results

    def test_overall_healthy(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: ("healthy", "up"))
        h.check_all()
        assert h.get_overall_status() == HealthStatus.HEALTHY

    def test_overall_unhealthy(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: ("unhealthy", "down"))
        h.check_all()
        assert h.get_overall_status() == HealthStatus.UNHEALTHY

    def test_overall_degraded(self):
        h = HealthCheckSystem()
        h.register("a", lambda: ("healthy", "up"))
        h.register("b", lambda: ("degraded", "slow"))
        h.check_all()
        assert h.get_overall_status() == HealthStatus.DEGRADED

    def test_get_dependents(self):
        h = HealthCheckSystem()
        h.register("db", lambda: ("healthy", "up"))
        h.register("app", lambda: ("healthy", "up"), dependencies=["db"])
        dependents = h.get_dependents("db")
        assert "app" in dependents

    def test_get_stats(self):
        h = HealthCheckSystem()
        h.register("a", lambda: ("healthy", "up"))
        h.register("b", lambda: ("degraded", "slow"))
        h.check_all()
        stats = h.get_stats()
        assert stats["overall"] == "degraded"
        assert stats["checks"] == 2
        assert stats["healthy"] == 1
        assert stats["degraded"] == 1

    def test_export_json(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: ("healthy", "up"))
        h.check_all()
        j = h.export_json()
        assert "svc" in j
        assert "overall" in j

    def test_to_dict(self):
        h = HealthCheckSystem()
        h.register("svc", lambda: ("healthy", "up"))
        h.check_all()
        d = h.to_dict()
        assert d["stats"]["checks"] == 1
