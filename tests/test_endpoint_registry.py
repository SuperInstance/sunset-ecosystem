import pytest
from fleet.endpoint_registry import EndpointRegistry, ServiceEndpoint


class TestServiceEndpoint:
    def test_to_dict(self):
        e = ServiceEndpoint("svc", "host", 8080, "http", "healthy", 0.0)
        d = e.to_dict()
        assert d["service_name"] == "svc"
        assert d["host"] == "host"

    def test_url(self):
        e = ServiceEndpoint("svc", "host", 8080, "http")
        assert e.url() == "http://host:8080"


class TestEndpointRegistry:
    def test_init(self):
        reg = EndpointRegistry()
        assert reg.fleet_node_id == "default"
        assert reg.get_stats()["total"] == 0

    def test_register(self):
        reg = EndpointRegistry()
        e = reg.register("breeder", "10.0.0.1", 8080)
        assert e.service_name == "breeder"
        assert e.health_status == "healthy"
        assert reg.get_stats()["total"] == 1

    def test_unregister(self):
        reg = EndpointRegistry()
        reg.register("svc", "host", 8080)
        assert reg.unregister("svc") is True
        assert reg.unregister("svc") is False

    def test_get(self):
        reg = EndpointRegistry()
        reg.register("svc", "host", 8080)
        e = reg.get("svc")
        assert e is not None
        assert e.port == 8080

    def test_get_missing(self):
        reg = EndpointRegistry()
        assert reg.get("missing") is None

    def test_get_all(self):
        reg = EndpointRegistry()
        reg.register("a", "host1", 8080)
        reg.register("b", "host2", 8081)
        assert len(reg.get_all()) == 2

    def test_get_healthy(self):
        reg = EndpointRegistry()
        reg.register("a", "host1", 8080)
        reg.update_health("a", "unhealthy")
        reg.register("b", "host2", 8081)
        healthy = reg.get_healthy()
        assert len(healthy) == 1
        assert healthy[0].service_name == "b"

    def test_update_health(self):
        reg = EndpointRegistry()
        reg.register("svc", "host", 8080)
        assert reg.update_health("svc", "degraded") is True
        assert reg.get("svc").health_status == "degraded"
        assert reg.update_health("missing", "healthy") is False

    def test_find_by_metadata(self):
        reg = EndpointRegistry()
        reg.register("a", "host1", 8080, metadata={"env": "prod"})
        reg.register("b", "host2", 8081, metadata={"env": "dev"})
        results = reg.find_by_metadata("env", "prod")
        assert len(results) == 1
        assert results[0].service_name == "a"

    def test_get_stats(self):
        reg = EndpointRegistry()
        reg.register("a", "host", 8080)
        reg.register("b", "host", 8081)
        reg.update_health("b", "unhealthy")
        stats = reg.get_stats()
        assert stats["total"] == 2
        assert stats["statuses"]["healthy"] == 1
        assert stats["statuses"]["unhealthy"] == 1

    def test_export_json(self):
        reg = EndpointRegistry()
        reg.register("svc", "host", 8080)
        j = reg.export_json()
        assert "svc" in j
        assert "endpoints" in j

    def test_to_dict(self):
        reg = EndpointRegistry()
        reg.register("svc", "host", 8080)
        d = reg.to_dict()
        assert d["stats"]["total"] == 1
