"""tests/test_health_bridge.py — Test suite for the cocapn-health bridge.

Covers:
- CheckResult dataclass construction and serialization
- ServiceDef URL building
- HTTP endpoint checking (mocked)
- TCP port checking
- System checks (disk, memory, CPU)
- Fleet-wide checking
- Markdown/JSON/oneline reporting
- EventBusHealthChecker transition detection
- HealthCache TTL behavior
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from fleet.health_bridge import (
    CheckResult,
    ServiceDef,
    HealthChecker,
    EventBusHealthChecker,
    HealthCache,
    FLEET_SERVICES,
    _extract_json_path,
)


class TestCheckResult:
    def test_construction(self):
        r = CheckResult(name="test", ok=True, latency_ms=23.1, status="UP")
        assert r.name == "test"
        assert r.ok is True
        assert r.latency_ms == 23.1
        assert r.status == "UP"
        assert r.details == {}

    def test_to_dict(self):
        r = CheckResult(name="test", ok=True, latency_ms=23.1, status="UP", details={"a": 1})
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["ok"] is True
        assert d["latency_ms"] == 23.1
        assert d["details"] == {"a": 1}

    def test_from_dict_roundtrip(self):
        r = CheckResult(name="test", ok=True, latency_ms=23.1, status="UP", details={"a": 1})
        d = r.to_dict()
        r2 = CheckResult.from_dict(d)
        assert r2.name == r.name
        assert r2.ok == r.ok
        assert r2.latency_ms == r.latency_ms
        assert r2.status == r.status
        assert r2.details == r.details


class TestServiceDef:
    def test_url_construction(self):
        svc = ServiceDef(name="test", host="127.0.0.1", port=8080, path="/health")
        assert svc.url() == "http://127.0.0.1:8080/health"

    def test_url_root(self):
        svc = ServiceDef(name="test", host="127.0.0.1", port=8080, path="/")
        assert svc.url() == "http://127.0.0.1:8080/"

    def test_default_timeout(self):
        svc = ServiceDef(name="test", host="127.0.0.1", port=8080)
        assert svc.timeout == 5.0

    def test_extract_field(self):
        svc = ServiceDef(
            name="test", host="127.0.0.1", port=8080,
            extract={"rooms": "rooms"}
        )
        assert svc.extract == {"rooms": "rooms"}


class TestExtractJsonPath:
    def test_simple_key(self):
        data = {"rooms": 42}
        assert _extract_json_path(data, "rooms") == 42

    def test_nested_path(self):
        data = {"status": {"rooms": 42}}
        assert _extract_json_path(data, "status.rooms") == 42

    def test_missing_key(self):
        data = {"a": 1}
        assert _extract_json_path(data, "b") is None

    def test_missing_nested(self):
        data = {"a": {"b": 1}}
        assert _extract_json_path(data, "a.c") is None


class TestHealthCheckerCheckHttp:
    def test_success_200(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.getcode.return_value = 200
            mock_resp.read.return_value = b'{"rooms": 42}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = HealthChecker.check_http("http://test.example/")
            assert result.ok is True
            assert result.status.startswith("UP")
            assert "status_code" in result.details
            assert result.details["status_code"] == 200

    def test_success_with_extract(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.getcode.return_value = 200
            mock_resp.read.return_value = b'{"rooms": 42}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = HealthChecker.check_http(
                "http://test.example/",
                extract={"rooms": "rooms"}
            )
            assert result.ok is True
            assert result.details.get("rooms") == 42

    def test_404_treated_as_up(self):
        """HTTP 404 from a live server is treated as UP."""
        from urllib.error import HTTPError
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                "http://test.example/", 404, "Not Found", {}, None
            )
            result = HealthChecker.check_http("http://test.example/")
            assert result.ok is True
            assert "404" in result.status

    def test_connection_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")
            result = HealthChecker.check_http("http://localhost:59999/")
            assert result.ok is False
            assert "DOWN" in result.status

    def test_expect_status_mismatch(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.getcode.return_value = 200
            mock_resp.read.return_value = b'{}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = HealthChecker.check_http(
                "http://test.example/",
                expect_status=201
            )
            assert result.ok is False
            assert "DEGRADED" in result.status


class TestHealthCheckerCheckTcp:
    def test_open_port(self):
        with patch("socket.create_connection") as mock_conn:
            result = HealthChecker.check_tcp("127.0.0.1", 8080)
            assert result.ok is True
            assert "UP" in result.status
            mock_conn.assert_called_once()

    def test_closed_port(self):
        with patch("socket.create_connection") as mock_conn:
            mock_conn.side_effect = Exception("Connection refused")
            result = HealthChecker.check_tcp("127.0.0.1", 59999)
            assert result.ok is False
            assert "DOWN" in result.status


class TestHealthCheckerSystemChecks:
    def test_check_disk(self):
        result = HealthChecker.check_disk("/", min_percent_free=1.0)
        assert result.name == "/"
        assert isinstance(result.ok, bool)
        assert "free" in result.status
        assert "total_gb" in result.details
        assert "percent_free" in result.details

    def test_check_memory(self):
        result = HealthChecker.check_memory(min_percent_free=1.0)
        assert result.name == "memory"
        assert isinstance(result.ok, bool)
        assert "available" in result.status
        assert "total_mb" in result.details

    def test_check_cpu(self):
        result = HealthChecker.check_cpu(max_percent=99.0)
        assert result.name == "cpu"
        assert isinstance(result.ok, bool)
        assert "load" in result.status
        assert "load_1m" in result.details
        assert "cpu_count" in result.details

    def test_check_system(self):
        results = HealthChecker.check_system()
        assert len(results) == 3
        names = [r.name for r in results]
        assert "/" in names
        assert "memory" in names
        assert "cpu" in names


class TestHealthCheckerFleet:
    def test_check_all_fleet(self):
        checker = HealthChecker(FLEET_SERVICES)

        def make_result(*args, **kwargs):
            return CheckResult(name="test", ok=True, latency_ms=10.0, status="UP")

        with patch("fleet.health_bridge.HealthChecker.check_http") as mock_check:
            mock_check.side_effect = make_result
            results = checker.check_all()
            assert len(results) == len(FLEET_SERVICES)
            for r, svc in zip(results, FLEET_SERVICES):
                assert r.name == svc.name
            assert mock_check.call_count == len(FLEET_SERVICES)

    def test_check_one(self):
        checker = HealthChecker(FLEET_SERVICES)
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name="test", ok=True, latency_ms=10.0, status="UP"
            )
            result = checker.check_one(FLEET_SERVICES[0])
            assert result.name == FLEET_SERVICES[0].name


class TestHealthCheckerReporting:
    def test_report_markdown(self):
        results = [
            CheckResult(name="a", ok=True, latency_ms=10.0, status="UP"),
            CheckResult(name="b", ok=False, latency_ms=1000.0, status="DOWN"),
        ]
        report = HealthChecker.report(results, format="md")
        assert "Fleet Health Report" in report
        assert "1/2 services UP" in report
        assert "🟢 a" in report
        assert "🔴 b" in report

    def test_report_json(self):
        results = [
            CheckResult(name="a", ok=True, latency_ms=10.0, status="UP"),
            CheckResult(name="b", ok=False, latency_ms=1000.0, status="DOWN"),
        ]
        report = HealthChecker.report(results, format="json")
        data = json.loads(report)
        assert data["summary"]["total"] == 2
        assert data["summary"]["up"] == 1
        assert data["summary"]["down"] == 1
        assert len(data["services"]) == 2

    def test_report_oneline(self):
        results = [
            CheckResult(name="a", ok=True, latency_ms=10.0, status="UP"),
            CheckResult(name="b", ok=False, latency_ms=1000.0, status="DOWN"),
        ]
        report = HealthChecker.report(results, format="oneline")
        assert "Fleet: 1/2 up, 1 down" in report


class TestEventBusHealthChecker:
    def test_emits_down_transition(self):
        bus = MagicMock()
        checker = EventBusHealthChecker(FLEET_SERVICES[:1], bus=bus)
        # First check: UP
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            checker.check_all()
        assert not bus.emit.called

        # Second check: DOWN
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=False, latency_ms=1000.0, status="DOWN"
            )
            checker.check_all()
        bus.emit.assert_called_once()
        args = bus.emit.call_args[0]
        assert args[0] == "service_down"

    def test_emits_recovered_transition(self):
        bus = MagicMock()
        checker = EventBusHealthChecker(FLEET_SERVICES[:1], bus=bus)
        # First check: DOWN
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=False, latency_ms=1000.0, status="DOWN"
            )
            checker.check_all()

        # Second check: UP
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            checker.check_all()
        # Should emit service_recovered
        calls = [call for call in bus.emit.call_args_list if call[0][0] == "service_recovered"]
        assert len(calls) == 1

    def test_emit_on_every_check(self):
        bus = MagicMock()
        checker = EventBusHealthChecker(
            FLEET_SERVICES[:1], bus=bus, emit_on_every_check=True
        )
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            checker.check_all()
        bus.emit.assert_called_once()
        args = bus.emit.call_args[0]
        assert args[0] == "fleet_health"

    def test_no_bus_no_crash(self):
        checker = EventBusHealthChecker(FLEET_SERVICES[:1], bus=None)
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            result = checker.check_all()
        assert len(result) == 1

    def test_publish_interface(self):
        class FakeBus:
            def __init__(self):
                self.calls = []
            def publish(self, event_type, payload):
                self.calls.append((event_type, payload))
        bus = FakeBus()
        checker = EventBusHealthChecker(FLEET_SERVICES[:1], bus=bus)
        # First check: DOWN (to trigger transition on next check)
        with patch("fleet.health_bridge.HealthChecker.check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=False, latency_ms=1000.0, status="DOWN"
            )
            checker.check_all()
        # Second check: UP → recovered
        with patch("fleet.health_bridge.HealthChecker.check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            checker.check_all()
        assert len(bus.calls) >= 1
        assert bus.calls[-1][0] == "service_recovered"

    def test_send_interface(self):
        class FakeBus:
            def __init__(self):
                self.calls = []
            def send(self, event_type, payload):
                self.calls.append((event_type, payload))
        bus = FakeBus()
        checker = EventBusHealthChecker(FLEET_SERVICES[:1], bus=bus)
        # First check: DOWN
        with patch("fleet.health_bridge.HealthChecker.check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=False, latency_ms=1000.0, status="DOWN"
            )
            checker.check_all()
        # Second check: UP → recovered
        with patch("fleet.health_bridge.HealthChecker.check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            checker.check_all()
        assert len(bus.calls) >= 1
        assert bus.calls[-1][0] == "service_recovered"


class TestHealthCache:
    def test_cache_miss_checks(self):
        cache = HealthCache(ttl=60.0)
        checker = HealthChecker(FLEET_SERVICES[:1])
        cache.set_services(FLEET_SERVICES[:1])

        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            results = cache.get(checker)
            assert len(results) == 1
            mock_check.assert_called_once()

    def test_cache_hit_no_check(self):
        cache = HealthCache(ttl=60.0)
        checker = HealthChecker(FLEET_SERVICES[:1])
        cache.set_services(FLEET_SERVICES[:1])

        # Prime cache
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            cache.get(checker)

        # Second call should hit cache
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            results = cache.get(checker)
            mock_check.assert_not_called()
            assert results[0].name == FLEET_SERVICES[0].name

    def test_force_refresh(self):
        cache = HealthCache(ttl=60.0)
        checker = HealthChecker(FLEET_SERVICES[:1])
        cache.set_services(FLEET_SERVICES[:1])

        # Prime cache
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            cache.get(checker)

        # Force refresh
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=20.0, status="UP"
            )
            results = cache.get(checker, force=True)
            mock_check.assert_called_once()
            assert results[0].latency_ms == 20.0

    def test_clear(self):
        cache = HealthCache(ttl=60.0)
        checker = HealthChecker(FLEET_SERVICES[:1])
        cache.set_services(FLEET_SERVICES[:1])

        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            cache.get(checker)

        cache.clear()
        with patch.object(HealthChecker, "check_http") as mock_check:
            mock_check.return_value = CheckResult(
                name=FLEET_SERVICES[0].name, ok=True, latency_ms=10.0, status="UP"
            )
            cache.get(checker)
            mock_check.assert_called_once()


class TestFleetServicesCount:
    def test_fleet_services_length(self):
        assert len(FLEET_SERVICES) == 18
