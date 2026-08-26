"""Tests for health_probe.py — Multi-type health probes.

Run: python3 -m pytest tests/test_health_probe.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.health_probe import HealthProbe


class TestHealthProbe:
    def test_create(self):
        probe = HealthProbe()
        assert len(probe.list_probes()) == 0

    def test_tcp_probe_success(self):
        probe = HealthProbe()
        # Connect to a known public DNS server on port 53
        probe.add_tcp("dns", host="8.8.8.8", port=53, timeout=2.0)
        result = probe.check("dns")
        assert result.healthy is True
        assert result.latency_ms >= 0

    def test_tcp_probe_failure(self):
        probe = HealthProbe()
        probe.add_tcp("bad", host="127.0.0.1", port=1, timeout=0.5)
        result = probe.check("bad")
        assert result.healthy is False

    def test_command_probe_success(self):
        probe = HealthProbe()
        probe.add_command("echo", cmd=["echo", "ok"])
        result = probe.check("echo")
        assert result.healthy is True
        assert "ok" in result.message

    def test_command_probe_failure(self):
        probe = HealthProbe()
        probe.add_command("false", cmd=["false"])
        result = probe.check("false")
        assert result.healthy is False

    def test_custom_probe_success(self):
        probe = HealthProbe()
        probe.add_custom("custom", fn=lambda: True)
        result = probe.check("custom")
        assert result.healthy is True

    def test_custom_probe_failure(self):
        probe = HealthProbe()
        probe.add_custom("custom", fn=lambda: False)
        result = probe.check("custom")
        assert result.healthy is False

    def test_probe_not_found(self):
        probe = HealthProbe()
        result = probe.check("missing")
        assert result.healthy is False
        assert "not found" in result.message

    def test_remove_probe(self):
        probe = HealthProbe()
        probe.add_tcp("test", host="127.0.0.1", port=80)
        assert probe.remove("test") is True
        assert probe.remove("missing") is False

    def test_check_all(self):
        probe = HealthProbe()
        probe.add_custom("a", fn=lambda: True)
        probe.add_custom("b", fn=lambda: False)
        results = probe.check_all()
        assert len(results) == 2
        assert any(r.name == "a" and r.healthy for r in results)
        assert any(r.name == "b" and not r.healthy for r in results)

    def test_probe_types(self):
        probe = HealthProbe()
        probe.add_tcp("tcp", host="127.0.0.1", port=80)
        probe.add_command("cmd", cmd=["true"])
        assert probe.probe_types()["tcp"] == "tcp"
        assert probe.probe_types()["cmd"] == "command"

    def test_repr(self):
        probe = HealthProbe()
        probe.add_custom("x", fn=lambda: True)
        assert "HealthProbe" in repr(probe)
