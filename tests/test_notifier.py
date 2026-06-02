"""Tests for fleet.notifier — Multi-channel notification system."""

import tempfile
from pathlib import Path

import pytest

from fleet.notifier import FleetNotifier, BreedingAlert, FileChannel, Channel


class TestBreedingAlert:
    def test_thermal_critical(self):
        a = BreedingAlert.thermal_critical(pressure=0.95, source="gpu-0")
        assert a.severity == "critical"
        assert a.category == "thermal"
        assert "95" in a.body
        assert "gpu-0" in a.body

    def test_flux_gate_block(self):
        a = BreedingAlert.flux_gate_block(candidate_id="c1", violations={"bound": 0.5})
        assert a.severity == "warning"
        assert a.category == "flux_gate"
        assert "c1" in a.body

    def test_breeding_failure(self):
        a = BreedingAlert.breeding_failure(error="oom", generation=5)
        assert a.severity == "critical"
        assert a.category == "breeding"
        assert "oom" in a.body

    def test_proof_generated(self):
        a = BreedingAlert.proof_generated(candidate_id=10, proof_hash="abc123", cycles=100)
        assert a.severity == "info"
        assert a.category == "proof"
        assert "10" in a.body

    def test_service_down(self):
        a = BreedingAlert.service_down("oracle1", "host", 8080)
        assert a.severity == "warning"
        assert a.category == "health"

    def test_to_dict(self):
        a = BreedingAlert("T", "B", "warning", "thermal")
        d = a.to_dict()
        assert d["title"] == "T"
        assert d["severity"] == "warning"
        assert d["category"] == "thermal"


class TestChannel:
    def test_abstract_send_raises(self):
        c = Channel("test")
        a = BreedingAlert("T", "B", "info", "health")
        with pytest.raises(NotImplementedError):
            c.send(a)


class TestFleetNotifier:
    def test_add_and_remove_channel(self):
        n = FleetNotifier()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        n.add_file(path)
        assert len(n.channels) == 1
        assert n.channels[0].name == "file"
        Path(path).unlink(missing_ok=True)

    def test_send_without_channels(self):
        n = FleetNotifier()
        a = BreedingAlert("T", "B", "info", "health")
        # Should not raise
        n.send(a)

    def test_send_routes_by_category(self):
        n = FleetNotifier()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        n.add_file(path)
        # Proof category should route to file only
        a = BreedingAlert.proof_generated(candidate_id=1, cycles=10, proof_hash="x")
        results = n.send(a)
        assert results.get("file") is True
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 1
        data = __import__("json").loads(lines[0])
        assert data["category"] == "proof"
        Path(path).unlink(missing_ok=True)

    def test_send_no_route_for_unknown_category(self):
        n = FleetNotifier()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        n.add_file(path)
        a = BreedingAlert("T", "B", "info", "unknown_cat")
        results = n.send(a)
        # All channels get it when category is unknown
        assert len(results) == 1
        Path(path).unlink(missing_ok=True)
