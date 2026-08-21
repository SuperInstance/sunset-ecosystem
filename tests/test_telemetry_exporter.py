import pytest
from fleet.telemetry_exporter import MetricSample, TelemetryExporter


class TestMetricSample:
    def test_init(self):
        s = MetricSample(name="test", value=1.0, timestamp=0.0)
        assert s.name == "test"
        assert s.value == 1.0

    def test_to_prometheus_no_labels(self):
        s = MetricSample(name="test", value=1.0, timestamp=0.0)
        line = s.to_prometheus()
        assert "test" in line
        assert "1.0" in line

    def test_to_prometheus_with_labels(self):
        s = MetricSample(name="test", value=1.0, timestamp=0.0, labels={"node": "n1"})
        line = s.to_prometheus()
        assert 'node="n1"' in line

    def test_to_otel(self):
        s = MetricSample(name="test", value=1.0, timestamp=0.0)
        d = s.to_otel()
        assert d["name"] == "test"
        assert d["value"] == 1.0


class TestTelemetryExporter:
    def test_init(self):
        ex = TelemetryExporter()
        assert ex.fleet_node_id == "default"
        assert ex.samples == []

    def test_record(self):
        ex = TelemetryExporter()
        ex.record("test", 1.0)
        assert len(ex.samples) == 1
        assert ex.samples[0].name == "test"

    def test_record_counter(self):
        ex = TelemetryExporter()
        ex.record("cnt", 1.0, metric_type="counter")
        ex.record("cnt", 2.0, metric_type="counter")
        assert ex.counters["cnt"] == 3.0

    def test_record_breeding_metrics(self):
        ex = TelemetryExporter()
        ex.record_breeding_metrics(1, 10, 0.9, 0.7, 0.5)
        assert len(ex.samples) == 5
        names = [s.name for s in ex.samples]
        assert "breeding_generation" in names

    def test_record_spatial_metrics(self):
        ex = TelemetryExporter()
        ex.record_spatial_metrics(5, 3, 1.5, 0.0)
        assert len(ex.samples) == 4
        names = [s.name for s in ex.samples]
        assert "spatial_agents" in names

    def test_record_health_metrics(self):
        ex = TelemetryExporter()
        ex.record_health_metrics(50.0, 70.0, 10, 2)
        assert len(ex.samples) == 4
        names = [s.name for s in ex.samples]
        assert "health_cpu" in names

    def test_export_prometheus(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        prom = ex.export_prometheus()
        assert "# Fleet Telemetry" in prom
        assert "x" in prom

    def test_export_otel(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        otel = ex.export_otel()
        assert len(otel) == 1
        assert otel[0]["name"] == "x"

    def test_export_json(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        j = ex.export_json()
        assert "x" in j
        assert "node" in j

    def test_get_summary(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        ex.record("x", 3.0)
        s = ex.get_summary()
        assert s["x"]["mean"] == 2.0
        assert s["x"]["count"] == 2

    def test_get_summary_empty(self):
        ex = TelemetryExporter()
        assert ex.get_summary() == {}

    def test_clear(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        ex.clear()
        assert ex.samples == []

    def test_to_dict(self):
        ex = TelemetryExporter()
        ex.record("x", 1.0)
        d = ex.to_dict()
        assert d["node"] == "default"
        assert d["samples"] == 1
