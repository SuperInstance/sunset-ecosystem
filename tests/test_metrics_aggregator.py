import time
import pytest
from fleet.metrics_aggregator import MetricPoint, MetricsAggregator


class TestMetricPoint:
    def test_to_dict(self):
        p = MetricPoint("cpu", 0.5, 0.0, {"host": "a"})
        d = p.to_dict()
        assert d["metric_name"] == "cpu"
        assert d["value"] == 0.5


class TestMetricsAggregator:
    def test_init(self):
        m = MetricsAggregator()
        assert m.fleet_node_id == "default"
        assert m.get_all_stats() == {}

    def test_record(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5)
        stats = m.get_stats("cpu")
        assert stats["count"] == 1
        assert stats["last"] == 0.5

    def test_record_multiple(self):
        m = MetricsAggregator()
        for i in range(10):
            m.record("cpu", i)
        stats = m.get_stats("cpu")
        assert stats["count"] == 10
        assert stats["mean"] == 4.5

    def test_record_with_tags(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5, {"host": "a"})
        points = m.get_series("cpu")
        assert len(points) == 1
        assert points[0].tags["host"] == "a"

    def test_get_series_since(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5)
        time.sleep(0.01)
        now = time.time()
        m.record("cpu", 0.6)
        points = m.get_series("cpu", since=now)
        assert len(points) == 1
        assert points[0].value == 0.6

    def test_buffer_trim(self):
        m = MetricsAggregator(buffer_size=5)
        for i in range(10):
            m.record("cpu", i)
        stats = m.get_stats("cpu")
        assert stats["count"] == 5
        assert stats["last"] == 9.0

    def test_increment(self):
        m = MetricsAggregator()
        m.increment("requests", 1.0)
        m.increment("requests", 2.0)
        assert m.get_counters()["requests"] == 3.0

    def test_gauge(self):
        m = MetricsAggregator()
        m.gauge("temperature", 98.6)
        assert m.get_gauges()["temperature"] == 98.6

    def test_get_all_stats(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5)
        m.record("mem", 0.8)
        all_stats = m.get_all_stats()
        assert "cpu" in all_stats
        assert "mem" in all_stats

    def test_export_json(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5)
        m.increment("requests", 1.0)
        j = m.export_json()
        assert "cpu" in j
        assert "requests" in j

    def test_to_dict(self):
        m = MetricsAggregator()
        m.record("cpu", 0.5)
        m.gauge("temp", 100.0)
        d = m.to_dict()
        assert d["counters"] == 0
        assert d["gauges"] == 1
        assert "cpu" in d["stats"]
