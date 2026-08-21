"""Tests for metrics_pipeline.py — Metrics transformation pipeline.

Run: python3 -m pytest tests/test_metrics_pipeline.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.metrics_pipeline import MetricsPipeline


class TestMetricsPipeline:
    def test_create(self):
        pipe = MetricsPipeline()
        assert pipe.stats()["processed"] == 0

    def test_single_transform(self):
        pipe = MetricsPipeline()
        pipe.add_transform(lambda m: {**m, "value": m["value"] * 2})
        result = pipe.process({"name": "cpu", "value": 50})
        assert result["value"] == 100

    def test_filter(self):
        pipe = MetricsPipeline()
        pipe.add_filter(lambda m: m["value"] > 10)
        assert pipe.process({"value": 20}) is not None
        assert pipe.process({"value": 5}) is None
        assert pipe.stats()["dropped"] == 1

    def test_chained_transforms(self):
        pipe = MetricsPipeline()
        pipe.add_transform(lambda m: {**m, "a": 1})
        pipe.add_transform(lambda m: {**m, "b": 2})
        result = pipe.process({"x": 0})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_batch_processing(self):
        pipe = MetricsPipeline()
        pipe.add_filter(lambda m: m.get("keep", False))
        results = pipe.process_batch(
            [
                {"keep": True, "v": 1},
                {"keep": False, "v": 2},
                {"keep": True, "v": 3},
            ]
        )
        assert len(results) == 2
        assert results[0]["v"] == 1
        assert results[1]["v"] == 3

    def test_aggregator(self):
        pipe = MetricsPipeline()
        pipe.add_aggregator(lambda metrics: {"sum": sum(m["v"] for m in metrics)})
        results = pipe.process_batch([{"v": 1}, {"v": 2}])
        assert any(r.get("sum") == 3 for r in results)

    def test_no_filters_passes_all(self):
        pipe = MetricsPipeline()
        result = pipe.process({"x": 1})
        assert result == {"x": 1}

    def test_reset_stats(self):
        pipe = MetricsPipeline()
        pipe.process({"x": 1})
        pipe.reset_stats()
        assert pipe.stats()["processed"] == 0

    def test_repr(self):
        pipe = MetricsPipeline()
        pipe.add_filter(lambda m: True)
        assert "MetricsPipeline" in repr(pipe)
