"""Tests for data_pipeline.py — Streaming ETL pipeline.

Run: python3 -m pytest tests/test_data_pipeline.py -v --tb=short
"""
from __future__ import annotations

import pytest

from fleet.data_pipeline import DataPipeline


class TestDataPipeline:
    def test_create(self):
        pl = DataPipeline()
        assert pl._stages == []

    def test_source_only(self):
        pl = DataPipeline()
        items = [1, 2, 3]
        pl.add_source(lambda: items.pop(0) if items else None)
        stats = pl.run(max_items=3)
        assert stats.items_processed == 3

    def test_transform(self):
        pl = DataPipeline()
        items = [1, 2, 3]
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.add_transform(lambda x: x * 2)
        stats = pl.run(max_items=3)
        assert stats.items_processed == 3

    def test_filter(self):
        pl = DataPipeline()
        items = [1, 2, 3, 4, 5]
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.add_filter(lambda x: x > 2)
        stats = pl.run(max_items=5)
        assert stats.items_processed == 3  # 3, 4, 5
        assert stats.items_dropped == 2  # 1, 2

    def test_sink(self):
        pl = DataPipeline()
        items = [1, 2, 3]
        sink_data = []
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.add_sink(lambda batch: sink_data.extend(batch))
        pl.run(max_items=3)
        assert len(sink_data) == 3
        assert sink_data == [1, 2, 3]

    def test_batching(self):
        pl = DataPipeline(max_batch=2)
        items = [1, 2, 3, 4]
        batches = []
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.add_sink(lambda batch: batches.append(list(batch)))
        pl.run(max_items=4)
        assert len(batches) == 2
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]

    def test_transform_error(self):
        pl = DataPipeline()
        items = [1, 2, 3]
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.add_transform(lambda x: (_ for _ in ()).throw(ValueError("bad")) if x == 2 else x)
        stats = pl.run(max_items=3)
        assert stats.items_processed == 2  # 1 and 3
        assert stats.errors == 1

    def test_empty_pipeline(self):
        pl = DataPipeline()
        stats = pl.run()
        assert stats.items_processed == 0

    def test_backpressure(self):
        pl = DataPipeline(max_batch=100, backpressure_limit=2)
        items = list(range(10))
        pl.add_source(lambda: items.pop(0) if items else None)
        stats = pl.run()
        # Should stop at backpressure limit
        assert stats.items_processed <= 2

    def test_max_items(self):
        pl = DataPipeline()
        items = list(range(100))
        pl.add_source(lambda: items.pop(0) if items else None)
        stats = pl.run(max_items=5)
        assert stats.items_processed == 5

    def test_stop(self):
        pl = DataPipeline()
        items = list(range(100))
        pl.add_source(lambda: items.pop(0) if items else None)
        # Stop before run to test early exit
        pl._stopped = True
        stats = pl.run()
        assert stats.items_processed == 0

    def test_reset(self):
        pl = DataPipeline()
        items = [1]
        pl.add_source(lambda: items.pop(0) if items else None)
        pl.run(max_items=1)
        pl.reset()
        assert pl.stats.items_processed == 0

    def test_repr(self):
        pl = DataPipeline()
        assert "DataPipeline" in repr(pl)