"""Tests for batch_processor.py — Batch processing pipeline.

Run: python3 -m pytest tests/test_batch_processor.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.batch_processor import BatchProcessor, BatchResult


class TestBatchProcessor:
    def test_create(self):
        bp = BatchProcessor(max_workers=2)
        assert bp.max_workers == 2

    def test_process_simple(self):
        bp = BatchProcessor(max_workers=2)
        items = [1, 2, 3, 4, 5]
        result = bp.process(items, lambda x: x * 2)
        assert result.total == 5
        assert len(result.successful) == 5
        assert sorted(result.successful) == [2, 4, 6, 8, 10]
        assert result.failed_count == 0
        assert result.success_rate == 1.0

    def test_process_with_failures(self):
        bp = BatchProcessor(max_workers=2)
        items = [1, 2, 3]

        def bad_processor(x):
            if x == 2:
                raise ValueError("boom")
            return x * 2

        result = bp.process(items, bad_processor)
        assert result.total == 3
        assert len(result.successful) == 2
        assert len(result.failed) == 1
        assert result.failed[0][0] == 2
        assert result.success_rate == pytest.approx(2 / 3)

    def test_process_empty(self):
        bp = BatchProcessor()
        result = bp.process([], lambda x: x)
        assert result.total == 0
        assert result.successful == []

    def test_process_progress_callback(self):
        bp = BatchProcessor(max_workers=2)
        progress = []
        items = [1, 2, 3]
        bp.process(
            items,
            lambda x: x,
            on_progress=lambda done, total: progress.append((done, total)),
        )
        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_process_sequential(self):
        bp = BatchProcessor()
        items = [1, 2, 3]
        result = bp.process_sequential(items, lambda x: x * 2)
        assert len(result.successful) == 3
        assert result.successful == [2, 4, 6]

    def test_process_sequential_with_failure(self):
        bp = BatchProcessor()
        items = [1, 2, 3]

        def bad(x):
            if x == 2:
                raise RuntimeError("fail")
            return x

        result = bp.process_sequential(items, bad)
        assert len(result.successful) == 2
        assert len(result.failed) == 1

    def test_throughput(self):
        bp = BatchProcessor(max_workers=2)
        items = list(range(10))
        result = bp.process(items, lambda x: x)
        assert result.throughput > 0.0
        assert result.time_ms > 0.0

    def test_stats(self):
        bp = BatchProcessor(max_workers=2)
        bp.process([1, 2], lambda x: x)
        s = bp.stats()
        assert s["processed"] == 2
        assert s["failed"] == 0

    def test_repr(self):
        bp = BatchProcessor(max_workers=4)
        assert "BatchProcessor" in repr(bp)
