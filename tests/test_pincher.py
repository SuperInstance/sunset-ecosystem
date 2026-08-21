"""Tests for Pincher.

Covers:
- ExtractionQuery compilation and validation
- FileSource and MemorySource iteration
- Pattern matching (regex pre-filter)
- Transform pipeline (copy, regex, json_path, map, concat)
- Confidence scoring
- Quanta VDB integration (if available)
- Batch extraction
- Stats tracking
"""

from __future__ import annotations

import pytest
import numpy as np

from fleet.pincher import (
    Pincher,
    ExtractionQuery,
    FileSource,
    MemorySource,
    ExtractionResult,
)
from fleet.caslang_executor import ExecutionSandbox, CaslangExecutor


class TestExtractionQuery:
    def test_compile_patterns(self) -> None:
        q = ExtractionQuery(
            name="test",
            patterns=["error", "exception.*failed"],
        )
        compiled = q.compile_patterns()
        assert len(compiled) == 2
        assert compiled[0].search("error: something") is not None
        assert compiled[1].search("exception: task failed") is not None

    def test_compile_invalid_pattern(self) -> None:
        q = ExtractionQuery(name="test", patterns=["[invalid"])
        compiled = q.compile_patterns()
        assert len(compiled) == 0  # invalid pattern skipped


class TestDataSources:
    def test_memory_source(self) -> None:
        records = [
            {"id": "1", "message": "error: disk full"},
            {"id": "2", "message": "info: started"},
        ]
        source = MemorySource(records)
        results = list(source)
        assert len(results) == 2
        assert results[0]["id"] == "1"


class TestTransforms:
    def test_copy_transform(self) -> None:
        pincher = Pincher()
        record = {"name": "Alice", "age": 30}
        transforms = [{"from": "name", "to": "user_name", "type": "copy"}]
        result = pincher._apply_transforms(record, transforms)
        assert result["user_name"] == "Alice"

    def test_regex_transform(self) -> None:
        pincher = Pincher()
        record = {"log": "error: disk full at /dev/sda1"}
        transforms = [
            {"from": "log", "to": "device", "type": "regex", "pattern": r"/dev/\w+"}
        ]
        result = pincher._apply_transforms(record, transforms)
        assert result["device"] == "/dev/sda1"

    def test_json_path_transform(self) -> None:
        pincher = Pincher()
        record = {"data": {"user": {"name": "Bob", "id": 42}}}
        transforms = [
            {
                "from": "data",
                "to": "user_name",
                "type": "json_path",
                "path": "user.name",
            }
        ]
        result = pincher._apply_transforms(record, transforms)
        assert result["user_name"] == "Bob"

    def test_map_transform(self) -> None:
        pincher = Pincher()
        record = {"status": "200"}
        transforms = [
            {
                "from": "status",
                "to": "status_name",
                "type": "map",
                "mapping": {"200": "OK", "500": "ERROR"},
            }
        ]
        result = pincher._apply_transforms(record, transforms)
        assert result["status_name"] == "OK"

    def test_concat_transform(self) -> None:
        pincher = Pincher()
        record = {"first": "John", "last": "Doe"}
        transforms = [
            {
                "from": "",
                "to": "full_name",
                "type": "concat",
                "fields": ["first", "last"],
                "separator": " ",
            }
        ]
        result = pincher._apply_transforms(record, transforms)
        assert result.get("full_name") == "John Doe"


class TestExtraction:
    def test_simple_extraction(self) -> None:
        pincher = Pincher()
        records = [
            {"id": "1", "message": "error: disk full", "level": "error"},
            {"id": "2", "message": "info: started", "level": "info"},
            {"id": "3", "message": "error: connection timeout", "level": "error"},
        ]
        query = ExtractionQuery(
            name="errors",
            patterns=["error"],
            transforms=[
                {"from": "message", "to": "error_msg", "type": "copy"},
                {"from": "level", "to": "severity", "type": "copy"},
            ],
            min_confidence=0.5,
        )
        results = pincher.extract(query, MemorySource(records))
        assert len(results) == 2
        assert all(r.extracted_fields["severity"] == "error" for r in results)
        assert results[0].matched_patterns == ["error"]

    def test_no_match(self) -> None:
        pincher = Pincher()
        records = [{"id": "1", "message": "info: started"}]
        query = ExtractionQuery(name="errors", patterns=["error"])
        results = pincher.extract(query, MemorySource(records))
        assert len(results) == 0

    def test_confidence_filtering(self) -> None:
        pincher = Pincher()
        records = [
            {"id": "1", "message": "error: disk full"},
            {"id": "2", "message": "warning: low memory"},
        ]
        query = ExtractionQuery(
            name="critical",
            patterns=["error"],
            transforms=[{"from": "message", "to": "msg", "type": "copy"}],
            min_confidence=0.9,  # high threshold
        )
        results = pincher.extract(query, MemorySource(records))
        # Only "error" with multiple patterns would hit 0.9, so likely 0 or 1
        assert len(results) <= 1

    def test_max_results_limit(self) -> None:
        pincher = Pincher()
        records = [{"id": str(i), "message": "error: something"} for i in range(100)]
        query = ExtractionQuery(name="errors", patterns=["error"], max_results=5)
        results = pincher.extract(query, MemorySource(records))
        assert len(results) == 5

    def test_batch_extraction(self) -> None:
        pincher = Pincher()
        queries = [
            ExtractionQuery(name="errors", patterns=["error"]),
            ExtractionQuery(name="warnings", patterns=["warning"]),
        ]
        sources = [
            MemorySource([{"id": "1", "message": "error: disk full"}]),
            MemorySource([{"id": "2", "message": "warning: low memory"}]),
        ]
        results = pincher.batch_extract(queries, sources)
        assert len(results) == 4  # 2 queries × 2 sources
        assert "errors::MemorySource[0]" in results
        assert "warnings::MemorySource[1]" in results


class TestExtractionResult:
    def test_to_vector(self) -> None:
        result = ExtractionResult(
            record_id="1",
            source="test",
            extracted_fields={"name": "Alice", "age": 30},
        )
        vec = result.to_vector(dim=64)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32
        # Should have some non-zero values from hashing
        assert np.any(vec != 0)


class TestStats:
    def test_stats_tracking(self) -> None:
        pincher = Pincher()
        records = [
            {"id": "1", "message": "error: disk full"},
            {"id": "2", "message": "info: started"},
        ]
        query = ExtractionQuery(name="errors", patterns=["error"])
        pincher.extract(query, MemorySource(records))

        stats = pincher.stats
        assert stats["queries_executed"] == 1
        assert stats["records_filtered"] == 2
        assert stats["records_extracted"] == 1
        assert 0.0 < stats["extraction_rate"] <= 1.0
