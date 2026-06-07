"""Pincher — Selective data extraction for the fleet.

An emergent application that combines Quanta's VDB for fast pattern matching
with caslang's constrained queries for deterministic, sandboxed data extraction.

Use Cases
---------
- **Intelligence Gathering**: Extract relevant signals from massive telemetry streams
- **Document Mining**: Pinch specific facts from large document corpora
- **Log Analysis**: Find anomalies and patterns in distributed logs
- **Data Cleaning**: Selectively extract and transform messy data sources

The metaphor: a crab's claw — precise, selective, powerful.  It doesn't
scoop everything; it pinches exactly what matters.

Architecture
------------
The Pincher operates on a "query pipeline" abstraction:

1. **Source Adapter** — Connects to data sources (files, streams, APIs,
   Quanta VDB partitions).  Normalizes data into extractable records.

2. **Pattern Matcher** — Uses Quanta's Hnswlib ANN search or regex/glob
   patterns to identify candidate records.  Fast pre-filter.

3. **Constraint Engine** — caslang sandbox evaluates each candidate against
   precise extraction rules.  Only records that pass all constraints are
   kept.

4. **Transform Pipeline** — Extracted fields are transformed, aggregated,
   and formatted into the target output schema.

5. **Sink Adapter** — Writes results to destination (file, VDB, message bus).

Reference
---------
- Quanta VDB: https://github.com/CantorAI/Quanta
- caslang: https://github.com/xlang-foundation/caslang
"""

from __future__ import annotations

__all__ = [
    "Pincher",
    "ExtractionQuery",
    "DataSource",
    "ExtractionResult",
]

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ── ExtractionResult ────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """A single extracted record."""

    record_id: str
    source: str
    matched_patterns: list[str] = field(default_factory=list)
    extracted_fields: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_vector(self, dim: int = 128) -> np.ndarray:
        """Serialize to a vector for VDB storage."""
        vec = np.zeros(dim, dtype=np.float32)
        # Hash-based encoding of extracted fields
        for i, (k, v) in enumerate(self.extracted_fields.items()):
            idx = i % dim
            vec[idx] += float(hash(str(v)) % 10000) / 10000.0
        return vec


# ── DataSource ───────────────────────────────────────────────────


class DataSource:
    """Abstract base for data sources."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileSource(DataSource):
    """Read records from a JSONL or text file."""

    def __init__(self, path: Path | str, parser: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.path = Path(path)
        self.parser = parser or self._default_parser

    def __iter__(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield self.parser(line)
                except Exception as exc:
                    logger.debug("Parse failed: %s", exc)

    @staticmethod
    def _default_parser(line: str) -> dict[str, Any]:
        return json.loads(line)


class QuantaSource(DataSource):
    """Query records from Quanta VDB as a data source."""

    def __init__(
        self,
        quanta_bridge: Any,
        query_vector: np.ndarray,
        k: int = 100,
        partition: str | None = None,
    ) -> None:
        self.quanta_bridge = quanta_bridge
        self.query_vector = query_vector
        self.k = k
        self.partition = partition
        self._results: list[dict[str, Any]] | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._results is None:
            self._results = self.quanta_bridge.search(
                self.query_vector, k=self.k, partition=self.partition
            )
        for r in self._results:
            yield {
                "record_id": str(r.get("id", "")),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}),
                "partition": r.get("partition", "default"),
            }


class MemorySource(DataSource):
    """In-memory data source for testing."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.records


# ── ExtractionQuery ───────────────────────────────────────────────


@dataclass
class ExtractionQuery:
    """Specification for a pincher extraction job.

    - **patterns**: Regex or glob patterns for pre-filtering
    - **constraints**: caslang script for precise validation
    - **transforms**: Field extraction and transformation rules
    - **output_schema**: Target field names and types
    """

    name: str
    patterns: list[str] = field(default_factory=list)
    constraints_script: str = ""  # caslang JSONL
    transforms: list[dict[str, Any]] = field(default_factory=list)
    output_schema: dict[str, str] = field(default_factory=dict)
    min_confidence: float = 0.5
    max_results: int = 1000

    def compile_patterns(self) -> list[re.Pattern[str]]:
        """Compile regex patterns for matching."""
        compiled: list[re.Pattern[str]] = []
        for p in self.patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as exc:
                logger.warning("Invalid pattern '%s': %s", p, exc)
        return compiled


# ── Pincher ───────────────────────────────────────────────────


class Pincher:
    """Selective data extraction engine.

    Parameters
    ----------
    quanta_bridge : QuantaVdbBridge | None
        Optional VDB for fast vector-based pre-filtering.
    caslang_executor : CaslangExecutor | None
        Optional sandbox for constraint validation.
    """

    def __init__(
        self,
        quanta_bridge: Any | None = None,
        caslang_executor: Any | None = None,
    ) -> None:
        self.quanta_bridge = quanta_bridge
        self.caslang_executor = caslang_executor
        self._lock = threading.Lock()
        self._queries_executed = 0
        self._records_extracted = 0
        self._records_filtered = 0

    # ── core extraction ───────────────────────────────────────────

    def extract(
        self,
        query: ExtractionQuery,
        source: DataSource,
    ) -> list[ExtractionResult]:
        """Run a pincher extraction query over a data source.

        Pipeline:
        1. Pre-filter with regex patterns (fast, O(n))
        2. Vector similarity filter via Quanta (if available, O(log n))
        3. Constraint validation via caslang sandbox (precise, O(m) per candidate)
        4. Transform and format output
        """
        results: list[ExtractionResult] = []
        compiled_patterns = query.compile_patterns()

        for record in source:
            self._records_filtered += 1

            # 1. Pattern pre-filter
            record_str = json.dumps(record, separators=(",", ":"))
            matched_patterns: list[str] = []
            for pat in compiled_patterns:
                if pat.search(record_str):
                    matched_patterns.append(pat.pattern)
            if compiled_patterns and not matched_patterns:
                continue

            # 2. Vector similarity filter (if Quanta available)
            if self.quanta_bridge is not None and "vector" in record:
                vec = np.array(record["vector"], dtype=np.float32)
                search_results = self.quanta_bridge.search(vec, k=1)
                if not search_results or search_results[0].get("score", 0) < query.min_confidence:
                    continue

            # 3. Constraint validation (if caslang available)
            if self.caslang_executor is not None and query.constraints_script:
                from .caslang_executor import CaslangScript
                try:
                    script = CaslangScript.from_jsonl(query.constraints_script)
                    # Inject record into script variables
                    for cmd in script.commands:
                        if cmd.get("op") == "flow.set" and cmd.get("name") == "record":
                            cmd["value"] = json.dumps(record)
                    exec_result = self.caslang_executor.execute(script)
                    if exec_result["status"] != "success":
                        continue
                    # Check if constraints passed
                    output = exec_result.get("output", {})
                    if isinstance(output, dict) and not output.get("passed", True):
                        continue
                except Exception as exc:
                    logger.debug("Constraint validation failed: %s", exc)
                    continue

            # 4. Extract and transform fields
            extracted_fields = self._apply_transforms(record, query.transforms)

            # 5. Build result
            confidence = self._compute_confidence(matched_patterns, extracted_fields)
            if confidence >= query.min_confidence:
                result = ExtractionResult(
                    record_id=str(record.get("id", record.get("record_id", f"rec_{self._records_filtered}"))),
                    source=source.__class__.__name__,
                    matched_patterns=matched_patterns,
                    extracted_fields=extracted_fields,
                    confidence=confidence,
                    context={"raw": record},
                )
                results.append(result)
                self._records_extracted += 1

                if len(results) >= query.max_results:
                    break

        self._queries_executed += 1
        return results

    def extract_to_vdb(
        self,
        query: ExtractionQuery,
        source: DataSource,
        partition_tag: str = "pincher",
    ) -> dict[str, Any]:
        """Extract and immediately store results in Quanta VDB."""
        results = self.extract(query, source)
        if self.quanta_bridge is None:
            return {"extracted": len(results), "stored": 0, "error": "No Quanta bridge"}

        try:
            from .quanta_vdb_bridge import QuantaTableEntry
            for r in results:
                entry = QuantaTableEntry(
                    agent_id=r.record_id,
                    vector=r.to_vector(dim=128),
                    timestamp=r.timestamp,
                    node_id="pincher",
                    generation=0,
                    fitness=r.confidence,
                    signature="pincher",
                    partition_tag=partition_tag,
                    extra=r.extracted_fields,
                )
                self.quanta_bridge.insert(entry)
            return {"extracted": len(results), "stored": len(results)}
        except Exception as exc:
            return {"extracted": len(results), "stored": 0, "error": str(exc)}

    # ── batch operations ──────────────────────────────────────────

    def batch_extract(
        self,
        queries: list[ExtractionQuery],
        sources: list[DataSource],
    ) -> dict[str, list[ExtractionResult]]:
        """Run multiple extraction queries in parallel over multiple sources."""
        output: dict[str, list[ExtractionResult]] = {}
        for q in queries:
            for i, s in enumerate(sources):
                key = f"{q.name}::{s.__class__.__name__}[{i}]"
                output[key] = self.extract(q, s)
        return output

    # ── internal helpers ──────────────────────────────────────────

    def _apply_transforms(
        self,
        record: dict[str, Any],
        transforms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply field extraction and transformation rules."""
        output: dict[str, Any] = {}
        for t in transforms:
            source_field = t.get("from", "")
            target_field = t.get("to", source_field)
            transform_type = t.get("type", "copy")

            if transform_type == "concat":
                fields = t.get("fields", [])
                sep = t.get("separator", " ")
                output[target_field] = sep.join(str(record.get(f, "")) for f in fields)
                continue

            raw_value = record.get(source_field)
            if raw_value is None:
                continue

            if transform_type == "copy":
                output[target_field] = raw_value
            elif transform_type == "regex":
                pattern = t.get("pattern", "")
                match = re.search(pattern, str(raw_value))
                if match:
                    output[target_field] = match.group(t.get("group", 0))
            elif transform_type == "json_path":
                path = t.get("path", "").split(".")
                val = raw_value
                for p in path:
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        val = None
                        break
                if val is not None:
                    output[target_field] = val
            elif transform_type == "map":
                mapping = t.get("mapping", {})
                output[target_field] = mapping.get(str(raw_value), raw_value)
            elif transform_type == "concat":
                fields = t.get("fields", [])
                sep = t.get("separator", " ")
                output[target_field] = sep.join(str(record.get(f, "")) for f in fields)
        return output

    def _compute_confidence(
        self,
        matched_patterns: list[str],
        extracted_fields: dict[str, Any],
    ) -> float:
        """Compute extraction confidence score."""
        base = 0.5
        # More patterns matched = higher confidence
        base += min(0.3, len(matched_patterns) * 0.1)
        # More fields extracted = higher confidence
        base += min(0.2, len(extracted_fields) * 0.05)
        return min(1.0, base)

    # ── stats ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queries_executed": self._queries_executed,
                "records_extracted": self._records_extracted,
                "records_filtered": self._records_filtered,
                "extraction_rate": (
                    self._records_extracted / max(1, self._records_filtered)
                ),
            }
