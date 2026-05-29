from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class DataRecord:
    """A record in the data pipeline."""
    record_id: str
    data: Dict[str, Any]
    timestamp: float
    source: str
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "tags": self.tags,
        }


class DataPipeline:
    """
    Data ingestion and preprocessing pipeline.

    Ingests records, applies transformations, and routes to destinations.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._records: List[DataRecord] = []
        self._transforms: List[Callable] = []
        self._destinations: List[Callable] = []
        self._stats: Dict[str, int] = {"ingested": 0, "transformed": 0, "routed": 0}

    def add_transform(self, transform: Callable) -> None:
        """Add a transformation function."""
        self._transforms.append(transform)

    def add_destination(self, dest: Callable) -> None:
        """Add a destination function."""
        self._destinations.append(dest)

    def ingest(self, data: Dict[str, Any], source: str = "unknown",
               tags: Optional[Dict[str, str]] = None) -> DataRecord:
        """Ingest a data record."""
        record = DataRecord(
            record_id=f"rec_{int(time.time() * 1000000)}",
            data=data,
            timestamp=time.time(),
            source=source,
            tags=tags or {},
        )
        self._records.append(record)
        self._stats["ingested"] += 1
        return record

    def process(self, record: DataRecord) -> DataRecord:
        """Apply all transformations to a record."""
        for transform in self._transforms:
            try:
                record.data = transform(record.data)
            except Exception:
                pass
        self._stats["transformed"] += 1
        return record

    def route(self, record: DataRecord) -> int:
        """Route a record to all destinations."""
        count = 0
        for dest in self._destinations:
            try:
                dest(record)
                count += 1
            except Exception:
                pass
        self._stats["routed"] += count
        return count

    def process_all(self) -> List[DataRecord]:
        """Process and route all pending records."""
        results = []
        for record in self._records:
            processed = self.process(record)
            self.route(processed)
            results.append(processed)
        return results

    def get_records(self, source: Optional[str] = None) -> List[DataRecord]:
        """Get records, optionally filtered by source."""
        if source:
            return [r for r in self._records if r.source == source]
        return self._records

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "records": len(self._records),
            "transforms": len(self._transforms),
            "destinations": len(self._destinations),
            "stats": self._stats,
        }

    def export_json(self) -> str:
        """Export pipeline state as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "records": [r.to_dict() for r in self._records[-100:]],
            "stats": self.get_stats(),
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
