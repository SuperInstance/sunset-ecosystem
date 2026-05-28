"""data_pipeline.py — Streaming ETL pipeline for breeding data.

Provides:
1. Source/sink abstraction for data streams
2. Transform stages (map, filter, aggregate)
3. Batch and windowed processing
4. Backpressure handling
5. Pipeline metrics

Usage:
    pipeline = DataPipeline()
    pipeline.add_source(lambda: generate_tiles())
    pipeline.add_transform(lambda x: x["score"] > 0.5)
    pipeline.add_sink(lambda batch: write_to_db(batch))
    pipeline.run(batch_size=100)
"""
from __future__ import annotations

__all__ = [
    "DataPipeline",
    "PipelineStage",
    "PipelineStats",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class PipelineStage:
    """A named pipeline stage."""
    name: str
    fn: Callable[[Any], Any]
    stage_type: str  # "source", "transform", "sink"


@dataclass
class PipelineStats:
    """Pipeline execution statistics."""
    items_processed: int = 0
    items_dropped: int = 0
    batches_processed: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    avg_batch_time_ms: float = 0.0


class DataPipeline:
    """Streaming ETL pipeline for breeding data."""

    def __init__(self, max_batch: int = 1000, backpressure_limit: int = 10_000) -> None:
        self._max_batch = max_batch
        self._backpressure_limit = backpressure_limit
        self._stages: list[PipelineStage] = []
        self._stats = PipelineStats()
        self._running = False
        self._stopped = False

    def add_source(self, fn: Callable[[], list[Any] | Any]) -> None:
        """Add a source stage."""
        self._stages.append(PipelineStage(name="source", fn=fn, stage_type="source"))

    def add_transform(self, fn: Callable[[Any], Any | None]) -> None:
        """Add a transform stage. Returns None to drop item."""
        self._stages.append(PipelineStage(name="transform", fn=fn, stage_type="transform"))

    def add_filter(self, predicate: Callable[[Any], bool]) -> None:
        """Add a filter stage."""
        def _filter(x: Any) -> Any | None:
            return x if predicate(x) else None
        self._stages.append(PipelineStage(name="filter", fn=_filter, stage_type="transform"))

    def add_sink(self, fn: Callable[[list[Any]], None]) -> None:
        """Add a sink stage (receives batches)."""
        self._stages.append(PipelineStage(name="sink", fn=fn, stage_type="sink"))

    def run(self, max_items: int | None = None) -> PipelineStats:
        """Run the pipeline until max_items or exhaustion."""
        if not self._stages or self._stopped:
            return self._stats

        self._running = True
        start = time.time()
        items_processed = 0
        batch: list[Any] = []

        source_stage = self._stages[0]
        transform_stages = [s for s in self._stages[1:] if s.stage_type == "transform"]
        sink_stages = [s for s in self._stages[1:] if s.stage_type == "sink"]

        while self._running:
            # Pull from source
            try:
                raw = source_stage.fn()
                if raw is None:
                    break
                items = raw if isinstance(raw, list) else [raw]
            except Exception as e:
                logger.error(f"Source error: {e}")
                self._stats.errors += 1
                break

            for item in items:
                if max_items is not None and items_processed >= max_items:
                    self._running = False
                    break

                # Backpressure check
                if len(batch) >= self._backpressure_limit:
                    logger.warning("Pipeline backpressure limit reached")
                    self._running = False
                    break

                # Apply transforms
                current = item
                dropped = False
                for stage in transform_stages:
                    try:
                        current = stage.fn(current)
                        if current is None:
                            dropped = True
                            self._stats.items_dropped += 1
                            break
                    except Exception as e:
                        logger.warning(f"Transform error: {e}")
                        dropped = True
                        self._stats.errors += 1
                        break

                if not dropped:
                    batch.append(current)
                    items_processed += 1

                # Flush batch
                if len(batch) >= self._max_batch:
                    self._flush_batch(batch, sink_stages)
                    batch = []

            if not self._running:
                break

        # Final flush
        if batch:
            self._flush_batch(batch, sink_stages)

        elapsed = time.time() - start
        self._stats.items_processed += items_processed
        self._stats.batches_processed += max(1, (items_processed + self._max_batch - 1) // self._max_batch)
        self._stats.total_time_ms += elapsed * 1000
        self._stats.avg_batch_time_ms = (
            self._stats.total_time_ms / max(self._stats.batches_processed, 1)
        )

        return self._stats

    def _flush_batch(self, batch: list[Any], sinks: list[PipelineStage]) -> None:
        for sink in sinks:
            try:
                sink.fn(batch)
            except Exception as e:
                logger.error(f"Sink error: {e}")
                self._stats.errors += 1

    def stop(self) -> None:
        """Signal the pipeline to stop."""
        self._running = False
        self._stopped = True

    def reset(self) -> None:
        """Reset pipeline statistics."""
        self._stats = PipelineStats()

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    def __repr__(self) -> str:
        return f"DataPipeline(stages={len(self._stages)}, processed={self._stats.items_processed})"
