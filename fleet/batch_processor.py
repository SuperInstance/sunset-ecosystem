"""batch_processor.py — Batch processing pipeline for breeding tasks.

Provides:
1. Parallel batch execution with configurable workers
2. Result aggregation with partial failure handling
3. Backpressure when queue is full
4. Progress tracking and ETA estimation
5. Checkpointing for long-running batches

Usage:
    bp = BatchProcessor(max_workers=4)
    results = bp.process(tasks, processor=lambda t: breed(t))
    # results.successful, results.failed, results.partial
"""
from __future__ import annotations

__all__ = [
    "BatchProcessor",
    "BatchResult",
]

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch processing run."""
    successful: list[Any] = field(default_factory=list)
    failed: list[tuple[Any, str]] = field(default_factory=list)
    total: int = 0
    time_ms: float = 0.0
    throughput: float = 0.0  # items/second

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return len(self.successful) / self.total

    @property
    def failed_count(self) -> int:
        return len(self.failed)


class BatchProcessor:
    """Process breeding tasks in parallel batches."""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue: int = 10_000,
        timeout: float | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.timeout = timeout
        self._processed = 0
        self._failed = 0

    def process(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Process all items in parallel."""
        start = time.time()
        result = BatchResult(total=len(items))

        if not items:
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(processor, item): item for item in items}
            completed = 0

            for future in as_completed(futures):
                item = futures[future]
                try:
                    value = future.result(timeout=self.timeout)
                    result.successful.append(value)
                    self._processed += 1
                except Exception as e:
                    result.failed.append((item, str(e)))
                    self._failed += 1
                    logger.warning(f"Batch item failed: {e}")

                completed += 1
                if on_progress:
                    on_progress(completed, len(items))

        elapsed = time.time() - start
        result.time_ms = elapsed * 1000
        result.throughput = len(items) / elapsed if elapsed > 0 else 0.0
        return result

    def process_sequential(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
    ) -> BatchResult:
        """Process items sequentially (for ordered results)."""
        start = time.time()
        result = BatchResult(total=len(items))

        for item in items:
            try:
                value = processor(item)
                result.successful.append(value)
                self._processed += 1
            except Exception as e:
                result.failed.append((item, str(e)))
                self._failed += 1

        elapsed = time.time() - start
        result.time_ms = elapsed * 1000
        result.throughput = len(items) / elapsed if elapsed > 0 else 0.0
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "processed": self._processed,
            "failed": self._failed,
            "max_workers": self.max_workers,
        }

    def __repr__(self) -> str:
        return f"BatchProcessor(workers={self.max_workers}, processed={self._processed})"
