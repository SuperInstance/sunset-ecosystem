"""Aggregates results from multiple workers (all/any/n-first).

Collects results from distributed workers with configurable completion
strategies: wait for all, wait for any, or wait for N successful results.
Supports timeout and partial failure handling. Used for fleet parallel
processing, quorum reads, and breeding batch completion.

Usage:
    agg = ResultAggregator(strategy="all", timeout_sec=30)
    agg.submit("worker-1", result=42)
    agg.submit("worker-2", result=43)
    result = agg.result()  # Blocks until all complete
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class ResultAggregator:
    """
    Result aggregator with completion strategies.

    :param strategy: "all", "any", or "n_first".
    :param n: Required results for "n_first" strategy.
    :param timeout_sec: Optional timeout.
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        strategy: str = "all",
        n: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        clock: Optional[callable] = None,
    ):
        self._strategy = strategy
        self._n = n
        self._timeout = timeout_sec
        self._clock = clock or time.time
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._expected: Optional[int] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def expect(self, count: int) -> None:
        """Set expected number of submissions."""
        self._expected = count

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self, source: str, result: Any = None, error: Optional[str] = None
    ) -> None:
        """
        Submit a result from a source.

        :param source: Worker/source identifier.
        :param result: Successful result.
        :param error: Error message if failed.
        """
        if self._start_time is None:
            self._start_time = self._clock()
        if error:
            self._errors[source] = error
        else:
            self._results[source] = result

    # ------------------------------------------------------------------
    # Completion checking
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """Check if the aggregation is complete per strategy."""
        if self._strategy == "any" and (self._results or self._errors):
            return True
        if self._strategy == "n_first":
            target = self._n or 1
            return len(self._results) >= target
        if self._strategy == "all":
            if self._expected is None:
                return False
            total = len(self._results) + len(self._errors)
            return total >= self._expected
        return False

    def is_timed_out(self) -> bool:
        """Check if timeout has elapsed."""
        if self._timeout is None or self._start_time is None:
            return False
        return (self._clock() - self._start_time) >= self._timeout

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def result(self) -> Dict[str, Any]:
        """
        Get aggregation result.

        :returns: Dict with "results", "errors", "complete", "timed_out".
        """
        return {
            "results": dict(self._results),
            "errors": dict(self._errors),
            "complete": self.is_complete(),
            "timed_out": self.is_timed_out(),
            "success_count": len(self._results),
            "error_count": len(self._errors),
        }

    def get_result(self, source: str) -> Optional[Any]:
        return self._results.get(source)

    def get_error(self, source: str) -> Optional[str]:
        return self._errors.get(source)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sources(self) -> List[str]:
        return list(self._results.keys()) + list(self._errors.keys())

    def successful(self) -> List[str]:
        return list(self._results.keys())

    def failed(self) -> List[str]:
        return list(self._errors.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "strategy": self._strategy,
            "n": self._n,
            "timeout": self._timeout,
            "expected": self._expected,
            "success_count": len(self._results),
            "error_count": len(self._errors),
            "complete": self.is_complete(),
            "timed_out": self.is_timed_out(),
        }

    def __repr__(self) -> str:
        return f"<ResultAggregator strategy={self._strategy} success={len(self._results)} errors={len(self._errors)}>"
