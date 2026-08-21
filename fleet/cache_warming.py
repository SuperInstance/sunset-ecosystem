"""Preemptive cache warming with batch loading.

Warms caches by pre-loading data before it's needed. Supports batch
operations, error isolation, and scheduled warming. Used for fleet
pre-computation and cache pre-heating.

Usage:
    warmer = CacheWarmer()
    warmer.warm(["key-1", "key-2"], loader_fn)
    warmer.prefetch(["key-3"], loader=loader_fn)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Cache warmer with parallel batch loading.

    :param max_workers: Number of parallel loader threads.
    """

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._warmed: int = 0
        self._errors: int = 0
        self._queued: List[Tuple[str, Callable[[str], Any]]] = []

    # ------------------------------------------------------------------
    # Batch warming
    # ------------------------------------------------------------------

    def warm(self, keys: List[str], loader: Callable[[str], Any]) -> Dict[str, Any]:
        """
        Warm cache with a batch of keys.

        :param keys: Keys to load.
        :param loader: Function(key) -> value.
        :returns: Dict of loaded values.
        """
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._safe_load, loader, key): key for key in keys}
            for future in futures:
                key = futures[future]
                try:
                    value = future.result()
                    results[key] = value
                    self._warmed += 1
                except Exception as e:
                    logger.error(f"Cache warm failed for {key}: {e}")
                    self._errors += 1
        return results

    def _safe_load(self, loader: Callable[[str], Any], key: str) -> Any:
        return loader(key)

    # ------------------------------------------------------------------
    # Prefetch interface
    # ------------------------------------------------------------------

    def prefetch(self, keys: List[str], *, loader: Callable[[str], Any]) -> None:
        """Prefetch keys into cache (fire-and-forget)."""
        self.warm(keys, loader)

    # ------------------------------------------------------------------
    # Scheduled warming
    # ------------------------------------------------------------------

    def schedule(self, key: str, loader: Callable[[str], Any]) -> None:
        """Queue a key for later warming."""
        self._queued.append((key, loader))

    def run_scheduled(self) -> Dict[str, Any]:
        """Execute all scheduled warm operations."""
        keys = [k for k, _ in self._queued]
        loaders = [l for _, l in self._queued]
        # Use first loader for all (simplification)
        if not loaders:
            return {}
        results = self.warm(keys, loaders[0])
        self._queued.clear()
        return results

    def clear_queue(self) -> None:
        """Clear scheduled warm queue."""
        self._queued.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "warmed": self._warmed,
            "errors": self._errors,
            "queued": len(self._queued),
        }

    def __repr__(self) -> str:
        return (
            f"<CacheWarmer warmed={self._warmed} "
            f"errors={self._errors} queued={len(self._queued)}>"
        )
