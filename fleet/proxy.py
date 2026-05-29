"""Simple reverse proxy with load balancing and health checks.

Routes incoming requests to a pool of backend nodes using configurable
strategies (round-robin, random, least-connections). Periodically health-checks
backends and removes failed ones from rotation.

Usage:
    proxy = ReverseProxy(strategy="round_robin")
    proxy.add_backend("http://node-1:8080")
    proxy.add_backend("http://node-2:8080")
    target = proxy.pick()
    # forward request to target
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ProxyError(Exception):
    pass


@dataclass
class Backend:
    """A backend target."""

    url: str
    healthy: bool = True
    last_check: float = 0.0
    check_interval: float = 30.0
    failure_count: int = 0
    max_failures: int = 3
    connection_count: int = 0

    def mark_failed(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.healthy = False

    def mark_healthy(self) -> None:
        self.healthy = True
        self.failure_count = 0


class ReverseProxy:
    """
    Reverse proxy with backend pooling and health checks.

    :param strategy: "round_robin", "random", or "least_connections".
    :param check_fn: Optional health check callable(url) -> bool.
    """

    def __init__(
        self,
        strategy: str = "round_robin",
        check_fn: Optional[Callable[[str], bool]] = None,
    ):
        self._strategy = strategy
        self._backends: Dict[str, Backend] = {}
        self._rr_index = 0
        self._check_fn = check_fn
        self._stats: Dict[str, int] = {"requests": 0, "errors": 0}

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def add_backend(
        self,
        url: str,
        check_interval: float = 30.0,
        max_failures: int = 3,
    ) -> None:
        self._backends[url] = Backend(
            url=url,
            check_interval=check_interval,
            max_failures=max_failures,
        )

    def remove_backend(self, url: str) -> bool:
        if url in self._backends:
            del self._backends[url]
            return True
        return False

    def backends(self) -> List[Backend]:
        return list(self._backends.values())

    def healthy_backends(self) -> List[Backend]:
        return [b for b in self._backends.values() if b.healthy]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def pick(self) -> Optional[str]:
        """Select a backend URL. Returns None if none healthy."""
        healthy = self.healthy_backends()
        if not healthy:
            return None

        self._stats["requests"] += 1

        if self._strategy == "round_robin":
            if self._rr_index >= len(healthy):
                self._rr_index = 0
            target = healthy[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(healthy)
            target.connection_count += 1
            return target.url

        if self._strategy == "random":
            target = random.choice(healthy)
            target.connection_count += 1
            return target.url

        if self._strategy == "least_connections":
            target = min(healthy, key=lambda b: b.connection_count)
            target.connection_count += 1
            return target.url

        raise ProxyError(f"Unknown strategy: {self._strategy}")

    def release(self, url: str) -> None:
        """Release a backend connection (decrement count)."""
        backend = self._backends.get(url)
        if backend:
            backend.connection_count = max(0, backend.connection_count - 1)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, bool]:
        """Run health checks on all backends. Returns url -> healthy map."""
        now = time.time()
        results: Dict[str, bool] = {}
        for url, backend in self._backends.items():
            if now - backend.last_check < backend.check_interval:
                results[url] = backend.healthy
                continue
            backend.last_check = now
            if self._check_fn is None:
                results[url] = backend.healthy
                continue
            try:
                healthy = self._check_fn(url)
            except Exception:
                healthy = False
            if healthy:
                backend.mark_healthy()
            else:
                backend.mark_failed()
                self._stats["errors"] += 1
            results[url] = backend.healthy
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def __repr__(self) -> str:
        healthy = len(self.healthy_backends())
        total = len(self._backends)
        return f"<ReverseProxy strategy={self._strategy} healthy={healthy}/{total}>"
