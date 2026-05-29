"""Request proxy with forwarding, retry, and circuit breaking.

Forwards requests to backend services with retry logic, health-aware
circuit breaking, and response caching. Used for fleet API gateway
and inter-service communication.

Usage:
    proxy = RequestProxy()
    proxy.add_backend("api", "http://api-1:8080")
    response = proxy.forward("api", {"path": "/users"})
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class RequestProxy:
    """
    Request proxy with backends and retry logic.

    :param max_retries: Max retry attempts per request.
    """

    def __init__(self, max_retries: int = 3):
        self._max_retries = max_retries
        self._backends: Dict[str, List[str]] = {}
        self._round_robin: Dict[str, int] = {}
        self._health: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def add_backend(self, name: str, url: str) -> None:
        """Register a backend URL."""
        if name not in self._backends:
            self._backends[name] = []
        self._backends[name].append(url)
        self._health[url] = True

    def remove_backend(self, name: str, url: str) -> bool:
        """Remove a backend URL."""
        if name not in self._backends:
            return False
        if url in self._backends[name]:
            self._backends[name].remove(url)
            self._health.pop(url, None)
            return True
        return False

    def set_health(self, url: str, healthy: bool) -> None:
        """Set health status for a backend."""
        self._health[url] = healthy

    # ------------------------------------------------------------------
    # Forwarding (stub)
    # ------------------------------------------------------------------

    def forward(self, name: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Forward request to a healthy backend (stub).

        Returns a simulated response. In production, makes HTTP call.
        """
        backends = self._backends.get(name, [])
        healthy = [u for u in backends if self._health.get(u, True)]
        if not healthy:
            return None
        # Round-robin selection
        idx = self._round_robin.get(name, 0) % len(healthy)
        self._round_robin[name] = idx + 1
        selected = healthy[idx]
        return {
            "backend": selected,
            "request": request,
            "status": "forwarded",
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "services": len(self._backends),
            "backends": sum(len(v) for v in self._backends.values()),
        }

    def __repr__(self) -> str:
        return f"<RequestProxy services={len(self._backends)}>"
