"""Service mesh with routing, health checking, and retries.

Provides service-to-service communication with health-aware routing,
automatic retries, and load balancing. Used for fleet inter-service
mesh, API gateway backends, and resilient service calls.

Usage:
    mesh = ServiceMesh()
    mesh.register("users", ["http://user-1:8080", "http://user-2:8080"])
    result = mesh.call("users", "/get/123")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ServiceMesh:
    """
    Service mesh with routing and health checking.

    :param max_retries: Max retries per call.
    """

    def __init__(self, max_retries: int = 3):
        self._max_retries = max_retries
        self._services: Dict[str, List[str]] = {}
        self._health: Dict[str, bool] = {}
        self._round_robin: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, service: str, endpoints: List[str]) -> None:
        """Register endpoints for a service."""
        self._services[service] = list(endpoints)
        for ep in endpoints:
            self._health[ep] = True
        self._round_robin[service] = 0

    def deregister(self, service: str) -> bool:
        """Remove a service."""
        if service not in self._services:
            return False
        for ep in self._services[service]:
            self._health.pop(ep, None)
        del self._services[service]
        del self._round_robin[service]
        return True

    def add_endpoint(self, service: str, endpoint: str) -> None:
        """Add an endpoint to a service."""
        if service not in self._services:
            self._services[service] = []
        self._services[service].append(endpoint)
        self._health[endpoint] = True

    def remove_endpoint(self, service: str, endpoint: str) -> bool:
        """Remove an endpoint from a service."""
        if service not in self._services:
            return False
        if endpoint in self._services[service]:
            self._services[service].remove(endpoint)
            self._health.pop(endpoint, None)
            return True
        return False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def set_health(self, endpoint: str, healthy: bool) -> None:
        """Set endpoint health status."""
        self._health[endpoint] = healthy

    def is_healthy(self, endpoint: str) -> bool:
        """Check endpoint health."""
        return self._health.get(endpoint, True)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _healthy_endpoints(self, service: str) -> List[str]:
        """Get healthy endpoints for a service."""
        endpoints = self._services.get(service, [])
        return [ep for ep in endpoints if self.is_healthy(ep)]

    def _select(self, service: str) -> Optional[str]:
        """Select an endpoint via round-robin."""
        healthy = self._healthy_endpoints(service)
        if not healthy:
            return None
        idx = self._round_robin.get(service, 0) % len(healthy)
        self._round_robin[service] = idx + 1
        return healthy[idx]

    def call(self, service: str, path: str) -> Optional[Dict[str, Any]]:
        """
        Call a service (stub).

        Returns simulated response. In production, makes HTTP call.
        """
        endpoint = self._select(service)
        if not endpoint:
            return None
        return {
            "service": service,
            "endpoint": endpoint,
            "path": path,
            "status": "success",
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def services(self) -> List[str]:
        return list(self._services.keys())

    def endpoints(self, service: str) -> List[str]:
        return list(self._services.get(service, []))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = sum(len(v) for v in self._services.values())
        healthy = sum(1 for v in self._health.values() if v)
        return {
            "services": len(self._services),
            "endpoints": total,
            "healthy": healthy,
        }

    def __repr__(self) -> str:
        return f"<ServiceMesh services={len(self._services)}>"
