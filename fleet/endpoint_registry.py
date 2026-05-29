from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ServiceEndpoint:
    """A service endpoint in the fleet."""
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    health_status: str = "unknown"
    last_seen: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "health_status": self.health_status,
            "last_seen": self.last_seen,
            "metadata": self.metadata,
        }

    def url(self) -> str:
        """Get the full URL for this endpoint."""
        return f"{self.protocol}://{self.host}:{self.port}"


class EndpointRegistry:
    """
    Registry for fleet service endpoints.

    Tracks all service endpoints and their health status.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._endpoints: Dict[str, ServiceEndpoint] = {}

    def register(self, service_name: str, host: str, port: int,
                 protocol: str = "http", metadata: Optional[Dict[str, Any]] = None) -> ServiceEndpoint:
        """Register a service endpoint."""
        endpoint = ServiceEndpoint(
            service_name=service_name,
            host=host,
            port=port,
            protocol=protocol,
            health_status="healthy",
            last_seen=time.time(),
            metadata=metadata or {},
        )
        self._endpoints[service_name] = endpoint
        return endpoint

    def unregister(self, service_name: str) -> bool:
        """Unregister a service endpoint."""
        if service_name not in self._endpoints:
            return False
        del self._endpoints[service_name]
        return True

    def get(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Get an endpoint by service name."""
        return self._endpoints.get(service_name)

    def get_all(self) -> List[ServiceEndpoint]:
        """Get all registered endpoints."""
        return list(self._endpoints.values())

    def get_healthy(self) -> List[ServiceEndpoint]:
        """Get all healthy endpoints."""
        return [e for e in self._endpoints.values() if e.health_status == "healthy"]

    def update_health(self, service_name: str, status: str) -> bool:
        """Update health status of an endpoint."""
        endpoint = self._endpoints.get(service_name)
        if not endpoint:
            return False
        endpoint.health_status = status
        endpoint.last_seen = time.time()
        return True

    def find_by_metadata(self, key: str, value: Any) -> List[ServiceEndpoint]:
        """Find endpoints by metadata key."""
        return [e for e in self._endpoints.values() if e.metadata.get(key) == value]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        statuses = {}
        for e in self._endpoints.values():
            s = e.health_status
            statuses[s] = statuses.get(s, 0) + 1
        return {
            "total": len(self._endpoints),
            "statuses": statuses,
        }

    def export_json(self) -> str:
        """Export registry as JSON."""
        return json.dumps({
            "node": self.fleet_node_id,
            "endpoints": [e.to_dict() for e in self._endpoints.values()],
        }, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
