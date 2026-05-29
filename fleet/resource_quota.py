"""Resource quota management and enforcement.

Manages resource quotas (CPU, memory, storage, requests) per tenant or
service. Tracks usage, enforces limits, and supports burst allowances.
Used for fleet multi-tenancy, fair scheduling, and capacity protection.

Usage:
    quota = ResourceQuota(tenant="team-a")
    quota.set_limit("cpu", 100)
    quota.set_limit("memory", 1024)
    assert quota.request("cpu", 50) is True
    assert quota.request("cpu", 60) is False  # Would exceed
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ResourceQuota:
    """
    Resource quota manager per tenant.

    :param tenant: Tenant identifier.
    :param burst_ratio: Allow burst up to limit * burst_ratio.
    """

    def __init__(self, tenant: str = "default", burst_ratio: float = 1.2):
        self.tenant = tenant
        self._burst_ratio = burst_ratio
        self._limits: Dict[str, float] = {}
        self._used: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Limit management
    # ------------------------------------------------------------------

    def set_limit(self, resource: str, limit: float) -> None:
        """Set resource limit."""
        self._limits[resource] = limit

    def get_limit(self, resource: str) -> Optional[float]:
        """Get resource limit."""
        return self._limits.get(resource)

    def remove_limit(self, resource: str) -> bool:
        """Remove resource limit."""
        if resource in self._limits:
            del self._limits[resource]
            self._used.pop(resource, None)
            return True
        return False

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def request(self, resource: str, amount: float) -> bool:
        """
        Request resource allocation.

        :param resource: Resource type.
        :param amount: Amount requested.
        :returns: True if allocated, False if would exceed limit.
        """
        limit = self._limits.get(resource)
        if limit is None:
            # No limit set, track usage anyway
            self._used[resource] = self._used.get(resource, 0) + amount
            return True
        used = self._used.get(resource, 0)
        burst_limit = limit * self._burst_ratio
        if used + amount > burst_limit:
            return False
        self._used[resource] = used + amount
        return True

    def release(self, resource: str, amount: float) -> None:
        """Release allocated resource."""
        used = self._used.get(resource, 0)
        self._used[resource] = max(0, used - amount)

    def usage(self, resource: str) -> float:
        """Get current usage for a resource."""
        return self._used.get(resource, 0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def available(self, resource: str) -> float:
        """Get available resource capacity."""
        limit = self._limits.get(resource)
        if limit is None:
            return float("inf")
        return max(0, limit - self._used.get(resource, 0))

    def resources(self) -> List[str]:
        """List all resources with limits."""
        return list(self._limits.keys())

    def is_exceeded(self, resource: str) -> bool:
        """Check if resource is over limit (not burst)."""
        limit = self._limits.get(resource)
        if limit is None:
            return False
        return self._used.get(resource, 0) > limit

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "tenant": self.tenant,
            "resources": len(self._limits),
            "limits": dict(self._limits),
            "used": dict(self._used),
        }

    def __repr__(self) -> str:
        return f"<ResourceQuota tenant={self.tenant} resources={len(self._limits)}>"
