"""quota_manager.py — Resource quotas per tenant/user.

Provides:
1. Per-tenant quota tracking (CPU, memory, API calls, etc.)
2. Quota enforcement (reject over-quota requests)
3. Windowed quota (hourly, daily, monthly)
4. Quota burst allowance
5. Usage reporting and alerts

Usage:
    qm = QuotaManager()
    qm.set_quota("tenant-1", "api_calls", limit=1000, window=3600)
    qm.record_usage("tenant-1", "api_calls", 50)
    if qm.check("tenant-1", "api_calls", 10):
        process_request()
"""
from __future__ import annotations

__all__ = [
    "QuotaManager",
    "Quota",
    "QuotaExceeded",
]

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """Raised when a quota would be exceeded."""


@dataclass
class Quota:
    """Quota definition for a resource."""
    resource: str
    limit: float
    window: float  # seconds
    burst: float = 0.0  # Extra allowance above limit
    used: float = 0.0
    window_start: float = 0.0


class QuotaManager:
    """Resource quota tracking and enforcement per tenant."""

    def __init__(self) -> None:
        self._quotas: dict[str, dict[str, Quota]] = {}  # tenant -> resource -> Quota
        self._alerts: list[dict[str, Any]] = []

    def set_quota(
        self,
        tenant: str,
        resource: str,
        limit: float,
        window: float = 3600.0,
        burst: float = 0.0,
    ) -> None:
        """Set a quota for a tenant."""
        tenant_quotas = self._quotas.setdefault(tenant, {})
        tenant_quotas[resource] = Quota(
            resource=resource,
            limit=limit,
            window=window,
            burst=burst,
            used=0.0,
            window_start=time.time(),
        )

    def get_quota(self, tenant: str, resource: str) -> Quota | None:
        """Get quota info for a tenant/resource."""
        return self._quotas.get(tenant, {}).get(resource)

    def check(self, tenant: str, resource: str, amount: float = 1.0) -> bool:
        """Check if usage is within quota (does not consume)."""
        quota = self._get_active_quota(tenant, resource)
        if quota is None:
            return True
        effective_limit = quota.limit + quota.burst
        return quota.used + amount <= effective_limit

    def record_usage(self, tenant: str, resource: str, amount: float = 1.0) -> bool:
        """Record usage, returns True if within quota, False if exceeded."""
        quota = self._get_active_quota(tenant, resource)
        if quota is None:
            return True

        quota.used += amount
        effective_limit = quota.limit + quota.burst

        if quota.used > effective_limit:
            self._alerts.append({
                "tenant": tenant,
                "resource": resource,
                "used": quota.used,
                "limit": quota.limit,
                "timestamp": time.time(),
            })
            logger.warning(
                f"Quota exceeded for {tenant}/{resource}: {quota.used} > {effective_limit}"
            )
            return False
        return True

    def require(self, tenant: str, resource: str, amount: float = 1.0) -> None:
        """Require quota, raise QuotaExceeded if over."""
        if not self.record_usage(tenant, resource, amount):
            raise QuotaExceeded(
                f"Quota exceeded for {tenant}/{resource}"
            )

    def _get_active_quota(self, tenant: str, resource: str) -> Quota | None:
        """Get quota, resetting window if expired."""
        quota = self._quotas.get(tenant, {}).get(resource)
        if quota is None:
            return None
        now = time.time()
        if now - quota.window_start >= quota.window:
            quota.used = 0.0
            quota.window_start = now
        return quota

    def usage(self, tenant: str, resource: str) -> float:
        """Get current usage for a tenant/resource."""
        quota = self._get_active_quota(tenant, resource)
        return quota.used if quota else 0.0

    def remaining(self, tenant: str, resource: str) -> float:
        """Get remaining quota."""
        quota = self._get_active_quota(tenant, resource)
        if quota is None:
            return float("inf")
        return max(0.0, quota.limit + quota.burst - quota.used)

    def reset(self, tenant: str, resource: str) -> bool:
        """Manually reset a quota window."""
        quota = self._quotas.get(tenant, {}).get(resource)
        if quota:
            quota.used = 0.0
            quota.window_start = time.time()
            return True
        return False

    def tenants(self) -> list[str]:
        """List all tenants."""
        return list(self._quotas.keys())

    def resources(self, tenant: str) -> list[str]:
        """List resources for a tenant."""
        return list(self._quotas.get(tenant, {}).keys())

    def stats(self) -> dict[str, Any]:
        """Quota statistics."""
        total_quotas = sum(len(v) for v in self._quotas.values())
        return {
            "tenants": len(self._quotas),
            "total_quotas": total_quotas,
            "alerts": len(self._alerts),
        }

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._quotas.values())
        return f"QuotaManager(tenants={len(self._quotas)}, quotas={total})"
