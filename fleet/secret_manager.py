from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Secret:
    """A stored secret."""
    name: str
    value: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired(),
        }


class SecretManager:
    """
    Credential management for fleet secrets.

    Stores secrets with optional expiration and metadata.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self._secrets: Dict[str, Secret] = {}
        self._access_log: List[Dict[str, Any]] = []

    def set(self, name: str, value: str,
            metadata: Optional[Dict[str, Any]] = None,
            ttl_seconds: Optional[float] = None) -> Secret:
        """Store a secret."""
        secret = Secret(
            name=name,
            value=value,
            created_at=time.time(),
            metadata=metadata or {},
            expires_at=time.time() + ttl_seconds if ttl_seconds else None,
        )
        self._secrets[name] = secret
        return secret

    def get(self, name: str) -> Optional[str]:
        """Retrieve a secret value."""
        secret = self._secrets.get(name)
        if not secret:
            return None
        if secret.is_expired():
            del self._secrets[name]
            return None
        self._access_log.append({
            "name": name,
            "timestamp": time.time(),
            "action": "read",
        })
        return secret.value

    def delete(self, name: str) -> bool:
        """Delete a secret."""
        if name not in self._secrets:
            return False
        del self._secrets[name]
        self._access_log.append({
            "name": name,
            "timestamp": time.time(),
            "action": "delete",
        })
        return True

    def list_names(self) -> List[str]:
        """List all secret names."""
        return list(self._secrets.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a secret without exposing value."""
        secret = self._secrets.get(name)
        if not secret or secret.is_expired():
            return None
        return secret.metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get secret manager statistics."""
        expired = sum(1 for s in self._secrets.values() if s.is_expired())
        return {
            "total_secrets": len(self._secrets),
            "expired_secrets": expired,
            "access_count": len(self._access_log),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
        }
