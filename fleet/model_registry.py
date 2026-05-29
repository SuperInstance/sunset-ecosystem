"""
Model Registry

Versioned storage for trained models, breeding checkpoints, and artifacts.
Agents can publish models, discover them, and load specific versions.

Usage:
    from fleet.model_registry import ModelRegistry
    registry = ModelRegistry()
    registry.publish("breeder-v1", model_data, tags={"type": "breeder"})
    model = registry.load("breeder-v1", version="latest")
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelArtifact:
    """A versioned model artifact."""
    name: str
    version: str
    data: Any
    checksum: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    publisher: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "tags": self.tags,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "publisher": self.publisher,
        }


class ModelRegistry:
    """
    Versioned model registry for fleet artifacts.

    In production, this would use S3, GCS, or a model store.
    For now, maintains an in-memory registry with export capabilities.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.artifacts: Dict[str, Dict[str, ModelArtifact]] = {}  # name -> {version -> artifact}
        self.tags_index: Dict[str, List[str]] = {}  # tag -> [name:version]

    def _compute_checksum(self, data: Any) -> str:
        """Compute SHA-256 checksum of data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def publish(self, name: str, data: Any,
                tags: Optional[Dict[str, str]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                version: Optional[str] = None) -> ModelArtifact:
        """Publish a model artifact."""
        version = version or f"v{len(self.artifacts.get(name, {})) + 1}"
        checksum = self._compute_checksum(data)

        artifact = ModelArtifact(
            name=name,
            version=version,
            data=data,
            checksum=checksum,
            tags=tags or {},
            metadata=metadata or {},
            publisher=self.fleet_node_id,
        )

        if name not in self.artifacts:
            self.artifacts[name] = {}
        self.artifacts[name][version] = artifact

        # Index tags
        for tag_key, tag_value in (tags or {}).items():
            tag_str = f"{tag_key}:{tag_value}"
            self.tags_index.setdefault(tag_str, []).append(f"{name}:{version}")

        return artifact

    def load(self, name: str, version: str = "latest") -> Optional[Any]:
        """Load a model artifact."""
        if name not in self.artifacts:
            return None

        if version == "latest":
            versions = sorted(self.artifacts[name].keys())
            if not versions:
                return None
            version = versions[-1]

        artifact = self.artifacts[name].get(version)
        return artifact.data if artifact else None

    def get_metadata(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get metadata for an artifact."""
        if name not in self.artifacts:
            return None
        if version == "latest":
            versions = sorted(self.artifacts[name].keys())
            version = versions[-1] if versions else "latest"
        artifact = self.artifacts[name].get(version)
        return artifact.metadata if artifact else None

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model."""
        return sorted(self.artifacts.get(name, {}).keys())

    def search_by_tag(self, tag_key: str, tag_value: str) -> List[ModelArtifact]:
        """Search artifacts by tag."""
        tag_str = f"{tag_key}:{tag_value}"
        refs = self.tags_index.get(tag_str, [])
        results = []
        for ref in refs:
            name, version = ref.split(":", 1)
            artifact = self.artifacts.get(name, {}).get(version)
            if artifact:
                results.append(artifact)
        return results

    def verify_checksum(self, name: str, version: str) -> bool:
        """Verify artifact checksum against recomputed value."""
        artifact = self.artifacts.get(name, {}).get(version)
        if not artifact:
            return False
        expected = self._compute_checksum(artifact.data)
        return expected == artifact.checksum

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_artifacts = sum(len(v) for v in self.artifacts.values())
        return {
            "models": len(self.artifacts),
            "total_artifacts": total_artifacts,
            "tags": len(self.tags_index),
            "node": self.fleet_node_id,
        }

    def export_manifest(self) -> str:
        """Export registry manifest as JSON."""
        manifest = {
            "node": self.fleet_node_id,
            "models": {},
        }
        for name, versions in self.artifacts.items():
            manifest["models"][name] = {
                "versions": [v.to_dict() for v in versions.values()],
            }
        return json.dumps(manifest, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.get_stats(),
            "models": list(self.artifacts.keys()),
        }
