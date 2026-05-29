"""Schema versioning and compatibility checking.

Manages data schemas with versioning, backward/forward compatibility
checks, and evolution tracking. Used for fleet data contracts, API
schema governance, and migration safety.

Usage:
    registry = SchemaRegistry()
    registry.register("user", {"type": "object", "properties": {"name": {"type": "string"}}})
    assert registry.is_compatible("user", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}})
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class SchemaRegistry:
    """
    Schema registry with compatibility checking.
    """

    def __init__(self):
        self._schemas: Dict[str, List[Dict[str, Any]]] = {}
        self._compatibility: Dict[str, str] = {}  # schema_name -> "backward", "forward", "full", "none"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, schema: Dict[str, Any]) -> int:
        """
        Register a new schema version.

        :param name: Schema name.
        :param schema: Schema definition (dict).
        :returns: Version number assigned.
        """
        if name not in self._schemas:
            self._schemas[name] = []
        self._schemas[name].append(schema)
        return len(self._schemas[name])

    def set_compatibility(self, name: str, mode: str) -> None:
        """
        Set compatibility mode for a schema.

        :param mode: "backward", "forward", "full", "none".
        """
        self._compatibility[name] = mode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a schema by name and version.

        :param version: 1-based version number (latest if None).
        :returns: Schema dict or None.
        """
        versions = self._schemas.get(name, [])
        if not versions:
            return None
        if version is None:
            return versions[-1]
        if version < 1 or version > len(versions):
            return None
        return versions[version - 1]

    def latest_version(self, name: str) -> int:
        """Get latest version number."""
        return len(self._schemas.get(name, []))

    # ------------------------------------------------------------------
    # Compatibility checking (simplified)
    # ------------------------------------------------------------------

    def is_compatible(self, name: str, new_schema: Dict[str, Any]) -> bool:
        """
        Check if a new schema is compatible with the latest version.

        :param name: Schema name.
        :param new_schema: New schema to check.
        :returns: True if compatible.
        """
        latest = self.get(name)
        if latest is None:
            return True

        mode = self._compatibility.get(name, "backward")
        if mode == "none":
            return True
        if mode == "backward":
            return self._check_backward_compatible(latest, new_schema)
        if mode == "forward":
            return self._check_forward_compatible(latest, new_schema)
        if mode == "full":
            return (self._check_backward_compatible(latest, new_schema) and
                    self._check_forward_compatible(latest, new_schema))
        return True

    def _check_backward_compatible(self, old: Dict[str, Any], new: Dict[str, Any]) -> bool:
        """Check backward compatibility (new readers can read old data)."""
        # Simplified: new schema can have additional optional properties
        old_props = old.get("properties", {})
        new_props = new.get("properties", {})
        # All old required properties must exist in new schema
        old_required = set(old.get("required", []))
        new_required = set(new.get("required", []))
        if old_required - set(new_props.keys()):
            return False
        # New required fields must not break old data
        if new_required - old_required:
            # New required fields must be in old optional fields
            if new_required - set(old_props.keys()):
                return False
        return True

    def _check_forward_compatible(self, old: Dict[str, Any], new: Dict[str, Any]) -> bool:
        """Check forward compatibility (old readers can read new data)."""
        # Simplified: no new required fields in old schema
        old_props = old.get("properties", {})
        new_props = new.get("properties", {})
        new_required = set(new.get("required", []))
        old_required = set(old.get("required", []))
        # New required fields must already be in old required
        if new_required - old_required:
            return False
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def versions(self, name: str) -> int:
        return len(self._schemas.get(name, []))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total_versions = sum(len(v) for v in self._schemas.values())
        return {
            "schemas": len(self._schemas),
            "total_versions": total_versions,
        }

    def __repr__(self) -> str:
        return f"<SchemaRegistry schemas={len(self._schemas)}>"
