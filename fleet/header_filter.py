"""HTTP header filtering and normalization.

Filters, normalizes, and validates HTTP headers. Supports allow/deny
lists, header transformation, and required header checking. Used for
fleet API gateway header processing, security filtering, and request
normalization.

Usage:
    filter = HeaderFilter()
    filter.allow("content-type")
    filter.deny("x-internal-token")
    filter.require("authorization")
    result = filter.process({"content-type": "json", "x-internal-token": "secret"})
    # result: {"content-type": "json", "authorization": "missing"}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class HeaderFilter:
    """
    HTTP header filter with allow/deny/require rules.
    """

    def __init__(self):
        self._allowed: Optional[List[str]] = None
        self._denied: List[str] = []
        self._required: List[str] = []
        self._transforms: Dict[str, callable] = {}

    # ------------------------------------------------------------------
    # Rule configuration
    # ------------------------------------------------------------------

    def allow(self, header: str) -> None:
        """Add to allow list (if set, only allowed headers pass)."""
        if self._allowed is None:
            self._allowed = []
        self._allowed.append(header.lower())

    def deny(self, header: str) -> None:
        """Add to deny list."""
        self._denied.append(header.lower())

    def require(self, header: str) -> None:
        """Add to required headers list."""
        self._required.append(header.lower())

    def transform(self, header: str, fn: callable) -> None:
        """
        Register a transformation function for a header.

        :param header: Header name.
        :param fn: Function(header_value) -> new_value.
        """
        self._transforms[header.lower()] = fn

    def clear_rules(self) -> None:
        """Clear all rules."""
        self._allowed = None
        self._denied.clear()
        self._required.clear()
        self._transforms.clear()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Process headers through filter rules.

        :param headers: Input headers dict.
        :returns: Filtered headers with validation results.
        """
        result = {}
        missing = []

        # Check required headers
        for req in self._required:
            if req not in [h.lower() for h in headers.keys()]:
                missing.append(req)

        # Filter headers
        for name, value in headers.items():
            name_lower = name.lower()

            # Check deny list
            if name_lower in self._denied:
                continue

            # Check allow list
            if self._allowed is not None and name_lower not in self._allowed:
                continue

            # Apply transformation
            if name_lower in self._transforms:
                value = self._transforms[name_lower](value)

            result[name] = value

        # Add missing required headers marker
        if missing:
            result["_missing_required"] = missing

        return result

    def validate(self, headers: Dict[str, str]) -> bool:
        """
        Validate headers against requirements.

        :returns: True if all required headers present.
        """
        header_keys = [h.lower() for h in headers.keys()]
        return all(req in header_keys for req in self._required)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def rules(self) -> Dict[str, List[str]]:
        return {
            "allowed": list(self._allowed) if self._allowed else [],
            "denied": list(self._denied),
            "required": list(self._required),
            "transforms": list(self._transforms.keys()),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "allowed_count": len(self._allowed) if self._allowed else 0,
            "denied_count": len(self._denied),
            "required_count": len(self._required),
            "transform_count": len(self._transforms),
        }

    def __repr__(self) -> str:
        return f"<HeaderFilter allowed={len(self._allowed) if self._allowed else 0} denied={len(self._denied)} required={len(self._required)}>"
