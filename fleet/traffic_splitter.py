"""Traffic splitting for A/B testing and canary deployments.

Routes traffic to different variants based on percentage weights, user
hashing, or sticky sessions. Used for fleet A/B testing, canary
deployments, and gradual rollouts.

Usage:
    splitter = TrafficSplitter()
    splitter.add_variant("control", weight=80)
    splitter.add_variant("treatment", weight=20)
    variant = splitter.route(user_id="user-1")
    assert variant in ["control", "treatment"]
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


class TrafficSplitter:
    """
    Traffic splitter with weighted variants.

    :param sticky: If True, same user always routes to same variant.
    """

    def __init__(self, sticky: bool = True):
        self._sticky = sticky
        self._variants: Dict[str, int] = {}
        self._assignments: Dict[str, str] = {}  # user_id -> variant

    # ------------------------------------------------------------------
    # Variant management
    # ------------------------------------------------------------------

    def add_variant(self, name: str, weight: int) -> None:
        """
        Add a traffic variant.

        :param name: Variant name.
        :param weight: Traffic percentage (0-100).
        """
        self._variants[name] = weight

    def remove_variant(self, name: str) -> bool:
        """Remove a variant."""
        if name in self._variants:
            del self._variants[name]
            # Clean up assignments
            self._assignments = {
                k: v for k, v in self._assignments.items() if v != name
            }
            return True
        return False

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, user_id: Optional[str] = None) -> Optional[str]:
        """
        Route traffic to a variant.

        :param user_id: Optional user ID for sticky routing.
        :returns: Variant name or None if no variants.
        """
        if not self._variants:
            return None

        # Sticky assignment
        if self._sticky and user_id and user_id in self._assignments:
            assigned = self._assignments[user_id]
            if assigned in self._variants:
                return assigned

        # Hash-based selection
        if user_id:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            bucket = hash_val % 100
        else:
            import random

            bucket = random.randint(0, 99)

        cumulative = 0
        for variant, weight in self._variants.items():
            cumulative += weight
            if bucket < cumulative:
                if self._sticky and user_id:
                    self._assignments[user_id] = variant
                return variant

        # Fallback to last variant
        last = list(self._variants.keys())[-1]
        if self._sticky and user_id:
            self._assignments[user_id] = last
        return last

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def variants(self) -> List[str]:
        return list(self._variants.keys())

    def weights(self) -> Dict[str, int]:
        return dict(self._variants)

    def total_weight(self) -> int:
        return sum(self._variants.values())

    def get_assignment(self, user_id: str) -> Optional[str]:
        """Get sticky assignment for a user."""
        return self._assignments.get(user_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "variants": len(self._variants),
            "total_weight": self.total_weight(),
            "assignments": len(self._assignments),
            "sticky": self._sticky,
        }

    def __repr__(self) -> str:
        return f"<TrafficSplitter variants={len(self._variants)}>"
