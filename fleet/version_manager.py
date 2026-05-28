"""Semantic versioning manager with compatibility checks.

Tracks component versions, checks compatibility, and gates features by version.
Used for fleet-wide schema evolution and API compatibility.

Usage:
    vm = VersionManager()
    vm.register("breeder", "1.2.3")
    vm.register("conductor", "2.0.0")
    assert vm.is_compatible("breeder", ">=1.0.0")
    assert vm.feature_enabled("breeder", "flux-gating", since="1.2.0")
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    @classmethod
    def parse(cls, s: str) -> "SemVer":
        m = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$",
            s,
        )
        if not m:
            raise ValueError(f"Invalid semver: {s}")
        return cls(
            major=int(m.group(1)),
            minor=int(m.group(2)),
            patch=int(m.group(3)),
            prerelease=m.group(4) or "",
            build=m.group(5) or "",
        )

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            s += f"-{self.prerelease}"
        if self.build:
            s += f"+{self.build}"
        return s

    def __lt__(self, other: "SemVer") -> bool:
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # Pre-release versions are lower than release versions
        if self.prerelease and not other.prerelease:
            return True
        if other.prerelease and not self.prerelease:
            return False
        return self.prerelease < other.prerelease

    def __le__(self, other: "SemVer") -> bool:
        return self == other or self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return (
            self.major,
            self.minor,
            self.patch,
            self.prerelease,
        ) == (other.major, other.minor, other.patch, other.prerelease)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))


class VersionManager:
    """
    Tracks component versions and compatibility.

    :param features: Dict mapping feature names to (component, since_version) tuples.
    """

    def __init__(self, features: Optional[Dict[str, Tuple[str, str]]] = None):
        self._versions: Dict[str, SemVer] = {}
        self._features: Dict[str, Tuple[str, SemVer]] = {}
        if features:
            for feat, (comp, since) in features.items():
                self._features[feat] = (comp, SemVer.parse(since))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, component: str, version: str) -> None:
        self._versions[component] = SemVer.parse(version)

    def get(self, component: str) -> Optional[SemVer]:
        return self._versions.get(component)

    def list_components(self) -> List[str]:
        return list(self._versions.keys())

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def is_compatible(
        self,
        component: str,
        constraint: str,
    ) -> bool:
        """
        Check if *component* version satisfies *constraint*.

        Supports: >=x.y.z, >x.y.z, <=x.y.z, <x.y.z, =x.y.z, ^x.y.z, ~x.y.z
        """
        ver = self._versions.get(component)
        if ver is None:
            return False
        return _check_constraint(ver, constraint)

    def require(self, component: str, constraint: str) -> None:
        """Raise if component version does not satisfy constraint."""
        if not self.is_compatible(component, constraint):
            ver = self._versions.get(component)
            raise ValueError(
                f"Component {component} version {ver} does not satisfy {constraint}"
            )

    # ------------------------------------------------------------------
    # Feature gating
    # ------------------------------------------------------------------

    def feature_enabled(self, component: str, feature: str) -> bool:
        """Check if a feature is enabled for a component."""
        feat = self._features.get(feature)
        if feat is None:
            return False
        comp, since = feat
        if comp != component:
            return False
        ver = self._versions.get(component)
        if ver is None:
            return False
        return ver >= since

    def register_feature(self, feature: str, component: str, since: str) -> None:
        self._features[feature] = (component, SemVer.parse(since))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, str]:
        return {comp: str(ver) for comp, ver in self._versions.items()}

    def __repr__(self) -> str:
        return f"<VersionManager components={len(self._versions)}>"


# ------------------------------------------------------------------
# Constraint parsing
# ------------------------------------------------------------------

_OPS = {
    ">=": lambda v, c: v >= c,
    ">": lambda v, c: v > c,
    "<=": lambda v, c: v <= c,
    "<": lambda v, c: v < c,
    "=": lambda v, c: v == c,
    "==": lambda v, c: v == c,
    "^": lambda v, c: v >= c and v.major == c.major,
    "~": lambda v, c: v >= c and v.major == c.major and v.minor == c.minor,
}


def _check_constraint(ver: SemVer, constraint: str) -> bool:
    for op, fn in _OPS.items():
        if constraint.startswith(op):
            target = SemVer.parse(constraint[len(op):].strip())
            return fn(ver, target)
    # No operator — exact match
    return ver == SemVer.parse(constraint)
